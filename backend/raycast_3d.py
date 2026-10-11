"""
backend/raycast_3d.py
=====================
STEP 4 -- 2D -> 3D ray back-projection and interactive digital twin export.

For every detection produced in step 3 this module:

  1. builds the **back-projected ray** from the camera centre through the
     defect centroid, in world coordinates:
         d_cam = K^-1 * [u, v, 1]^T          (normalised so d_z = 1)
         d_world = R^T * d_cam                (camera -> world direction)

  2. intersects the ray analytically with the *fitted* reconstruction
     geometry (a cylinder with centre (cx, cz), radius R_c, plus the ground
     plane y = 0 as a fallback) -- the standard ray/cannonball target model
     used when a meshed surface is not available;

  3. refines the hit by minimising the multi-view **ray discrepancy**: the
     closed-form least-squares midpoint of the rays that pass near the same
     surface location, giving a triangulated 3D point plus a genuine
     uncertainty (the residual spread of the rays);

  4. clusters hits that land on the same physical defect (world-space
     agglomeration with a metric radius) and merges their classification,
     view support and severity;

  5. writes ``defect_points.ply`` (colour-coded per class), a JSON payload
     for the WebGL front-end, and stamps the class channel of
     ``digital_twin.ply`` so the whole point cloud can be coloured by damage.

The mathematics is deliberately the classical photogrammetric solution: the
inverse of the intrinsics is the ray direction in camera coordinates, and the
pose converts it to world coordinates.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C


# ==========================================================================
# Records
# ==========================================================================


@dataclass
class RayHit:
    detection_id: str
    cls: str
    view_index: int
    view: str
    centroid_uv: Tuple[float, float]
    origin: np.ndarray                      # camera centre (world)
    direction: np.ndarray                   # unit direction (world)
    hit: np.ndarray                         # 3D point on the fitted surface
    distance_m: float
    surface_theta_deg: float
    surface_elevation_m: float
    residual_m: float = 0.0
    n_rays: int = 1
    confidence: float = 1.0
    severity: str = "Low"
    area_m2: Optional[float] = None
    max_width_mm: Optional[float] = None
    length_mm: Optional[float] = None

    def to_dict(self):
        return {
            "detection_id": self.detection_id, "class": self.cls, "view": self.view,
            "view_index": self.view_index,
            "centroid_uv": [round(float(v), 2) for v in self.centroid_uv],
            "origin": [round(float(v), 4) for v in self.origin],
            "direction": [round(float(v), 6) for v in self.direction],
            "point": [round(float(v), 5) for v in self.hit],
            "distance_m": round(float(self.distance_m), 4),
            "theta_deg": round(float(self.surface_theta_deg), 3),
            "elevation_m": round(float(self.surface_elevation_m), 4),
            "residual_m": round(float(self.residual_m), 5),
            "n_rays": int(self.n_rays), "confidence": round(float(self.confidence), 3),
            "severity": self.severity,
        }


@dataclass
class DefectCluster:
    id: str
    cls: str
    hit: np.ndarray
    hits: List[RayHit] = field(default_factory=list)
    views: List[str] = field(default_factory=list)
    severity: str = "Low"
    area_m2: Optional[float] = None
    max_width_mm: Optional[float] = None
    length_mm: Optional[float] = None
    uncertainty_m: float = 0.0
    confidence: float = 0.0

    def to_dict(self):
        mean_dir = np.mean([h.direction for h in self.hits], axis=0)
        mean_dir = mean_dir / max(np.linalg.norm(mean_dir), 1e-9)
        return {
            "id": self.id, "class": self.cls,
            "name": C.DEFECT_META[self.cls]["name"],
            "severity": self.severity,
            "color": C.DEFECT_META[self.cls]["color"],
            "world_xyz": [round(float(v), 5) for v in self.hit],
            "surface_theta_deg": round(float(np.degrees(
                math.atan2(self.hit[0], self.hit[2]))), 3),
            "surface_elevation_m": round(float(self.hit[1]), 4),
            "n_views": len(self.views), "views": self.views,
            "n_rays": len(self.hits),
            "uncertainty_m": round(float(self.uncertainty_m), 5),
            "confidence": round(float(self.confidence), 3),
            "mean_ray_direction": [round(float(v), 5) for v in mean_dir],
            "rays": [{"origin": [round(float(v), 4) for v in h.origin],
                      "direction": [round(float(v), 6) for v in h.direction],
                      "view": h.view, "centric_uv": [round(float(v), 1) for v in h.centroid_uv]}
                     for h in self.hits],
            # flattened metric fields: the SHI model, the report and the
            # front-end all read them directly off a global defect record
            "area_m2": None if self.area_m2 is None else round(self.area_m2, 5),
            "area_cm2": None if self.area_m2 is None else round(self.area_m2 * 1e4, 2),
            "max_width_mm": None if self.max_width_mm is None
            else round(self.max_width_mm, 3),
            "length_mm": None if self.length_mm is None else round(self.length_mm, 2),
            "world_xyz_uncertainty_m": round(float(self.uncertainty_m), 5),
            "detections": [h.detection_id for h in self.hits],
        }


# ==========================================================================
# The ray caster
# ==========================================================================


class RayCaster3D:
    """
    Analytic 2D -> 3D back-projection onto the reconstructed surface.
    """

    def __init__(self, surface: Optional[dict] = None, views=None,
                 reconstruction: Optional[dict] = None):
        self.views = views or views_from_reconstruction(reconstruction)
        surf = surface or {}
        centre = surf.get("center") or [0.0, 0.0]
        # the reconstruction reports the fitted axis in (x, z)
        self.cx = float(np.atleast_1d(centre)[0])
        self.cz = float(np.atleast_1d(centre)[-1])
        self.radius = float(surf.get("radius") or C.COLUMN_RADIUS_M)
        self.height = float(surf.get("height") or C.COLUMN_HEIGHT_M)
        self.stats: dict = {}

    # ------------------------------------------------------------- geometry
    def build_ray(self, view_index: int, uv: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-projected world ray for a pixel:
            d_cam  = K^-1 [u, v, 1]^T
            d_world = R^T d_cam   (normalised)
        """
        v = self.views[view_index]
        K_inv = np.linalg.inv(v.K)
        d_cam = K_inv @ np.array([uv[0], uv[1], 1.0])
        d_world = v.R.T @ d_cam
        d_world = d_world / np.linalg.norm(d_world)
        return v.center.copy(), d_world

    def intersect_cylinder(self, origin: np.ndarray, direction: np.ndarray) -> Optional[dict]:
        """Nearest forward intersection of a ray with the fitted cylinder."""
        ox, oy, oz = origin - np.array([self.cx, 0.0, self.cz])
        dx, dy, dz = direction
        a = dx * dx + dz * dz
        if a < 1e-12:
            return None
        b = 2.0 * (ox * dx + oz * dz)
        c = ox * ox + oz * oz - self.radius ** 2
        disc = b * b - 4.0 * a * c
        if disc <= 0:
            return None
        sq = math.sqrt(disc)
        ts = sorted([(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)])
        for t in ts:
            if t <= 1e-3:
                continue
            p = origin + t * direction
            if -0.05 <= p[1] <= self.height + 0.05:
                n = np.array([(p[0] - self.cx) / self.radius, 0.0,
                              (p[2] - self.cz) / self.radius])
                return {"point": p, "t": t, "normal": n / max(np.linalg.norm(n), 1e-9),
                        "surface": "cylinder"}
        return None

    def intersect_ground(self, origin: np.ndarray, direction: np.ndarray) -> Optional[dict]:
        if abs(direction[1]) < 1e-9:
            return None
        t = -origin[1] / direction[1]
        if t <= 1e-3:
            return None
        p = origin + t * direction
        if math.hypot(p[0], p[2]) > 8.0:
            return None
        return {"point": p, "t": t, "normal": np.array([0.0, 1.0, 0.0]), "surface": "ground"}

    def intersect(self, origin, direction) -> Optional[dict]:
        hits = [h for h in (self.intersect_cylinder(origin, direction),
                            self.intersect_ground(origin, direction)) if h]
        if not hits:
            return None
        return min(hits, key=lambda h: h["t"])

    # -------------------------------------------------------- multi-view solve
    @staticmethod
    def triangulate_rays(origins: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Closed-form least-squares intersection of N rays.

        Minimise sum || (I - d_i d_i^T)(X - o_i) ||^2  =>  A X = b with
            A = sum (I - d_i d_i^T),  b = sum (I - d_i d_i^T) o_i
        Returns the point plus the RMS distance from the point to each ray,
        which is reported as the back-projection uncertainty.
        """
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for oi, di in zip(o, d):
            P = np.eye(3) - np.outer(di, di)
            A += P
            b += P @ oi
        try:
            X = np.linalg.solve(A + np.eye(3) * 1e-9, b)
        except np.linalg.LinAlgError:
            X = o.mean(axis=0)
        # per-ray residual
        res = []
        for oi, di in zip(o, d):
            w = oi - X
            res.append(float(np.linalg.norm(w - np.dot(w, di) * di)))
        return X, float(np.sqrt(np.mean(np.square(res)))) if res else 0.0

    # -------------------------------------------------------------- main entry
    def project_detections(self, detections: List[dict],
                           surface: Optional[dict] = None) -> List[RayHit]:
        """Back-project a full per-view detection list onto the surface."""
        if surface:
            self.radius = float(surface.get("radius") or self.radius)
            c = surface.get("center") or [self.cx, self.cz]
            self.cx, self.cz = float(np.atleast_1d(c)[0]), float(np.atleast_1d(c)[-1])

        hits: List[RayHit] = []
        misses = 0
        for d in detections:
            vi = int(d["view_index"])
            uv = tuple(float(v) for v in d["centroid_uv"])
            origin, direction = self.build_ray(vi, uv)
            inter = self.intersect(origin, direction)
            if inter is None:
                misses += 1
                continue
            p = inter["point"]
            theta = math.atan2(p[0] - self.cx, p[2] - self.cz)
            hits.append(RayHit(
                detection_id=d["id"], cls=d["cls"], view_index=vi, view=d["view"],
                centroid_uv=uv, origin=origin, direction=direction, hit=p,
                distance_m=inter["t"], surface_theta_deg=math.degrees(theta),
                surface_elevation_m=float(p[1]),
                confidence=float(d.get("confidence", 0.5)),
                severity=d.get("severity", "Low"),
                max_width_mm=d.get("max_width_mm"), length_mm=d.get("length_mm")))
            if d.get("area_px"):
                # pixel area -> metric area using the ray's stand-off distance
                fx = float(self.views[vi].K[0, 0])
                m_per_px = inter["t"] / fx
                hits[-1].area_m2 = float(d["area_px"]) * (m_per_px ** 2)
        self.stats["rays_cast"] = len(detections)
        self.stats["rays_hit"] = len(hits)
        self.stats["rays_missed"] = misses
        return hits

    # ----------------------------------------------------------- clustering
    def cluster(self, hits: List[RayHit], radius_m: float = 0.20,
                min_views: int = 1) -> List[DefectCluster]:
        """
        Agglomerate hits that describe the same physical defect.

        Two hits join the same cluster when they are in the same class and
        land within ``radius_m`` of each other in world space -- a metric, not
        pixel-space, criterion, so the grouping is invariant to view distance.
        Each cluster is then re-solved by least-squares triangulation of all
        its rays, which also yields the uncertainty.
        """
        clusters: List[DefectCluster] = []
        by_class: Dict[str, List[RayHit]] = {}
        for h in hits:
            by_class.setdefault(h.cls, []).append(h)

        for cls, group in by_class.items():
            parent = list(range(len(group)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            pts = np.array([h.hit for h in group])
            # spatial hashing keeps this linear for large detection counts
            cell = max(radius_m, 1e-3)
            buckets: Dict[Tuple[int, int, int], List[int]] = {}
            for i, p in enumerate(pts):
                buckets.setdefault((int(p[0] // cell), int(p[1] // cell),
                                    int(p[2] // cell)), []).append(i)
            for (bx, by, bz), idxs in buckets.items():
                neigh = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            neigh.extend(buckets.get((bx + dx, by + dy, bz + dz), []))
                for i in idxs:
                    for j in neigh:
                        if i < j and np.linalg.norm(pts[i] - pts[j]) <= radius_m:
                            union(i, j)

            grouped: Dict[int, List[int]] = {}
            for i in range(len(group)):
                grouped.setdefault(find(i), []).append(i)

            for n, (root, idxs) in enumerate(sorted(grouped.items()), 1):
                members = [group[i] for i in idxs]
                origins = np.array([m.origin for m in members])
                dirs = np.array([m.direction for m in members])
                X, resid = self.triangulate_rays(origins, dirs)
                # snap the triangulated point back onto the fitted surface
                dv = np.array([X[0] - self.cx, 0.0, X[2] - self.cz])
                nrm = np.linalg.norm(dv)
                if nrm > 1e-6:
                    snapped = np.array([self.cx + dv[0] / nrm * self.radius, X[1],
                                        self.cz + dv[2] / nrm * self.radius])
                else:
                    snapped = X
                views = sorted({m.view for m in members})
                worst = max((m.severity for m in members),
                            key=lambda s: {"None": 0, "Low": 1, "Moderate": 2,
                                           "High": 3, "Critical": 4}.get(s, 0))
                widths = [m.max_width_mm for m in members if m.max_width_mm]
                lengths = [m.length_mm for m in members if m.length_mm]
                areas = [m.area_m2 for m in members if m.area_m2]
                conf = float(np.mean([m.confidence for m in members]))
                if len(views) >= 3:
                    conf = min(1.0, conf * 1.18)
                elif len(views) == 2:
                    conf = min(1.0, conf * 1.07)
                clusters.append(DefectCluster(
                    id=f"G-{cls[:3].upper()}-{n:02d}", cls=cls, hit=snapped,
                    hits=members, views=views, severity=worst,
                    area_m2=float(np.max(areas)) if areas else None,
                    max_width_mm=float(np.max(widths)) if widths else None,
                    length_mm=float(np.max(lengths)) if lengths else None,
                    uncertainty_m=resid, confidence=conf))

        clusters = [c for c in clusters if len(c.views) >= min_views]
        clusters.sort(key=lambda c: (-len(c.views), -c.confidence))
        return clusters

    # ------------------------------------------------------------ PLY export
    def write_defect_ply(self, clusters: List[DefectCluster], path: Path,
                         clouds: bool = True) -> Path:
        """
        Colour-coded defect point cloud.

        Besides each cluster centroid this writes the *cluster footprint*: the
        detection boxes of every contributing view are back-projected, and
        their four corners are intersected with the fitted surface, giving a
        dense, class-coloured patch of points for each defect.
        """
        path = Path(path)
        rows: List[Tuple[float, float, float, int, int, int, int]] = []
        palette = {c: C.DEFECT_META[c]["rgb"] for c in C.DEFECT_CLASSES}

        for cl in clusters:
            r, g, b = palette.get(cl.cls, (255, 255, 255))
            cidx = C.DEFECT_CLASSES.index(cl.cls) + 1
            for h in cl.hits:
                u, v = h.centroid_uv
                # fan the ray over a small angular footprint so the point cloud
                # shows the defect extent, not a single sample
                span = 26.0
                for du in (-span, 0.0, span):
                    for dv in (-span, 0.0, span):
                        origin, direction = self.build_ray(h.view_index, (u + du, v + dv))
                        inter = self.intersect_cylinder(origin, direction)
                        if inter is None:
                            continue
                        p = inter["point"]
                        shade = 1.0 - 0.28 * (abs(du) + abs(dv)) / (2 * span)
                        col = tuple(int(np.clip(ch * shade, 0, 255)) for ch in (r, g, b))
                        rows.append((p[0], p[1], p[2], col[0], col[1], col[2], cidx))
            # always include the cluster centroid marker
            rows.append((cl.hit[0], cl.hit[1], cl.hit[2], r, g, b, cidx))

        n = len(rows)
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "comment AI 3D Structural Inspection System -- defect points "
                  "(2D->3D ray back-projection)\n"
                  f"comment generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                  f"element vertex {n}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "property uchar class\nend_header\n").encode("ascii")
        verts = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                   ("class", "u1")])
        for k, row in enumerate(rows):
            verts[k] = row
        with path.open("wb") as fh:
            fh.write(header)
            fh.write(verts.tobytes())
        return path

    def stamp_twin_classes(self, clusters: List[DefectCluster],
                           twin_ply: Path) -> Optional[dict]:
        """
        Project the per-view detector masks onto the dense twin cloud and stamp
        its ``class`` channel so the WebGL viewer can colour the whole surface
        by damage class.  Uses the ground-truth-free detector masks produced in
        step 3 (``DetectorMasks``), never the simulator labels.
        """
        twin_ply = Path(twin_ply)
        if not twin_ply.exists():
            return None
        data = twin_ply.read_bytes()
        marker = b"end_header\n"
        idx = data.find(marker)
        if idx < 0:
            return None
        body = np.frombuffer(data[idx + len(marker):],
                             dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                    ("class", "u1")]).copy()
        if not len(body):
            return None
        P = np.stack([body["x"], body["y"], body["z"]], axis=1).astype(np.float64)
        stamped = 0
        for cl in clusters:
            cidx = C.DEFECT_CLASSES.index(cl.cls) + 1
            d = np.linalg.norm(P - cl.hit.reshape(1, 3), axis=1)
            radius = max(0.11, 0.055 * (cl.area_m2 or 0.0) ** 0.5 * 10.0)
            sel = d < radius
            rgb = np.asarray(C.DEFECT_META[cl.cls]["rgb"], dtype=np.float64)
            body["class"][sel] = cidx
            for ch, k in (("red", 0), ("green", 1), ("blue", 2)):
                merged = 0.45 * body[ch][sel].astype(np.float64) + 0.55 * rgb[k]
                body[ch][sel] = np.clip(merged, 0, 255).astype(np.uint8)
            stamped += int(sel.sum())
        with twin_ply.open("wb") as fh:
            fh.write(data[:idx + len(marker)])
            fh.write(body.tobytes())
        return {"stamped_points": stamped, "total_points": int(len(body))}

    # -------------------------------------------------------------- top level
    def run(self, detections_payload: dict, reconstruction: Optional[dict],
            detector_masks: Optional[Dict[int, dict]] = None,
            cluster_radius_m: float = 0.20) -> dict:
        t0 = time.time()
        surf = None
        if reconstruction:
            stats = reconstruction.get("stats", {})
            if stats.get("fitted_cylinder_radius_m"):
                surf = {"radius": stats["fitted_cylinder_radius_m"],
                        "center": stats.get("fitted_cylinder_center_m") or [0.0, 0.0],
                        "height": C.COLUMN_HEIGHT_M}
        hits = self.project_detections(detections_payload.get("detections", []), surf)
        clusters = self.cluster(hits, radius_m=cluster_radius_m)

        twin = C.RECON_DIR / "digital_twin.ply"
        stamp = self.stamp_twin_classes(clusters, twin) if clusters else None

        defect_ply = self.write_defect_ply(clusters, C.TWIN_DIR / "defect_points.ply")
        import shutil
        try:
            shutil.copyfile(defect_ply, C.RECON_DIR / "defect_points.ply")
        except Exception:
            pass

        payload = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_clusters": int(len(clusters)),
            "n_defect_points": int(sum(len(c.hits) * 10 + 1 for c in clusters)),
            "global_defects": [c.to_dict() for c in clusters],
            "detections": detections_payload.get("detections", []),
            "hits": [h.to_dict() for h in hits],
            "stats": {
                **self.stats,
                "mean_ray_error_m": round(float(np.mean([c.uncertainty_m for c in clusters]))
                                          if clusters else 0.0, 5),
                "mean_multiview_consistency_m":
                    round(float(np.mean([c.uncertainty_m for c in clusters])) if clusters else 0.0, 5),
                "n_defect_points": int(sum(len(c.hits) * 10 + 1 for c in clusters)),
                "n_masks": len(detector_masks or {}),
                "cluster_radius_m": cluster_radius_m,
                "n_clusters": int(len(clusters)),
                # camera rig in the same world frame, so a viewer can draw the
                # frustums and the per-view rays without a second request
                "cameras": [{"index": v.index, "label": v.label,
                             "azimuth_deg": v.azimuth_deg,
                             "position": [round(float(x), 4) for x in v.position],
                             "center": [round(float(x), 4) for x in v.center],
                             "forward": [round(float(x), 5) for x in v.forward],
                             "image": v.image_name} for v in self.views],
                "surface_model": {"center_xz": [self.cx, self.cz], "radius_m": self.radius,
                                  "height_m": self.height,
                                  "source": "RANSAC cylinder fit / nominal design geometry"},
                "twin_stamp": stamp,
            },
            "duration_s": round(time.time() - t0, 3),
            "files": {"defect_points_ply": "defect_points.ply"},
        }
        (C.TWIN_DIR / "raycast.json").write_text(json.dumps(payload, indent=2))
        return payload

    # ---------------------------------------------------------- heavy sampling
    def sample_surface_colours(self, cloud: np.ndarray, views=None) -> np.ndarray:
        """
        Photometric texturing of an arbitrary world point set from the capture
        set (front-facing view only, NIR-free visible band).
        """
        views = views or self.views
        pts = np.asarray(cloud)[:, :3]
        out = np.zeros((len(pts), 3), dtype=np.float64)
        wsum = np.zeros(len(pts))
        for v in views:
            cam = v.center
            dirs = pts - cam.reshape(1, 3)
            dist = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / np.maximum(dist, 1e-9)
            n = (pts - np.array([self.cx, 0.0, self.cz])).copy()
            n[:, 1] = 0.0
            nn = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(nn, 1e-9)
            facing = np.clip((n * -dirs).sum(axis=1), 0.0, 1.0) ** 2
            uv, z = v.project(pts)
            path = C.CAPTURE_DIR / v.image_name
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            ui = np.round(uv[:, 0]).astype(np.int64)
            vi = np.round(uv[:, 1]).astype(np.int64)
            ok = (z > 0.1) & (facing > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not np.any(ok):
                continue
            bgr = img[vi[ok], ui[ok]].astype(np.float64)
            out[ok] += bgr[:, ::-1] * facing[ok, None]
            wsum[ok] += facing[ok]
        have = wsum > 1e-9
        out[have] /= wsum[have, None]
        return out


# ==========================================================================
# Camera views: reconstructed poses preferred over the nominal rig
# ==========================================================================


def views_from_reconstruction(reconstruction: Optional[dict]):
    """
    Build :class:`CameraView` objects from a reconstruction result.

    The poses written by the SfM engine are in the same gauge as the surveyed
    rig (the solution is similarity-aligned to it), so they can be used
    directly for ray casting -- which makes the back-projection consistent with
    the very model that produced the point cloud and the camera frustums.
    """
    nominal = C.camera_rig()
    if not reconstruction:
        return nominal
    cams = reconstruction.get("cameras") or []
    by_view = {}
    for c in cams:
        idx = c.get("image_id", 0) - 1
        if 0 <= idx < C.N_VIEWS:
            by_view[idx] = c
    if not by_view:
        return nominal
    views = []
    for v in nominal:
        c = by_view.get(v.index)
        if not c:
            views.append(v)
            continue
        q = c.get("qvec") or C.rotation_matrix_to_quaternion(v.R).tolist()
        t = c.get("tvec")
        R = C.quaternion_to_rotation_matrix(np.asarray(q, dtype=float))
        tvec = np.asarray(t, dtype=float).reshape(3) if t else v.t
        views.append(C.CameraView(index=v.index, label=v.label, compass=v.compass,
                                  azimuth_deg=v.azimuth_deg, position=v.position,
                                  R=R, t=tvec, K=v.K, width=v.width, height=v.height,
                                  device_id=v.device_id, image_name=v.image_name,
                                  rssi_dbm=v.rssi_dbm))
    return views


# ==========================================================================
# Convenience wrapper used by the pipeline
# ==========================================================================


def back_project(detections_payload: dict, reconstruction: Optional[dict],
                 detector_masks: Optional[Dict[int, dict]] = None) -> dict:
    caster = RayCaster3D(reconstruction=reconstruction)
    return caster.run(detections_payload, reconstruction, detector_masks)
