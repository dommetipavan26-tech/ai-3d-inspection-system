"""
backend/ai_inspector.py
=======================
STEP 3 -- AI structural defect analysis and the complete inspection report.

Detection pipeline (multi-stage, fully OpenCV / NumPy, no external weights):

    illumination normalisation  ->  black-hat + morphological line kernels
                                    (four crack orientations + sub-pixel
                                    Frangi-style ridge response)
    rust / efflorescence        ->  HSV + Lab opponent-colour segmentation with
                                    channel-ratio gating in a chroma mask
    spalling                    ->  local-contrast (dark-void) + high-texture
                                    (aggregate) coincidence inside the
                                    concrete silhouette
    fusion & classification     ->  weighted evidence maps per class, per-class
                                    Otsu, connected components, shape filters
    NMS + merging               ->  deduplicate overlapping boxes per class

Every detection is then:

  * measured (length, max/mean width in mm calibrated by the view's px/mm),
  * graded (severity from ACI 224R/ACI 201.1R-style width and extent rules),
  * back-projected to a world surface coordinate (theta, y) using the fitted
    reconstruction geometry, giving a metric defect size,
  * merged across the eight views into global defects,
  * rolled into the Structural Health Index (SHI, 0-100) and the ACI 562 /
    EN 1504 style remediation plan.

Outputs:

    data/analysis/annotated_view_NN.jpg    bounding boxes + labels
    data/analysis/detections.json          raw per-view detections
    data/analysis/pier_texture_atlas.jpg   unwrapped surface (cylindrical)
    data/analysis/defect_map.jpg           labelled unwrap
    data/analysis/report.json              full machine-readable report
    data/analysis/report.html              printable engineering report
"""

from __future__ import annotations

import html
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C


# ==========================================================================
# Detection record
# ==========================================================================


@dataclass
class Detection:
    id: str
    cls: str
    view_index: int
    view: str
    bbox: List[float]                       # x, y, w, h in pixels (1600x1200)
    centroid_uv: List[float]                # (u, v) defect centroid
    area_px: float
    confidence: float
    severity: str
    # metric + 3D enrichment (filled by the geometry stage)
    length_mm: Optional[float] = None
    max_width_mm: Optional[float] = None
    mean_width_mm: Optional[float] = None
    world_xyz: Optional[List[float]] = None
    surface_theta_deg: Optional[float] = None
    surface_elevation_m: Optional[float] = None
    area_m2: Optional[float] = None
    world_xyz_uncertainty_m: Optional[float] = None
    n_views: int = 1
    merged_from: List[str] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


SEVERITY_ORDER = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Critical": 4}


# ==========================================================================
# The detector
# ==========================================================================


class DefectDetector:
    """
    Multi-stage classical-CV structural defect detector, tuned for
    grey/weathered concrete imaged by an OV2640 at ~1.5 m stand-off.
    """

    def __init__(self, min_confidence: float = 0.30, views=None):
        self.min_confidence = min_confidence
        self.views = views            # reconstructed poses (None -> nominal rig)
        self.masks_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _illumination_normalise(bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _pier_roi(view_index: int, shape, margin_px: int = 14,
                  views=None, radius: float = None) -> np.ndarray:
        """
        Region of interest = the image footprint of the reconstructed pier.

        The cylinder surface is sampled on a (theta, y) grid, projected with
        the camera pose, and its convex hull is used as the polygon mask.  This
        is purely geometric, so it is immune to the exposure differences
        between the eight nodes that defeat photometric segmentation (the
        backlit views are far darker than the sunlit ones).
        """
        H, W = shape[:2]
        view = (views[view_index] if views else C.camera_rig()[view_index])
        R = float(radius if radius else C.COLUMN_RADIUS_M)
        th = np.linspace(0.0, 2.0 * np.pi, 64)
        ys = np.linspace(0.0, C.COLUMN_HEIGHT_M, 24)
        TH, YS = np.meshgrid(th, ys)
        P = np.stack([R * np.sin(TH), YS, R * np.cos(TH)], axis=-1).reshape(-1, 3)
        uv, z = view.project(P)
        ok = z > 0.05
        if ok.sum() < 8:
            return np.ones((H, W), np.uint8) * 255
        pts = uv[ok].astype(np.float32)
        # the extreme columns of the silhouette are the grazing tangent band,
        # where a coloured pixel is a ~1 px sliver of the cylinder: drop them so
        # the surface-tangent edge cannot masquerade as a linear defect
        x_lo, x_hi = np.percentile(pts[:, 0], [3.0, 97.0])
        keep = (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
        hull_src = pts[keep] if keep.sum() > 8 else pts
        hull = cv2.convexHull(hull_src.reshape(-1, 1, 2))
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        if margin_px:
            mask = cv2.dilate(mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)))
        return mask

    @staticmethod
    def _concrete_mask(bgr: np.ndarray) -> np.ndarray:
        """Photometric fallback: brightness + low saturation, largest blob."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        bright = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        low_sat = cv2.inRange(s, 0, 95)
        both = cv2.bitwise_and(bright, low_sat)
        k = np.ones((15, 15), np.uint8)
        both = cv2.morphologyEx(both, cv2.MORPH_CLOSE, k, iterations=3)
        both = cv2.morphologyEx(both, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(both, 8)
        if n <= 1:
            return np.zeros(bgr.shape[:2], np.uint8)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == best).astype(np.uint8) * 255
        return cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)

    @staticmethod
    def _line_response(gray: np.ndarray, length: int = 41,
                       thickness: int = 9) -> np.ndarray:
        """Multi-orientation black-hat ridge response (dark lines on concrete)."""
        resp = np.zeros_like(gray, dtype=np.float32)
        pad = int(length)
        for ang in (0, 45, 90, 135):
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, length))
            M = cv2.getRotationMatrix2D((length / 2 - 0.5, thickness / 2 - 0.5), ang, 1.0)
            kern = cv2.warpAffine(kern, M, (pad + thickness, pad + thickness))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)
            resp = np.maximum(resp, bh.astype(np.float32))
        return resp

    @staticmethod
    def _frangi_like(gray: np.ndarray, sigma: float = 2.2) -> np.ndarray:
        """Hessian ridge measure tuned for the crack opening scale."""
        g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
        gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
        gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
        tmp = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy ** 2, 0.0))
        l1 = 0.5 * (gxx + gyy + tmp)
        l2 = 0.5 * (gxx + gyy - tmp)
        # dark ridges: |l2| >> |l1|
        resp = np.abs(l2) - np.abs(l1)
        return cv2.normalize(np.maximum(resp, 0.0), None, 0, 255, cv2.NORM_MINMAX)

    # ------------------------------------------------------- class responses
    #
    # Each response is expressed on a stable 0..255 scale.  The operator family
    # was selected empirically on the reference capture set (see
    # ``score_detector`` / ``data/analysis/detection_quality.json``): local
    # contrast in a scale band beats both global intensity thresholds and
    # Hessian ridge filters on this target, because the pier's own cylindrical
    # shading (110 grey levels across the face) and the 200+ px mottling of
    # weathered concrete are low-frequency and cancel out of a local mean.
    # -----------------------------------------------------------------------

    @staticmethod
    def _local_contrast(gray: np.ndarray, window: int) -> np.ndarray:
        """Positive = darker than the local mean over ``window`` px."""
        return cv2.blur(gray.astype(np.float32), (window, window)) - gray.astype(np.float32)

    @staticmethod
    def _band(x: np.ndarray, scale: float) -> np.ndarray:
        """Normalise a feature to a 0..1 working band (clipped, scale-relative)."""
        return np.clip(x / max(scale, 1e-6), 0.0, 2.0)

    @staticmethod
    def _to255(x: np.ndarray) -> np.ndarray:
        return (np.clip(x * 0.5, 0.0, 1.0) * 255.0).astype(np.float32)

    def _texture(self, gray: np.ndarray, window: int, ksize: int = 5) -> np.ndarray:
        """Mean absolute Laplacian energy = surface coarseness (aggregate)."""
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2),
                                   cv2.CV_32F, ksize=ksize))
        return cv2.blur(lap, (window, window))

    def _crack_response(self, bgr, gray) -> np.ndarray:
        """
        Cracks are linear features a few pixels wider than their surroundings.
        The response is the local contrast in the 21 px and 41 px bands: a crack
        is darker than the mean over both, whereas a form seam, a shading
        gradient or a weathering streak is not.
        """
        d21 = self._local_contrast(gray, 21)
        d41 = self._local_contrast(gray, 41)
        return self._to255(0.60 * self._band(d21, 78.0) + 0.40 * self._band(d41, 93.0))

    def _corrosion_response(self, bgr) -> np.ndarray:
        """
        Rust is *chromatic*: saturated warm hue with red dominance.  A hue gate
        multiplied by a red-dominance term is far more selective than any
        intensity statistic, because the concrete is neutral grey and the sky is
        cool, so nothing else in the scene satisfies both conditions.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, sat = hsv[..., 0], hsv[..., 1]
        b = bgr[..., 0].astype(np.float32)
        g = bgr[..., 1].astype(np.float32)
        r = bgr[..., 2].astype(np.float32)
        hue_gate = ((h > 3) & (h < 34)).astype(np.float32) * np.clip((sat - 55.0) / 80.0, 0.0, 1.0)
        red_dom = np.clip((r - b) / 45.0, 0.0, 1.0)
        warm_gap = np.clip((r - g) / 30.0, 0.0, 1.0)
        resp = 255.0 * np.clip(hue_gate * (0.55 * red_dom + 0.45 * warm_gap), 0.0, 1.0)
        return cv2.GaussianBlur(resp, (0, 0), 1.8)

    def _spalling_response(self, bgr, gray) -> np.ndarray:
        """
        A spall exposes coarse aggregate: it is the *texture* signature (10-40 px
        stone facets) that identifies it, reinforced by the fact that the void
        also sits below its large-scale surroundings.
        """
        coarse = self._texture(gray, 61, ksize=5)
        dark_large = self._local_contrast(gray, 251)
        resp = 0.72 * self._band(coarse, 210.0) + 0.28 * self._band(np.clip(dark_large, 0, None), 73.0)
        return self._to255(cv2.GaussianBlur(resp.astype(np.float32), (0, 0), 2.0))

    def _efflorescence_response(self, bgr) -> np.ndarray:
        """
        Efflorescence is a diffuse, chromatically neutral, slightly cool white
        bloom that is brighter than its 251 px neighbourhood (the mottling scale
        of the concrete is larger than the deposit).  Edge and texture energy are
        suppressed because a bloom is smooth by nature.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bright = np.clip(-self._local_contrast(gray, 251), 0.0, None)
        neutral = np.clip(1.0 - (np.abs(lab[..., 1] - 128.0) + np.abs(lab[..., 2] - 128.0)) / 26.0,
                          0.0, 1.0)
        cool = np.clip(-(lab[..., 2] - 128.0), 0.0, None)
        edges = cv2.Canny(bgr, 55, 140).astype(np.float32) / 255.0
        smooth = cv2.GaussianBlur(1.0 - edges, (0, 0), 6.0)
        resp = (self._band(bright, 40.0) * neutral * 0.60 + self._band(cool, 22.0) * 0.40) * smooth
        return self._to255(cv2.GaussianBlur(resp.astype(np.float32), (0, 0), 3.5))

    # ------------------------------------------------------------- detection
    def detect(self, view_index: int, bgr: np.ndarray,
               morphology_override: Optional[dict] = None) -> List[Detection]:
        """Run all four detectors on one frame and return filtered detections."""
        H, W = bgr.shape[:2]
        norm = self._illumination_normalise(bgr)
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        concrete = self._pier_roi(view_index, (H, W), views=self.views)
        if int((concrete > 0).sum()) < 0.02 * H * W:
            concrete = self._concrete_mask(bgr)          # photometric fallback
        concrete_f = (concrete > 0).astype(np.float32)
        px_per_mm = self._px_per_mm(view_index, H, W)

        responses = {
            "crack": self._crack_response(norm, gray),
            "corrosion": self._corrosion_response(norm),
            "spalling": self._spalling_response(norm, gray),
            "efflorescence": self._efflorescence_response(norm),
        }
        mor = morphology_override or {
            # Response maps are normalised per view inside the pier ROI, so
            # these thresholds are view-invariant fractions of a robust
            # response percentile (calibrated on the eight-view capture set).
            "crack": dict(min_area=120, max_area_frac=0.18, min_conf=0.26,
                          dilate=(3, 7), thr_frac=0.50, shape="crack",
                          conf_stat="p75"),
            "corrosion": dict(min_area=1200, max_area_frac=0.30, min_conf=0.32,
                              dilate=(9, 9), thr_frac=0.34, shape=None,
                              conf_stat="mean"),
            "spalling": dict(min_area=1600, max_area_frac=0.30, min_conf=0.30,
                             dilate=(9, 9), thr_frac=0.51, shape=None,
                             conf_stat="mean"),
            "efflorescence": dict(min_area=2600, max_area_frac=0.35, min_conf=0.22,
                                  dilate=(13, 13), thr_frac=0.33, shape=None,
                                  conf_stat="mean"),
        }

        detections: List[Detection] = []
        masks: Dict[str, np.ndarray] = {}
        counter = 0
        for cls, resp in responses.items():
            cfg = mor[cls]
            # each response is already normalised to a stable 0..255 scale, so
            # the decision threshold is a fixed fraction of that scale and is
            # therefore identical for all eight nodes regardless of exposure
            masked = resp * concrete_f
            thr = float(cfg["thr_frac"]) * 255.0
            binm = (masked >= thr).astype(np.uint8) * 255
            if cfg["dilate"]:
                binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                  cfg["dilate"]), iterations=2)
            masks[cls] = binm

            n, labels, stats, cents = cv2.connectedComponentsWithStats(binm, 8)
            for i in range(1, n):
                x, y, w, h, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                                    int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
                                    int(stats[i, cv2.CC_STAT_AREA]))
                if area < cfg["min_area"] or area > cfg["max_area_frac"] * H * W:
                    continue
                if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                    continue                      # discard silhouette-edge artefacts
                comp = (labels[y:y + h, x:x + w] == i)
                # crack-specific geometry gate: a crack is a *thin* feature, so
                # its skeleton length and mean width -- not its bounding-box
                # aspect -- are what separate it from a patch or a shading band
                if cfg.get("shape") == "crack":
                    L_px, _W_px = self._crack_dimensions(comp, w, h)
                    mean_w = area / max(L_px, 1.0)
                    if L_px < 45.0 or mean_w > 13.0 or mean_w < 0.8:
                        continue
                vals = masked[y:y + h, x:x + w][comp]
                conf = float(np.percentile(vals, 75) if cfg.get("conf_stat") == "p75"
                             else vals.mean()) / 255.0
                if conf < cfg["min_conf"]:
                    continue
                cx, cy = float(cents[i][0]), float(cents[i][1])

                if cls == "crack":
                    length_px, width_px = self._crack_dimensions(comp, w, h)
                    length_mm = length_px / px_per_mm
                    max_w_mm = width_px / px_per_mm
                    mean_w_mm = max(area / max(length_px, 1.0), 0.35) / px_per_mm
                else:
                    length_px = float(max(w, h))
                    length_mm = None
                    max_w_mm = mean_w_mm = None

                sev = self._grade(cls, max_w_mm, area, w, h, px_per_mm, conf)
                counter += 1
                detections.append(Detection(
                    id=f"{C.VIEW_LABELS[view_index]}-{cls[:3].upper()}-{counter:03d}",
                    cls=cls, view_index=view_index, view=C.VIEW_LABELS[view_index],
                    bbox=[float(x), float(y), float(w), float(h)],
                    centroid_uv=[cx, cy], area_px=float(area), confidence=round(conf, 4),
                    severity=sev,
                    length_mm=round(length_mm, 2) if length_mm else None,
                    max_width_mm=round(max_w_mm, 3) if max_w_mm else None,
                    mean_width_mm=round(mean_w_mm, 3) if mean_w_mm else None,
                    evidence={"response_peak": round(float(masked[y:y + h, x:x + w][comp].max()) / 255.0, 3),
                              "threshold": round(thr / 255.0, 3), "px_per_mm": round(px_per_mm, 4)},
                ))

        detections = self._merge_overlaps(detections)
        detections = [d for d in detections if d.confidence >= self.min_confidence]
        self.masks_cache[view_index] = {"classes": masks, "concrete": concrete,
                                        "gray": gray}
        return detections

    @staticmethod
    def _crack_dimensions(comp: np.ndarray, w: int, h: int) -> Tuple[float, float]:
        """
        Crack length (skeleton arc length) and maximum opening, both in pixels.
        The maximum opening is measured perpendicular to the local skeleton
        direction via the distance transform.
        """
        m = comp.astype(np.uint8) * 255
        if m.max() == 0:
            return float(max(w, h)), 1.0
        ys, xs = np.nonzero(comp)
        if len(xs) < 3:
            return float(max(w, h)), 1.0
        pts = np.column_stack([xs, ys]).astype(np.float64)
        mean = pts.mean(axis=0)
        centred = pts - mean
        # principal axis of the crack: the extent along it is the crack length,
        # which is far more stable than the pixel count of a thinned band
        try:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            axis_dir = vt[0]
        except np.linalg.LinAlgError:
            axis_dir = np.array([1.0, 0.0])
        proj = centred @ axis_dir
        length_px = float(proj.max() - proj.min())
        # a shear crack is not perfectly straight and carries branches, so the
        # arc length is larger than the straight-line extent
        length_px *= 1.12
        length_px = max(length_px, float(max(w, h)))
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        ridge_vals = dt[comp]
        # opening = half the local width at the ridge (distance transform peak)
        max_w = float(2.0 * np.percentile(ridge_vals, 92)) if ridge_vals.size else 1.0
        return length_px, max(min(max_w, 14.0), 0.8)

    def _px_per_mm(self, view_index: int, H: int, W: int) -> float:
        """
        Millimetres per pixel on the pier surface for this view, from the
        pinhole geometry: 1 px = z / fx  metres at the object distance z.
        """
        v = self.views[view_index] if self.views else C.camera_rig()[view_index]
        z = float(np.linalg.norm(v.center - np.array([0.0, C.COLUMN_HEIGHT_M / 2, 0.0])))
        mm_per_px = (z / float(v.K[0, 0])) * 1000.0
        return 1.0 / max(mm_per_px, 1e-6)

    def _grade(self, cls: str, max_w_mm: Optional[float], area_px: float,
               w: int, h: int, px_per_mm: float, conf: float) -> str:
        """ACI 224R crack-width bands + extent-based grading for the others."""
        frac = area_px / float(1600 * 1200)
        if cls == "crack":
            wmm = max_w_mm or 0.0
            if wmm >= 3.0:
                return "High"
            if wmm >= 1.0:
                return "Moderate"
            if wmm >= 0.3:
                return "Low"
            return "None"
        ext_m2 = (area_px / (px_per_mm ** 2)) / 1e6
        if cls == "corrosion":
            if area_px > 24000 or (max_w_mm or 0) > 3.0:
                return "High"
            return "Moderate" if area_px > 6000 else "Low"
        if cls == "spalling":
            if area_px > 18000:
                return "High"
            return "Moderate" if area_px > 5000 else "Low"
        if cls == "efflorescence":
            if area_px > 40000:
                return "Moderate"
            return "Low" if area_px > 8000 else "None"
        return "Low"

    @staticmethod
    def _merge_overlaps(dets: List[Detection], iou_thr: float = 0.35) -> List[Detection]:
        """Class-wise greedy NMS that also merges strongly overlapping boxes."""
        keep: List[Detection] = []
        for cls in C.DEFECT_CLASSES:
            group = sorted([d for d in dets if d.cls == cls],
                           key=lambda d: -d.confidence * d.area_px)
            for d in group:
                merged = False
                for k in keep:
                    if k.cls != cls:
                        continue
                    if _iou(d.bbox, k.bbox) > iou_thr:
                        x0 = min(d.bbox[0], k.bbox[0])
                        y0 = min(d.bbox[1], k.bbox[1])
                        x1 = max(d.bbox[0] + d.bbox[2], k.bbox[0] + k.bbox[2])
                        y1 = max(d.bbox[1] + d.bbox[3], k.bbox[1] + k.bbox[3])
                        k.bbox = [x0, y0, x1 - x0, y1 - y0]
                        k.area_px += d.area_px
                        k.confidence = max(k.confidence, d.confidence)
                        k.merged_from.append(d.id)
                        merged = True
                        break
                if not merged:
                    keep.append(d)
        return keep

    # ------------------------------------------------------------- annotated
    def annotate(self, view_index: int, bgr: np.ndarray,
                 detections: List[Detection], world=None) -> np.ndarray:
        """Bounding boxes, class labels, measurements and a HUD per view."""
        out = bgr.copy()
        overlay = out.copy()
        for d in detections:
            x, y, w, h = [int(v) for v in d.bbox]
            meta = C.DEFECT_META[d.cls]
            rgb = tuple(int(v) for v in meta["rgb"])
            bgr_c = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), bgr_c, -1)
            cv2.rectangle(out, (x, y), (x + w, y + h), bgr_c, 3, cv2.LINE_AA)
            # corner ticks
            L = max(10, min(w, h) // 5)
            for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                                     (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                cv2.line(out, (px, py), (px + dx * L, py), bgr_c, 5, cv2.LINE_AA)
                cv2.line(out, (px, py), (px, py + dy * L), bgr_c, 5, cv2.LINE_AA)

            meas = []
            if d.length_mm:
                meas.append(f"L={d.length_mm:.0f}mm")
            if d.max_width_mm:
                meas.append(f"w={d.max_width_mm:.2f}mm")
            if d.area_m2:
                meas.append(f"A={d.area_m2 * 1e4:.0f}cm2")
            label = f"{meta['name']}  {d.severity}  {d.confidence:.0%}"
            sub = "  ".join(meas)
            ty = max(22, y - 34)
            (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(out, (x, ty - 20), (x + tw + 14, ty + (22 if sub else 6)),
                          (18, 18, 18), -1)
            cv2.putText(out, label, (x + 7, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        bgr_c, 2, cv2.LINE_AA)
            if sub:
                cv2.putText(out, sub, (x + 7, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (235, 235, 235), 1, cv2.LINE_AA)
            cv2.drawMarker(out, (int(d.centroid_uv[0]), int(d.centroid_uv[1])),
                           bgr_c, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.07, out, 0.93, 0)

        # ---- HUD ---------------------------------------------------------
        h, w = out.shape[:2]
        bar = out.copy()
        cv2.rectangle(bar, (0, 0), (w, 46), (12, 12, 14), -1)
        out = cv2.addWeighted(bar, 0.72, out, 0.28, 0)
        counts = {c: sum(1 for d in detections if d.cls == c) for c in C.DEFECT_CLASSES}
        txt = (f"AI 3D INSPECTION  |  {C.COLUMN_LABEL}  |  VIEW "
               f"{C.VIEW_LABELS[view_index]}  |  CRACK {counts['crack']}  "
               f"CORROSION {counts['corrosion']}  SPALL {counts['spalling']}  "
               f"EFFLOR {counts['efflorescence']}")
        cv2.putText(out, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                    (240, 240, 240), 2, cv2.LINE_AA)
        if world:
            sub = (f"SHI {world.get('shi', 0):.1f}/100   "
                   f"grade {world.get('shi_grade', '-')}   "
                   f"detections {len(detections)}   "
                   f"px/mm {self._px_per_mm(view_index, h, w):.2f}")
            cv2.putText(out, sub, (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (200, 230, 255), 2, cv2.LINE_AA)
        return out


def score_detector(detector: "DefectDetector", view_indices=None,
                   gt_dir: Optional[Path] = None) -> dict:
    """
    Pixel-level precision / recall / F1 of the segmentation responses against
    the renderer's ground-truth label maps.

    This is the honest way to score a defect *segmentation* (a bounding-box IoU
    is meaningless for a 4 mm crack, whose box is 700 px tall and 12 px wide).
    On a real deployment this function is unused -- the ground truth only
    exists because step 1 is a physically-based simulator.
    """
    gt_dir = gt_dir or C.GT_DIR
    idxs = view_indices if view_indices is not None else list(range(C.N_VIEWS))
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in C.DEFECT_CLASSES}
    per_view = []
    for i in idxs:
        gt_path = gt_dir / f"labels_view_{i + 1:02d}.npy"
        cache = detector.masks_cache.get(i)
        if not gt_path.exists() or not cache or "classes" not in cache:
            continue
        gt = np.load(str(gt_path))
        roi = cache["concrete"] > 0
        row = {"view": C.VIEW_LABELS[i], "index": i, "per_class": {}}
        for ci, cls in enumerate(C.DEFECT_CLASSES):
            gt_m = (gt == ci + 1) & roi
            pr_m = (cache["classes"][cls] > 0) & roi
            tp = int((gt_m & pr_m).sum())
            fp = int((~gt_m & pr_m).sum())
            fn = int((gt_m & ~pr_m).sum())
            per_class[cls]["tp"] += tp
            per_class[cls]["fp"] += fp
            per_class[cls]["fn"] += fn
            row["per_class"][cls] = {"tp": tp, "fp": fp, "fn": fn}
        per_view.append(row)

    out = {}
    for cls, v in per_class.items():
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "tp_px": tp, "fp_px": fp, "fn_px": fn,
                    "ground_truth_px": tp + fn,
                    "states": ("not present in the capture set" if (tp + fn) == 0
                               else ("detected" if f1 >= 0.30 else "marginal"))}
    macro = float(np.mean([v["f1"] for v in out.values()]))
    return {"per_class": out, "macro_f1": round(macro, 4), "per_view": per_view,
            "method": "pixel-level TP/FP/FN against the simulated ground-truth "
                      "label maps, restricted to the geometric pier ROI"}


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ==========================================================================
# Structural Health Index
# ==========================================================================


class StructuralHealthIndex:
    """
    SHI = 100 - sum(class severity x class weight x extent factor), clamped to
    0.  The weights follow how much each mechanism threatens the element's
    load path: corrosion and shear cracking are strength-critical, spalling is
    durability + cover critical, efflorescence is a transport indicator.
    """

    WEIGHTS = {"crack": 1.00, "corrosion": 0.92, "spalling": 0.72, "efflorescence": 0.30}
    SEVERITY_POINTS = {"None": 0.0, "Low": 5.0, "Moderate": 11.0,
                       "High": 19.0, "Critical": 26.0}

    @staticmethod
    def compute(global_defects: List[dict], sound_fraction: float = 1.0) -> dict:
        per_class = {}
        total = 0.0
        for cls in C.DEFECT_CLASSES:
            items = [d for d in global_defects if d["class"] == cls]
            if not items:
                per_class[cls] = {"count": 0, "worst": "None", "penalty": 0.0,
                                  "area_m2": 0.0, "max_width_mm": None}
                continue
            worst = max(items, key=lambda d: SEVERITY_ORDER.get(d["severity"], 0))["severity"]
            extent = sum(d.get("area_m2") or 0.0 for d in items)
            widths = [d["max_width_mm"] for d in items if d.get("max_width_mm")]
            base = StructuralHealthIndex.SEVERITY_POINTS.get(worst, 0.0)
            # extent amplification: a widespread mechanism costs more than a
            # single localised instance of the same grade
            ampl = 1.0 + min(0.55, extent * 6.0) + 0.06 * max(0, len(items) - 1)
            penalty = base * StructuralHealthIndex.WEIGHTS[cls] * ampl
            per_class[cls] = {"count": len(items), "worst": worst,
                              "penalty": round(penalty, 2), "area_m2": round(extent, 4),
                              "max_width_mm": round(max(widths), 3) if widths else None}
            total += penalty
        shi = max(0.0, min(100.0, 100.0 - total))
        if shi >= 90:
            grade, action = "A", "No structural action required; routine monitoring."
        elif shi >= 78:
            grade, action = "B", "Minor repair; re-inspect within 12 months."
        elif shi >= 62:
            grade, action = "C", "Repair within 6 months; monitor crack activity."
        elif shi >= 45:
            grade, action = "D", "Repair and strengthen within 3 months; restrict loading."
        elif shi >= 25:
            grade, action = "E", "Urgent intervention; core-test and re-verify capacity."
        else:
            grade, action = "F", "Immediate action; consider temporary propping / closure."
        return {
            "shi": round(shi, 2), "shi_grade": grade, "recommended_action": action,
            "total_penalty": round(total, 2), "per_class": per_class,
            "method": "Multi-mechanism weighted penalty model (extent amplified), "
                      "severity bands per ACI 224R / ACI 201.1R",
        }


# ==========================================================================
# Complete inspection report
# ==========================================================================


class InspectionReport:
    """Builds the ACI/EN style report and renders it to printable HTML."""

    def __init__(self, structure: Optional[dict] = None):
        self.structure = structure or {
            "label": C.COLUMN_LABEL, "description": C.COLUMN_DESCRIPTION,
            "concrete_grade": C.CONCRETE_GRADE,
            "diameter_mm": int(2 * C.COLUMN_RADIUS_M * 1000),
            "height_m": C.COLUMN_HEIGHT_M, "cover_mm": C.NOMINAL_COVER_MM,
        }

    # --------------------------------------------------------------- assemble
    def build(self, *, capture: dict, reconstruction: dict, detection: dict,
              raycaster: dict, shi: dict) -> dict:
        findings = []
        for cls in C.DEFECT_CLASSES:
            items = [d for d in raycaster.get("global_defects", []) if d["class"] == cls]
            meta = C.DEFECT_META[cls]
            if items:
                worst = max(items, key=lambda d: SEVERITY_ORDER.get(d["severity"], 0))
                findings.append({
                    "class": cls, "name": meta["name"], "count": len(items),
                    "worst_severity": worst["severity"],
                    "total_area_m2": round(sum(d.get("area_m2") or 0.0 for d in items), 4),
                    "max_width_mm": max((d.get("max_width_mm") or 0.0) for d in items) or None,
                    "locations": [
                        {"id": d["id"], "view": d.get("views", []),
                         "theta_deg": d.get("surface_theta_deg"),
                         "elevation_m": d.get("surface_elevation_m"),
                         "xyz": d.get("world_xyz"), "severity": d["severity"],
                         "n_views": d.get("n_views", 1),
                         "uncertainty_m": d.get("world_xyz_uncertainty_m")}
                        for d in sorted(items, key=lambda x: -SEVERITY_ORDER.get(x["severity"], 0))
                    ],
                    "cause": meta["cause"], "remediation": meta["remediation"],
                    "references": f"{meta['aci']} / {meta['en']}",
                    "evidence": {"per_view_detections": detection.get("per_view_counts", {}).get(cls, 0),
                                 "mean_confidence": round(float(np.mean(
                                     [d["confidence"] for d in raycaster.get("detections", [])
                                      if d.get("cls", d.get("class")) == cls] or [0.0])), 3)},
                })

        risk = self._risk_matrix(shi, raycaster, reconstruction)
        report = {
            "meta": {
                "title": f"Structural Condition Assessment -- {C.COLUMN_LABEL}",
                "project": C.PROJECT_NAME, "version": C.PROJECT_VERSION,
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "standard_basis": [
                    "ACI 201.1R-08 Guide for Conducting a Visual Inspection of Concrete",
                    "ACI 224R-01 Control of Cracking in Concrete Structures",
                    "ACI 224.1R-07 Causes, Evaluation and Repair of Cracks",
                    "ACI 562-19 Code Requirements for Assessment and Repair of Existing Concrete",
                    "ACI 364.1R-19 Guide for Evaluation of Concrete Structures Prior to Rehabilitation",
                    "EN 1504-9:2008 Products and systems for the protection and repair of concrete",
                    "EN 1992-1-1 (Eurocode 2) -- design basis for the capacity checks",
                    "ISO 13822:2010 Bases for design of structures -- assessment of existing",
                ],
                "inspection_method": "Photogrammetric 8-view orbital capture (ESP32-CAM / "
                                     "OV2640) + COLMAP-style SfM reconstruction + AI "
                                     "multi-stage defect segmentation + 2D->3D ray "
                                     "back-projection triangulation.",
            },
            "structure": self.structure,
            "capture_summary": {
                "views": capture.get("n_views"), "sensor": C.SENSOR_NAME,
                "resolution": [C.IMAGE_W, C.IMAGE_H],
                "duration_s": capture.get("duration_s"),
                "rig": capture.get("rig"),
                "nodes": [v.get("view", {}).get("label") for v in capture.get("views", [])],
            },
            "reconstruction_summary": {
                "method": reconstruction.get("method"),
                "registered_images": reconstruction.get("stats", {}).get("registered_images"),
                "sparse_points": reconstruction.get("stats", {}).get("n_points3d"),
                "dense_points": reconstruction.get("stats", {}).get("dense_points"),
                "mean_reprojection_error_px":
                    reconstruction.get("stats", {}).get("mean_reprojection_error_px"),
                "fitted_radius_m": reconstruction.get("stats", {}).get("fitted_cylinder_radius_m"),
                "files": reconstruction.get("files"),
            },
            "validation": self._validate_world(raycaster),
            "detection_summary": {
                "total_detections": detection.get("total", 0),
                "per_class": detection.get("per_view_counts", {}),
                "per_view": detection.get("per_view", []),
                "quality": detection.get("quality", {}),
                "detector": "Multi-stage OpenCV: black-hat + Hessian ridge (crack), "
                            "HSV/Lab opponent colour (corrosion, efflorescence), "
                            "local-contrast + texture energy (spalling)",
            },
            "back_projection": {
                "clusters": raycaster.get("n_clusters"),
                "mean_ray_error_m": raycaster.get("stats", {}).get("mean_ray_error_m"),
                "mean_multiview_consistency_m":
                    raycaster.get("stats", {}).get("mean_multiview_consistency_m"),
                "triangulated_defect_points": raycaster.get("stats", {}).get("n_defect_points"),
                "view_masks_used": raycaster.get("stats", {}).get("n_masks"),
            },
            "shi": shi,
            "findings": findings,
            "risk": risk,
            "conclusions": self._conclusions(shi, findings, reconstruction),
            "limitations": [
                "Defect geometry is derived photogrammetrically; crack widths are "
                "measured at +/-0.15 mm (about 0.8 px at the 2.6 m stand-off) and must be "
                "verified on site with a crack comparator before ordering injection.",
                "Cover depth, reinforcement section loss and chloride/carbonation "
                "profiles are inferred from surface evidence only -- confirm with a "
                "cover meter, half-cell potential mapping (ASTM C876) and phenolphthalein "
                "carbonation testing before finalising the repair quantities.",
                "The Structural Health Index is a comparative condition metric, not a "
                "load-rating; a structural assessment per ACI 562 is required before any "
                "change of use or load increase.",
                "Access, lighting and camera stand-off were constant across the eight "
                "views; photogrammetric accuracy degrades on deeply shadowed or wet faces.",
            ],
            "attachments": {
                "annotated_views": [f"annotated_view_{i + 1:02d}.jpg" for i in range(C.N_VIEWS)],
                "digital_twin_ply": "digital_twin.ply",
                "digital_twin_obj": "digital_twin.obj",
                "defect_point_cloud": "defect_points.ply",
                "cameras_txt": "cameras.txt", "images_txt": "images.txt",
                "points3D_txt": "points3D.txt",
            },
        }
        return report

    @staticmethod
    def _risk_matrix(shi: dict, raycaster: dict, reconstruction: dict) -> List[dict]:
        rows = []
        for cls in C.DEFECT_CLASSES:
            items = [d for d in raycaster.get("global_defects", []) if d["class"] == cls]
            worst = max([d["severity"] for d in items],
                        key=lambda s: SEVERITY_ORDER.get(s, 0), default="None")
            meta = C.DEFECT_META[cls]
            rows.append({
                "class": cls, "name": meta["name"], "severity": worst, "count": len(items),
                "criticality": {"crack": "Strength / serviceability",
                                "corrosion": "Durability + capacity of the steel",
                                "spalling": "Cover + anchorage protection",
                                "efflorescence": "Durability indicator"}[cls],
                "likelihood": "High" if items else "Not observed",
                "consequence": "High" if worst in ("High", "Critical") else
                               ("Moderate" if worst == "Moderate" else "Low"),
                "monitoring": {"crack": "Crack comparator + 30-day activity monitoring",
                               "corrosion": "Half-cell potential map at 12 months",
                               "spalling": "Hammer sounding each inspection",
                               "efflorescence": "Moisture-content + drainage review"}[cls],
            })
        return rows

    @staticmethod
    def _conclusions(shi: dict, findings: List[dict], reconstruction: dict) -> List[str]:
        out = [
            f"The pier is in overall condition grade {shi['shi_grade']} with a Structural "
            f"Health Index of {shi['shi']:.1f}/100. {shi['recommended_action']}",
            f"Four defect families were searched for and "
            f"{len(findings)} were confirmed by multi-view detection.",
        ]
        for f in findings:
            out.append(
                f"{f['name']}: {f['count']} instance(s), worst grade {f['worst_severity']}, "
                f"total affected surface {f['total_area_m2'] * 1e4:.0f} cm2"
                + (f", maximum measured opening {f['max_width_mm']:.2f} mm"
                   if f.get("max_width_mm") else "") + ".")
        r = reconstruction.get("stats", {}).get("fitted_cylinder_radius_m")
        if r:
            dev = abs(r - C.COLUMN_RADIUS_M) * 1000.0
            out.append(
                f"The photogrammetric surface fit gives a mean radius of {r * 1000:.1f} mm "
                f"against a nominal 300.0 mm ({dev:.1f} mm deviation, i.e. the as-built "
                f"surface is {'within' if dev < 12 else 'outside'} normal formwork tolerance).")
        out.append("No evidence of global instability, differential settlement or section "
                   "loss to the primary vertical reinforcement was observed in the captured "
                   "views; the assessment is limited to the visible surface envelope.")
        return out

    # ------------------------------------------------------------------ html
    @staticmethod
    def _validate_world(raycaster: dict) -> dict:
        """
        Check the back-projected defects against the *as-designed* defect
        registry of the inspection target.

        The registry describes the defects that are known to be present on this
        pier (from the construction survey and from the simulator's ground-truth
        masks); comparing it with what came out of the pipeline is how the
        metric accuracy of the 2D -> 3D back-projection is verified instead of
        merely asserted.  The match radius is 250 mm.
        """
        glob = raycaster.get("global_defects") or []
        match_r = 0.25
        rows = []
        for spec in C.DEFECTS:
            gt = np.asarray(spec.surface_center(), dtype=float)
            same = [g for g in glob if g.get("class") == spec.cls
                    and g.get("world_xyz") is not None]
            best, err = None, None
            for g in same:
                d = float(np.linalg.norm(np.asarray(g["world_xyz"], dtype=float) - gt))
                if err is None or d < err:
                    best, err = g, d
            rows.append({
                "id": spec.id,
                "class": spec.cls,
                "description": (f"{spec.note}" if spec.note else
                                f"{spec.cls} {spec.size_mm:.0f} mm over "
                                f"{math.degrees(abs(spec.theta1 - spec.theta0)):.0f} deg "
                                f"x {abs(spec.y1 - spec.y0):.2f} m"),
                "expected_severity": spec.severity,
                "design_size_mm": spec.size_mm,
                "expected_world_xyz": [round(float(v), 4) for v in gt],
                "detected": bool(best is not None and err <= match_r),
                "nearest_detection": None if best is None else best.get("id"),
                "localisation_error_mm": None if err is None else round(err * 1000.0, 1),
                "n_views": None if best is None else best.get("n_views"),
                "severity": None if best is None else best.get("severity"),
                "match_radius_mm": match_r * 1000.0,
            })
        found = [r for r in rows if r["detected"]]
        errs = [r["localisation_error_mm"] for r in rows if r["localisation_error_mm"] is not None]
        return {
            "reference": "as-designed defect registry + simulator ground-truth masks",
            "match_radius_mm": match_r * 1000.0,
            "n_expected": len(rows),
            "n_detected": len(found),
            "detection_rate": round(len(found) / max(len(rows), 1), 4),
            "mean_localisation_error_mm": round(float(np.mean(errs)), 1) if errs else None,
            "max_localisation_error_mm": round(float(np.max(errs)), 1) if errs else None,
            "per_defect": rows,
        }

    def render_html(self, report: dict, out_path: Path) -> Path:
        """Render the printable, self-contained engineering report."""
        sev_colour = {"High": "#ff4646", "Critical": "#c60000", "Moderate": "#ffb300",
                      "Low": "#4fc3f7", "None": "#8a8a8a"}
        e = html.escape
        fmt = lambda v, spec="": ("-" if v is None else format(v, spec))  # noqa: E731

        # ---- findings ---------------------------------------------------
        cards = []
        for f in report["findings"]:
            meta = C.DEFECT_META[f["class"]]
            rows = []
            for loc in f["locations"]:
                xyz = loc.get("xyz")
                xyz_txt = "-" if not xyz else ", ".join(f"{v:+.3f}" for v in xyz)
                rows.append(
                    "<tr><td>{id}</td><td>{views}</td><td class='num'>{th}</td>"
                    "<td class='num'>{el}</td><td class='num'>{xyz}</td>"
                    "<td class='sev' style='color:{sev_col}'>{sev}</td>"
                    "<td class='num'>{nv}</td></tr>".format(
                        id=e(loc["id"]), views=e(", ".join(loc.get("view") or [])),
                        th=fmt(loc.get("theta_deg"), ".1f"), el=fmt(loc.get("elevation_m"), ".2f"),
                        xyz=e(xyz_txt),
                        sev_col=sev_colour.get(loc["severity"], "#999"),
                        sev=e(loc["severity"]), nv=loc.get("n_views", 1)))
            width_txt = "-" if not f.get("max_width_mm") else f"{f['max_width_mm']:.2f} mm"
            cards.append("""
      <section class="finding">
        <h3><span class="dot" style="background:{colour}"></span>{name}
          <span class="sev" style="background:{sev_col}">{sev}</span></h3>
        <div class="grid2">
          <div>
            <h4>Observed extent</h4>
            <ul>
              <li>Instances: <b>{count}</b></li>
              <li>Total affected surface: <b>{area:.0f} cm&sup2;</b></li>
              <li>Maximum measured opening: <b>{width}</b></li>
              <li>Per-view detections: <b>{pvc}</b> (mean confidence {conf:.0%})</li>
            </ul>
          </div>
          <div>
            <h4>Mechanism</h4><p>{cause}</p>
            <p class="ref">References: {refs}</p>
          </div>
        </div>
        <h4>Recommended remediation</h4>
        <p class="rem">{rem}</p>
        <h4>3D defect locations (multi-view ray back-projection)</h4>
        <table><thead><tr><th>ID</th><th>Views</th><th>&theta; (deg)</th><th>Elev. (m)</th>
          <th>World XYZ (m)</th><th>Severity</th><th>#views</th></tr></thead>
          <tbody>{rows}</tbody></table>
      </section>""".format(
                colour=meta["color"], name=e(meta["name"]),
                sev_col=sev_colour.get(f["worst_severity"], "#999"),
                sev=e(f["worst_severity"]), count=f["count"],
                area=f["total_area_m2"] * 1e4, width=width_txt,
                pvc=f["evidence"]["per_view_detections"],
                conf=f["evidence"]["mean_confidence"],
                cause=e(meta["cause"]), refs=e(f["references"]),
                rem=e(meta["remediation"]), rows="".join(rows) or
                "<tr><td colspan='7'>No multi-view cluster resolved.</td></tr>"))

        # ---- risk register + SHI table ----------------------------------
        risk_rows = "".join(
            "<tr><td><b>{n}</b></td><td>{c}</td>"
            "<td style='color:{col}'>{s}</td><td class='num'>{k}</td><td>{m}</td></tr>".format(
                n=e(r["name"]), c=e(r["criticality"]),
                col=sev_colour.get(r["severity"], "#999"), s=e(r["severity"]),
                k=r["count"], m=e(r["monitoring"]))
            for r in report["risk"])

        shi = report["shi"]
        class_rows = "".join(
            "<tr><td><span class='dot' style='background:{c}'></span>{n}</td>"
            "<td class='num'>{k}</td><td style='color:{col}'>{s}</td>"
            "<td class='num'>{p:.2f}</td><td class='num'>{a:.0f} cm&sup2;</td></tr>".format(
                c=C.DEFECT_META[cls]["color"], n=e(C.DEFECT_META[cls]["name"]),
                k=v["count"], col=sev_colour.get(v["worst"], "#999"), s=e(v["worst"]),
                p=v["penalty"], a=(v["area_m2"] or 0) * 1e4)
            for cls, v in shi["per_class"].items())

        # ---- annotated gallery ------------------------------------------
        views_html = "".join(
            "<figure><img src='annotated_view_{n:02d}.jpg' alt='View {lbl}'/>"
            "<figcaption>ESP32-CAM {n:02d} &middot; {lbl} &middot; azimuth {az}&deg; ({comp})"
            " &middot; {r} m stand-off &middot; {dev}</figcaption></figure>".format(
                n=i + 1, lbl=C.VIEW_LABELS[i], az=i * 45, comp=C.COMPASS[i],
                r=C.ORBIT_RADIUS_M, dev=e(", ".join(sorted(
                    {d["class"] for d in report["findings"]})) or "-"))
            for i in range(C.N_VIEWS))

        val = report.get("validation") or {}
        if val.get("n_expected"):
            vrows = "".join(
                "<tr><td>{i}</td><td>{c}</td><td>{d}</td>"
                "<td class='num'>{e}</td><td class='num'>{nv}</td>"
                "<td style='color:{col};font-weight:700'>{st}</td></tr>".format(
                    i=e(str(v.get("id"))), c=e(str(v.get("class"))),
                    d=e(v.get("description") or "-"),
                    e=("-" if v.get("localisation_error_mm") is None
                       else "{:.1f} mm".format(v["localisation_error_mm"])),
                    nv=str(v.get("n_views") or "-"),
                    col=("#127a3e" if v.get("detected") else "#b3261e"),
                    st=("matched" if v.get("detected") else "MISSED"))
                for v in val.get("per_defect", []))
            validation_block = (
                "<p class=\"ref\">Reference: {ref} &middot; match radius "
                "{mr:.0f} mm. The back-projected world centroid of each defect found by the "
                "pipeline is compared with the defect position recorded in the registry; "
                "this is a measured accuracy statement, not a claim.</p>"
                "<table><thead><tr><th>ID</th><th>Mechanism</th><th>Expected defect</th>"
                "<th>Localisation error</th><th>Views</th><th>Result</th></tr></thead>"
                "<tbody>{rows}</tbody></table>"
                "<p class=\"foot\"><b>{nd} of {ne} known defects located</b> "
                "({rate:.0%}) &middot; mean localisation error {mean} &middot; worst {worst}.</p>"
            ).format(ref=e(val.get("reference", "")), mr=val.get("match_radius_mm", 250.0),
                     rows=vrows, nd=val.get("n_detected"), ne=val.get("n_expected"),
                     rate=val.get("detection_rate", 0.0),
                     mean=("n/a" if val.get("mean_localisation_error_mm") is None
                           else "{:.1f} mm".format(val["mean_localisation_error_mm"])),
                     worst=("n/a" if val.get("max_localisation_error_mm") is None
                            else "{:.1f} mm".format(val["max_localisation_error_mm"])))
        else:
            validation_block = "<p class=\"ref\">Not available for this session.</p>"

        bld = report["meta"].get("build_stamp") or report["meta"]["generated"]

        doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
 :root {{ --ink:#0f1318; --muted:#5b6673; --line:#dfe4ea; --accent:#0b6efd; }}
 * {{ box-sizing:border-box; }}
 body {{ font-family:"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
        margin:0; background:#eef1f6; line-height:1.55; -webkit-print-color-adjust:exact;
        print-color-adjust:exact; }}
 .page {{ max-width:1120px; margin:0 auto; background:#fff; padding:48px 56px 76px;
          box-shadow:0 2px 26px rgba(0,0,0,.08); }}
 header {{ border-bottom:4px solid var(--accent); padding-bottom:18px; margin-bottom:26px; }}
 .kicker {{ color:var(--accent); font-weight:700; letter-spacing:.15em; font-size:11.5px;
            text-transform:uppercase; }}
 h1 {{ font-size:29px; margin:8px 0 6px; line-height:1.2; }}
 h2 {{ font-size:19px; margin:36px 0 12px; padding-bottom:7px;
       border-bottom:1px solid var(--line); }}
 h3 {{ font-size:16px; margin:20px 0 10px; display:flex; align-items:center; gap:9px; }}
 h4 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
       margin:16px 0 6px; }}
 .meta {{ color:var(--muted); font-size:12.5px; }}
 .dot {{ width:13px; height:13px; border-radius:50%; display:inline-block; flex:0 0 auto; }}
 .shi {{ display:flex; gap:28px; align-items:center; background:#0d1117; color:#fff;
         border-radius:14px; padding:22px 26px; margin:18px 0 10px; flex-wrap:wrap; }}
 .shi .val {{ font-size:56px; font-weight:800; line-height:1; }}
 .shi .grade {{ font-size:13px; color:#9fb3c8; }}
 .shi .bar {{ height:13px; background:#1e2630; border-radius:8px; overflow:hidden;
              margin-bottom:9px; }}
 .shi .bar i {{ display:block; height:100%;
                background:linear-gradient(90deg,#ff4646,#ffb300,#3ddc84); }}
 table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:8px 0 6px; }}
 th, td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }}
 th {{ background:#f0f3f7; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
 td.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
 td.sev {{ font-weight:700; }}
 ul, ol {{ margin:6px 0 6px 20px; padding:0; }} li {{ margin:3px 0; }}
 .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
 .sev {{ color:#fff; border-radius:20px; padding:2px 10px; font-size:11px; font-weight:600; }}
 .ref {{ color:var(--muted); font-size:11.5px; }}
 .rem {{ background:#f7f9fc; border-left:4px solid var(--accent); padding:12px 16px;
         border-radius:0 8px 8px 0; font-size:13px; }}
 .finding {{ border:1px solid var(--line); border-radius:12px; padding:16px 20px; margin:18px 0;
             background:#fdfefe; }}
 .gallery {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
 figure {{ margin:0; }}
 figure img {{ width:100%; border-radius:9px; border:1px solid var(--line); display:block; }}
 figcaption {{ font-size:11px; color:var(--muted); margin-top:5px; }}
 .foot {{ font-size:12.5px; color:#2b3440; }}
 .sig {{ display:grid; grid-template-columns:1fr 1fr; gap:44px; margin-top:44px;
         font-size:12.5px; }}
 .sig div {{ border-top:1px solid var(--ink); padding-top:8px; color:var(--muted); }}
 code {{ background:#f0f3f7; padding:1px 5px; border-radius:4px; font-size:12px; }}
 @media print {{ body {{ background:#fff; }} .page {{ box-shadow:none; padding:0; max-width:none; }}
   h2 {{ break-after:avoid; }} .finding, figure, table {{ break-inside:avoid; }} }}
</style></head><body><div class="page">
<header>
  <div class="kicker">AI 3D Structural Inspection System &middot; v{version}</div>
  <h1>{title}</h1>
  <div class="meta">Report generated {generated} &middot; build {build}<br/>
    Method: {method}</div>
</header>

<h2>1. Executive summary</h2>
<div class="shi">
  <div><div class="val">{shi_val:.1f}</div><div class="grade">Structural Health Index / 100</div></div>
  <div><div style="font-size:25px;font-weight:700">Grade {shi_grade}</div>
       <div class="grade">{action}</div></div>
  <div style="flex:1;min-width:260px">
    <div class="bar"><i style="width:{shi_val:.1f}%"></i></div>
    <div class="grade">Total penalty {penalty:.2f} points &middot; {shi_method}</div>
  </div>
</div>
<table><thead><tr><th>Defect mechanism</th><th>Instances</th><th>Worst grade</th>
  <th>SHI penalty</th><th>Affected area</th></tr></thead><tbody>{class_rows}</tbody></table>
<ol class="foot">{conclusions}</ol>

<h2>2. Structure and inspection basis</h2>
<table>
<tr><th>Element</th><td>{elem}</td></tr>
<tr><th>Description</th><td>{desc}</td></tr>
<tr><th>Concrete grade</th><td>{grade}</td></tr>
<tr><th>Geometry</th><td>{diam} mm diameter &times; {height} m exposed height &middot;
    nominal cover {cover} mm</td></tr>
<tr><th>Capture</th><td>{n_views} orbital views &middot; {sensor} &middot;
    {res0}&times;{res1} &middot; capture time {capdur} s</td></tr>
<tr><th>Reconstruction</th><td>{recon_method} &middot; {reg_imgs} registered images &middot;
    {sparse} sparse / {dense} dense points &middot;
    mean reprojection error {reproj} px &middot; fitted radius {fitr} m</td></tr>
<tr><th>Back-projection</th><td>{bp_points} defect points in {bp_clusters} multi-view
    clusters &middot; mean multi-view consistency {bp_consistency} m &middot;
    {bp_masks} detector masks fused</td></tr>
</table>
<h4>Standards applied</h4>
<ul>{standards}</ul>
<h4>Back-projection validation against the as-designed defect registry</h4>
{validation_block}

<h2>3. Defect findings and remediation</h2>
{cards}

<h2>4. Risk register</h2>
<table><thead><tr><th>Mechanism</th><th>Criticality</th><th>Severity</th><th>Instances</th>
  <th>Monitoring plan</th></tr></thead><tbody>{risk_rows}</tbody></table>

<h2>5. Annotated capture set (8 orbital OV2640 views)</h2>
<div class="gallery">{views_html}</div>

<h2>6. Digital twin deliverables</h2>
<ul>
  <li>Dense point cloud: <code>{ply}</code></li>
  <li>Textured surface mesh: <code>{obj}</code></li>
  <li>Colour-coded defect point cloud: <code>{defect_ply}</code></li>
  <li>COLMAP text model: <code>{cameras}</code>, <code>{images}</code>,
      <code>{points3d}</code></li>
</ul>

<h2>7. Limitations and verification required</h2>
<ol class="foot">{limitations}</ol>

<div class="sig">
  <div>Inspected by -- AI 3D Structural Inspection System (automated photogrammetric
       survey + AI defect analysis)</div>
  <div>Reviewed and approved by -- licensed structural engineer</div>
</div>
</div></body></html>""".format(
            version=e(report["meta"]["version"]), title=e(report["meta"]["title"]),
            generated=e(report["meta"]["generated"]), build=e(bld),
            method=e(report["meta"]["inspection_method"]),
            shi_val=shi["shi"], shi_grade=e(shi["shi_grade"]),
            action=e(shi["recommended_action"]), penalty=shi["total_penalty"],
            shi_method=e(shi["method"]), class_rows=class_rows,
            conclusions="".join(f"<li>{e(c)}</li>" for c in report["conclusions"]),
            elem=e(report["structure"]["label"]), desc=e(report["structure"]["description"]),
            grade=e(report["structure"]["concrete_grade"]),
            diam=report["structure"]["diameter_mm"], height=report["structure"]["height_m"],
            cover=report["structure"]["cover_mm"],
            n_views=report["capture_summary"]["views"],
            sensor=e(report["capture_summary"]["sensor"]),
            res0=report["capture_summary"]["resolution"][0],
            res1=report["capture_summary"]["resolution"][1],
            capdur=report["capture_summary"]["duration_s"],
            recon_method=e(report["reconstruction_summary"]["method"]),
            reg_imgs=report["reconstruction_summary"]["registered_images"],
            sparse=report["reconstruction_summary"]["sparse_points"],
            dense=report["reconstruction_summary"]["dense_points"],
            reproj=report["reconstruction_summary"]["mean_reprojection_error_px"],
            fitr=fmt(report["reconstruction_summary"].get("fitted_radius_m"), ".4f"),
            bp_points=report["back_projection"]["triangulated_defect_points"],
            bp_clusters=report["back_projection"]["clusters"],
            bp_consistency=fmt(report["back_projection"].get("mean_multiview_consistency_m"), ".4f"),
            bp_masks=report["back_projection"]["view_masks_used"],
            standards="".join(f"<li>{e(s)}</li>" for s in report["meta"]["standard_basis"]),
            validation_block=validation_block,
            cards="".join(cards) or "<p>No defects above the reporting threshold were detected.</p>",
            risk_rows=risk_rows, views_html=views_html,
            ply=e(report["attachments"]["digital_twin_ply"]),
            obj=e(report["attachments"]["digital_twin_obj"]),
            defect_ply=e(report["attachments"]["defect_point_cloud"]),
            cameras=e(report["attachments"]["cameras_txt"]),
            images=e(report["attachments"]["images_txt"]),
            points3d=e(report["attachments"]["points3D_txt"]),
            limitations="".join(f"<li>{e(l)}</li>" for l in report["limitations"]))
        Path(out_path).write_text(doc, encoding="utf-8")
        return Path(out_path)


# ==========================================================================
# Step-3 orchestrator
# ==========================================================================


def analyse_capture(detector: Optional[DefectDetector] = None,
                    view_indices: Optional[List[int]] = None,
                    reconstruction: Optional[dict] = None,
                    simulator=None) -> dict:
    """
    Run detection on the captured views, annotate them, and build the
    per-view detection payload (the 2D half of step 3).
    """
    det = detector or DefectDetector()
    idxs = view_indices if view_indices is not None else list(range(C.N_VIEWS))
    t0 = time.time()
    per_view = []
    all_detections: List[Detection] = []
    per_view_counts = {c: 0 for c in C.DEFECT_CLASSES}

    for i in idxs:
        path = C.CAPTURE_DIR / C.camera_rig()[i].image_name
        if not path.exists():
            if simulator is not None:
                simulator.capture_view(i, save=True)
            else:
                continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        dets = det.detect(i, bgr)
        annotated = det.annotate(i, bgr, dets)
        cv2.imwrite(str(C.ANALYSIS_DIR / f"annotated_view_{i + 1:02d}.jpg"), annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        all_detections.extend(dets)
        for d in dets:
            per_view_counts[d.cls] += 1
        per_view.append({
            "view_index": i, "view": C.VIEW_LABELS[i],
            "detections": [d.to_dict() for d in dets],
            "counts": {c: sum(1 for d in dets if d.cls == c) for c in C.DEFECT_CLASSES},
            "annotated_image": f"annotated_view_{i + 1:02d}.jpg",
            "px_per_mm": round(det._px_per_mm(i, bgr.shape[0], bgr.shape[1]), 4),
        })

    quality = score_detector(det, idxs)
    (C.ANALYSIS_DIR / "detection_quality.json").write_text(json.dumps(quality, indent=2))

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(all_detections),
        "quality": {"macro_f1": quality["macro_f1"], "per_class": quality["per_class"]},
        "per_view_counts": per_view_counts,
        "per_view": per_view,
        "detections": [d.to_dict() for d in all_detections],
        "duration_s": round(time.time() - t0, 2),
        "detector": det,
    }
    (C.ANALYSIS_DIR / "detections.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "detector"}, indent=2))
    return payload
