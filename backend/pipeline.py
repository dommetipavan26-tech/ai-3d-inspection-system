"""
backend/pipeline.py
===================
Orchestration of the four-step inspection workflow, plus the in-process job
runner that the dashboard polls while a run is in flight.

    Step 1  ESP32-CAM capture        esp32_simulator.ESP32CamSimulator
    Step 2  3D reconstruction        colmap_engine.ColmapEngine
    Step 3  AI defect analysis       ai_inspector.DefectDetector + report
    Step 4  2D->3D back-projection   raycast_3d.RayCaster3D

Every step is idempotent and can be re-run individually from the API, because
each step only depends on the *files* the previous step left on disk (images in
data/captures, the model in data/reconstruction, and so on).  That mirrors how
a field workflow actually behaves: you can re-shoot the capture set without
re-running the detector, and you can re-run the detector without re-shooting.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import config as C
from . import ai_inspector as AI
from . import colmap_engine as CE
from . import esp32_simulator as SIM
from . import raycast_3d as RC


# ==========================================================================
# Shared singletons (the simulator renders ~1.2 s/frame, so it is cached)
# ==========================================================================

_simulator: Optional[SIM.ESP32CamSimulator] = None
_detector: Optional[AI.DefectDetector] = None
_state_lock = threading.RLock()


def get_simulator(reset: bool = False, supersample: int = 1) -> SIM.ESP32CamSimulator:
    global _simulator
    with _state_lock:
        if _simulator is None or reset:
            _simulator = SIM.ESP32CamSimulator(supersample=supersample)
        return _simulator


def get_detector() -> AI.DefectDetector:
    global _detector
    with _state_lock:
        if _detector is None:
            _detector = AI.DefectDetector()
        return _detector


# ==========================================================================
# Step runners
# ==========================================================================


def step1_capture(reset: bool = False, save: bool = True) -> dict:
    """STEP 1 -- 8-view orbital OV2640 capture sweep."""
    sim = get_simulator(reset=reset)
    result = sim.capture_sweep(save=save)
    atlas = sim.save_texture_atlas()
    result["texture_atlas"] = Path(atlas).name
    result["label_stats"] = sim.label_stats()
    result["device_report"] = sim.fleet_status()
    (C.CAPTURE_DIR / "capture_manifest.json").write_text(json.dumps(result, indent=2))
    return result


def step2_reconstruct(use_native: bool = True) -> dict:
    """STEP 2 -- COLMAP / OpenCV hybrid reconstruction."""
    engine = CE.ColmapEngine(use_native=use_native)
    return engine.reconstruct()


def step3_analyse(reconstruction: Optional[dict] = None,
                  with_raycast: bool = True) -> dict:
    """
    STEP 3 + 4 -- detect defects on every view, build the report, back-project
    the defect centroids into world coordinates and colour the twin.
    """
    det = get_detector()
    detection = AI.analyse_capture(det, simulator=get_simulator(False))

    if reconstruction is None:
        reconstruction = _load_json(C.RECON_DIR / "reconstruction.json") or {}

    raycast: Dict[str, Any] = {}
    if with_raycast:
        caster = RC.RayCaster3D()
        raycast = caster.run(detection, reconstruction, det.masks_cache)
        (C.ANALYSIS_DIR / "detections.json").write_text(json.dumps(
            {k: v for k, v in detection.items() if k not in ("detector", "detections")}
            | {"detections": detection["detections"]}, indent=2))

    # ---- Structural Health Index + report ----------------------------
    glob = raycast.get("global_defects", [])
    if not glob:
        # fall back to per-view detections if step 4 produced no clusters
        glob = [{"class": d["cls"], "severity": d["severity"],
                 "area_m2": (d.get("area_px") or 0) * (0.25e-3 ** 2),
                 "max_width_mm": d.get("max_width_mm"), "views": [d["view"]],
                 "world_xyz": None, "surface_theta_deg": None,
                 "surface_elevation_m": None, "id": d["id"], "n_views": 1,
                 "world_xyz_uncertainty_m": None} for d in detection["detections"]]
    shi = AI.StructuralHealthIndex.compute(glob)
    reporter = AI.InspectionReport()
    capture = _load_json(C.CAPTURE_DIR / "capture_manifest.json") or {}
    report = reporter.build(capture=capture, reconstruction=reconstruction,
                            detection=detection, raycaster=raycast, shi=shi)
    report["meta"]["build_stamp"] = C.BUILD_STAMP
    report_json = C.ANALYSIS_DIR / "report.json"
    report_json.write_text(json.dumps(report, indent=2))
    report_html = reporter.render_html(report, C.ANALYSIS_DIR / "report.html")

    # ---- defect map unwrap (per-view detections painted on the atlas) ---
    _render_defect_map(detection)

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step3": {"total_detections": detection["total"],
                  "per_view_counts": detection["per_view_counts"],
                  "quality": detection.get("quality", {}),
                  "duration_s": detection["duration_s"]},
        "step4": raycast.get("stats", {}),
        "shi": shi,
        "validation": report.get("validation", {}),
        "report_json": str(report_json),
        "report_html": str(report_html),
        "capture": {"n_views": capture.get("n_views"), "duration_s": capture.get("duration_s")},
    }
    (C.ANALYSIS_DIR / "analysis.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_full_pipeline(progress: Optional[Callable[[str], None]] = None) -> dict:
    """All four steps, in order, with progress callbacks."""
    def say(msg):
        if progress:
            progress(msg)

    t0 = time.time()
    say("step 1/4 -- orbital ESP32-CAM capture sweep")
    cap = step1_capture(reset=False)
    say(f"step 1/4 -- captured {cap['n_views']} views in {cap['duration_s']} s")

    say("step 2/4 -- COLMAP 3D reconstruction")
    recon = step2_reconstruct()
    say(f"step 2/4 -- {recon['method']} produced "
        f"{recon['stats']['n_points3d']} sparse / {recon['stats']['dense_points']} dense points")

    say("step 3/4 -- AI defect analysis + report")
    analysis = step3_analyse(reconstruction=recon)
    say(f"step 3/4 -- {analysis['step3']['total_detections']} detections, "
        f"SHI {analysis['shi']['shi']:.1f} (grade {analysis['shi']['shi_grade']})")

    say("step 4/4 -- 2D->3D ray back-projection")
    say(f"step 4/4 -- {analysis['step4'].get('n_defect_points', 0)} defect points in "
        f"{analysis['step4'].get('n_clusters', 0)} clusters")

    result = {
        "status": "ok",
        "steps": {"capture": cap, "reconstruction": recon, "analysis": analysis},
        "total_duration_s": round(time.time() - t0, 2),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (C.DATA_DIR / "last_run.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "steps"} |
        {"steps": {kk: (vv if not isinstance(vv, dict) else
                        {k2: v2 for k2, v2 in vv.items()
                         if k2 not in ("views", "per_view", "device_report",
                                       "pair_geometry", "cameras", "log")})
                   for kk, vv in result["steps"].items()}}, indent=2))
    return result


# ==========================================================================
# Background job runner (single-flight, poll-able from the dashboard)
# ==========================================================================


class JobRunner:
    """Runs the pipeline in a worker thread and exposes its log + status."""

    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.progress: List[str] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.started: Optional[str] = None
        self.finished: Optional[str] = None
        self.thread: Optional[threading.Thread] = None
        self.duration_s: Optional[float] = None

    def status(self) -> dict:
        return {
            "running": self.running,
            "started": self.started,
            "finished": self.finished,
            "duration_s": self.duration_s,
            "progress": self.progress[-60:],
            "error": self.error,
            "has_result": self.result is not None,
        }

    def _say(self, msg: str):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        with self.lock:
            self.progress.append(line)
        print(line, flush=True)

    def start(self, supersample: int = 1) -> dict:
        with self.lock:
            if self.running:
                return {"started": False, "reason": "a run is already in progress",
                        **self.status()}
            self.running = True
            self.progress = []
            self.result = None
            self.error = None
            self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.finished = None
            t0 = time.time()

            def worker():
                try:
                    get_simulator(reset=True, supersample=supersample)
                    res = run_full_pipeline(self._say)
                    with self.lock:
                        self.result = res
                except Exception as exc:                          # pragma: no cover
                    with self.lock:
                        self.error = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                finally:
                    with self.lock:
                        self.running = False
                        self.duration_s = round(time.time() - t0, 2)
                        self.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")

            self.thread = threading.Thread(target=worker, name="inspection-pipeline",
                                           daemon=True)
            self.thread.start()
            return {"started": True, **self.status()}


JOB = JobRunner()


# ==========================================================================
# Helpers
# ==========================================================================


def _load_json(path: Path) -> Optional[dict]:
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text())
    except Exception:
        return None
    return None


def _render_defect_map(detection: dict) -> Optional[str]:
    """
    Paint the per-view detections onto the cylindrical unwrap so the report
    carries a single map of where the damage sits around the pier.
    """
    sim = get_simulator(False)
    albedo = sim.material.albedo
    H, W = albedo.shape[:2]
    canvas = (np.clip(albedo[..., ::-1], 0, 1) * 255).astype(np.uint8).copy()
    canvas = (canvas * 0.55 + 118).astype(np.uint8)

    painted = 0
    for pd in detection.get("per_view", []):
        vi = pd["view_index"]
        view = sim.views[vi]
        for d in pd["detections"]:
            cls = d["cls"]
            rgb = np.array(C.DEFECT_META[cls]["rgb"])
            x, y, w, h = d["bbox"]
            # sample a grid of pixels inside the box, back-project each onto the
            # pier surface by ray casting, and mark the unwrap texel
            gx = np.linspace(x, x + w, 7)
            gy = np.linspace(y, y + h, 7)
            GU, GV = np.meshgrid(gx, gy)
            uv = np.stack([GU.ravel(), GV.ravel()], axis=1)
            K_inv = np.linalg.inv(view.K)
            d_cam = (K_inv @ np.hstack([uv, np.ones((len(uv), 1))]).T).T
            d_world = d_cam @ view.R
            d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
            o = view.center
            a = d_world[:, 0] ** 2 + d_world[:, 2] ** 2
            b = 2.0 * (o[0] * d_world[:, 0] + o[2] * d_world[:, 2])
            c = o[0] ** 2 + o[2] ** 2 - C.COLUMN_RADIUS_M ** 2
            disc = b * b - 4 * a * c
            ok = disc > 0
            if not np.any(ok):
                continue
            t = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
            P = o.reshape(1, 3) + t[:, None] * d_world[ok]
            th = np.arctan2(P[:, 0], P[:, 2])
            col = np.clip(((th / (2 * np.pi)) + 0.5) * W - 0.5, 0, W - 1).astype(int)
            row = np.clip((1.0 - P[:, 1] / C.COLUMN_HEIGHT_M) * H - 0.5, 0, H - 1).astype(int)
            canvas[row, col] = rgb[::-1]
            painted += len(row)

    import cv2
    out = C.ANALYSIS_DIR / "defect_map.jpg"
    cv2.imwrite(str(out), cv2.resize(canvas, (1536, 768), interpolation=cv2.INTER_NEAREST),
                [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return str(out) if painted else None


def state_summary() -> dict:
    """Everything the dashboard needs in a single request."""
    sim = get_simulator(False)
    recon = _load_json(C.RECON_DIR / "reconstruction.json")
    analysis = _load_json(C.ANALYSIS_DIR / "analysis.json")
    report = _load_json(C.ANALYSIS_DIR / "report.json")
    raycast = _load_json(C.TWIN_DIR / "raycast.json")
    capture = _load_json(C.CAPTURE_DIR / "capture_manifest.json")

    files = []
    for label, p in (("cameras.txt", C.RECON_DIR / "cameras.txt"),
                     ("images.txt", C.RECON_DIR / "images.txt"),
                     ("points3D.txt", C.RECON_DIR / "points3D.txt"),
                     ("digital_twin.ply", C.RECON_DIR / "digital_twin.ply"),
                     ("digital_twin.obj", C.RECON_DIR / "digital_twin.obj"),
                     ("defect_points.ply", C.RECON_DIR / "defect_points.ply"),
                     ("report.html", C.ANALYSIS_DIR / "report.html"),
                     ("report.json", C.ANALYSIS_DIR / "report.json"),
                     ("pier_texture_atlas.jpg", C.ANALYSIS_DIR / "pier_texture_atlas.jpg"),
                     ("defect_map.jpg", C.ANALYSIS_DIR / "defect_map.jpg")):
        files.append({"name": label, "exists": p.exists(),
                      "bytes": p.stat().st_size if p.exists() else 0,
                      "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                      .isoformat(timespec="seconds") if p.exists() else None,
                      "kind": "reconstruction" if p.parent == C.RECON_DIR else "analysis"})

    return {
        "scene": C.scene_summary(),
        "capture": capture,
        "reconstruction": recon,
        "analysis": analysis,
        "report_meta": ({"shi": report["shi"], "meta": report["meta"],
                         "conclusions": report["conclusions"],
                         "findings": [{k: v for k, v in f.items() if k != "locations"}
                                      for f in report["findings"]]} if report else None),
        "raycast": raycast,
        "fleet": sim.fleet_status(),
        "files": files,
        "job": JOB.status(),
        "captures": [{"view": C.VIEW_LABELS[i], "index": i,
                      "filename": sim.views[i].image_name,
                      "exists": (C.CAPTURE_DIR / sim.views[i].image_name).exists(),
                      "device": sim.devices[i].device_id,
                      "rssi_dbm": sim.devices[i].rssi_dbm,
                      "azimuth_deg": i * 45, "compass": C.COMPASS[i],
                      "url": f"/captures/{sim.views[i].image_name}",
                      # the annotated frame is only produced once step 3 has run
                      "annotated_url": f"/analysis/annotated_view_{i + 1:02d}.jpg",
                      "annotated": (C.ANALYSIS_DIR
                                    / f"annotated_view_{i + 1:02d}.jpg").exists(),
                      "ground_truth_url": f"/ground_truth/gt_overlay_view_{i + 1:02d}.jpg"}
                     for i in range(C.N_VIEWS)],
    }
