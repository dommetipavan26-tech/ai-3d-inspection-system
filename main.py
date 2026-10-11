"""
main.py -- AI 3D Structural Inspection System (Render entrypoint)
=================================================================

FastAPI application that serves:

  * the interactive Three.js WebGL digital-twin dashboard (``/``)
  * the ESP32-CAM ingestion API used by the 16-pin AI-Thinker firmware
    (``/api/esp32/upload`` multipart HTTPS, ``/api/esp32/command`` 3 s poll)
  * the four-step pipeline API (capture -> reconstruct -> analyse -> raycast)
  * every generated artefact (JPEGs, COLMAP models, PLY/OBJ, PDF-ready report)

Run locally
-----------
    uvicorn main:app --host 0.0.0.0 --port 8000

On Render this module is the start command; see ``render.yaml`` and
``Dockerfile``.  ``PORT`` is honoured for platforms that inject it.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import (BackgroundTasks, FastAPI, File, Form, Header, HTTPException,
                     Query, Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, RedirectResponse)
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config as C                    # noqa: E402
from backend import pipeline as P                  # noqa: E402
from backend import colmap_engine as CE            # noqa: E402
from backend import raycast_3d as RC               # noqa: E402
from backend import ai_inspector as AI             # noqa: E402

APP_START = datetime.now(timezone.utc)

app = FastAPI(
    title=f"{C.PROJECT_NAME} API",
    version=C.PROJECT_VERSION,
    description=(
        "Full-stack AI 3D structural inspection: 8-view orbital ESP32-CAM "
        "(OV2640) capture, COLMAP 3D reconstruction, AI multi-stage defect "
        "analysis, and 2D->3D ray back-projection onto an interactive WebGL "
        "digital twin."),
    docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                 # field nodes post from arbitrary IPs
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = ROOT / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
app.mount("/captures", StaticFiles(directory=str(C.CAPTURE_DIR)), name="captures")
app.mount("/analysis", StaticFiles(directory=str(C.ANALYSIS_DIR)), name="analysis")
app.mount("/ground_truth", StaticFiles(directory=str(C.GT_DIR)), name="ground_truth")
app.mount("/model", StaticFiles(directory=str(C.RECON_DIR)), name="model")
app.mount("/twin", StaticFiles(directory=str(C.TWIN_DIR)), name="twin")


# ==========================================================================
# Dashboard
# ==========================================================================


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    page = FRONTEND / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>AI 3D Structural Inspection System</h1>"
                            "<p>frontend/index.html is missing.</p>", status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/report", include_in_schema=False)
def report_page():
    page = C.ANALYSIS_DIR / "report.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return RedirectResponse("/")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok", "uptime_s": (datetime.now(timezone.utc) - APP_START).total_seconds(),
            "data_dir": str(C.DATA_DIR), "version": C.PROJECT_VERSION}


# ==========================================================================
# Step 1 -- ESP32-CAM capture + ingestion
# ==========================================================================


@app.get("/api/status")
def api_status():
    """Everything the dashboard needs, in one poll."""
    return P.state_summary()


@app.get("/api/scene")
def api_scene():
    return C.scene_summary()


@app.get("/api/esp32/fleet")
def api_fleet():
    return P.get_simulator(False).fleet_status()


@app.post("/api/esp32/capture")
def api_capture(reset: bool = Query(False, description="rebuild the texture atlas"),
                save: bool = Query(True)):
    """Trigger the simulated 8-node orbital sweep (the dashboard 'CAPTURE')."""
    return P.step1_capture(reset=reset, save=save)


@app.post("/api/esp32/trigger/{view_index}")
def api_trigger(view_index: int):
    """Trigger a single ESP32-CAM node, exactly like the firmware command."""
    if not 0 <= view_index < C.N_VIEWS:
        raise HTTPException(404, f"view index must be 0..{C.N_VIEWS - 1}")
    sim = P.get_simulator(False)
    sim.commands.push("capture", view=f"V{view_index + 1}")
    result = sim.capture_view(view_index, save=True)
    result["command"] = sim.commands.status()["queue"][-1] if sim.commands.queue else None
    return result


@app.get("/api/esp32/command")
def api_esp32_command(device: str = Query("", description="ESP32-CAM device id"),
                      view: Optional[int] = Query(None)):
    """
    Polled by the firmware every 3 s.  Returns the next queued command or
    ``idle`` with the recommended sleep interval.
    """
    sim = P.get_simulator(False)
    cmd = sim.commands.pop(device)
    if view is not None and 0 <= view < C.N_VIEWS:
        cmd.setdefault("params", {})["view"] = view
    cmd["server_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cmd["sleep_ms"] = 3000
    cmd["capture_interval_s"] = int(os.environ.get("ESP32_CAPTURE_INTERVAL_S", 900))
    cmd["upload_path"] = "/api/esp32/upload"
    return cmd


@app.post("/api/esp32/queue")
def api_esp32_queue(command: str = Query("capture"),
                    device: str = Query(""),
                    view: Optional[int] = Query(None)):
    """Queue a command for the next firmware poll (capture/flash/status/reboot)."""
    sim = P.get_simulator(False)
    params = {"view": view} if view is not None else {}
    item = sim.commands.push(command, device=device, **params)
    return {"queued": item, "pending": len(sim.commands.queue)}


@app.post("/api/esp32/upload")
async def api_esp32_upload(
        file: UploadFile = File(..., description="JPEG captured by the OV2640"),
        device_id: str = Form(""),
        view: str = Form(""),
        rssi: str = Form(""),
        battery: str = Form(""),
        firmware: str = Form(""),
        x_inspection_token: Optional[str] = Header(None, alias="X-Inspection-Token")):
    """
    HTTPS multipart endpoint the 16-pin AI-Thinker ESP32-CAM posts to.

    The firmware sends ``file`` (the JPEG), ``device_id``, ``view``, ``rssi``,
    ``battery`` and ``firmware``.  When ``ESP32_UPLOAD_TOKEN`` is set on the
    server the request must carry a matching ``X-Inspection-Token`` header.
    """
    if C.ESP32_UPLOAD_TOKEN and x_inspection_token != C.ESP32_UPLOAD_TOKEN:
        raise HTTPException(401, "invalid or missing X-Inspection-Token")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "image larger than 12 MB")

    view_index = None
    if view:
        v = view.strip().upper().lstrip("V")
        if v.isdigit() and 1 <= int(v) <= C.N_VIEWS:
            view_index = int(v) - 1
    sim = P.get_simulator(False)
    result = sim.ingest_upload(file.filename or "esp32_capture.jpg", data,
                               view_index=view_index, device_id=device_id)
    result["telemetry"] = {"device_id": device_id, "view": view, "rssi_dbm": rssi,
                           "battery_v": battery, "firmware": firmware}
    result["server_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


@app.get("/api/esp32/upload", include_in_schema=False)
def api_esp32_upload_probe():
    """Friendly response when a node probes the endpoint with GET."""
    return {"endpoint": "/api/esp32/upload", "method": "POST",
            "content_type": "multipart/form-data",
            "fields": ["file", "device_id", "view", "rssi", "battery", "firmware"],
            "auth_header": "X-Inspection-Token (required only when the server sets "
                           "ESP32_UPLOAD_TOKEN)"}


@app.post("/api/upload/browser")
async def api_browser_upload(files: List[UploadFile] = File(...),
                             view: Optional[str] = Form(None)):
    """Browser photo upload (drag & drop onto the dashboard)."""
    sim = P.get_simulator(False)
    out = []
    for k, f in enumerate(files):
        data = await f.read()
        vi = None
        if view and view.isdigit():
            vi = int(view) - 1
        elif len(files) == C.N_VIEWS:
            vi = k
        out.append(sim.ingest_upload(f.filename or f"browser_{k}.jpg", data,
                                     view_index=vi,
                                     device_id="browser-upload"))
    return {"uploaded": len(out), "files": out}


# ==========================================================================
# Step 2 -- reconstruction
# ==========================================================================


@app.post("/api/reconstruct")
def api_reconstruct(use_native: bool = Query(True, description="use the COLMAP CLI if present")):
    try:
        return P.step2_reconstruct(use_native=use_native)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.get("/api/reconstruct/status")
def api_reconstruct_status():
    engine = CE.ColmapEngine(use_native=True)
    return {
        "colmap_binary": engine.colmap_bin,
        "colmap_version": engine.colmap_version(),
        "captures_present": [i for i in range(C.N_VIEWS)
                             if (C.CAPTURE_DIR / engine.views[i].image_name).exists()],
        "reconstruction_exists": (C.RECON_DIR / "reconstruction.json").exists(),
        "files": {n: (C.RECON_DIR / n).exists() for n in
                  ("cameras.txt", "images.txt", "points3D.txt",
                   "digital_twin.ply", "digital_twin.obj")},
    }


@app.get("/api/reconstruct/cameras")
def api_cameras():
    """Camera poses + intrinsics for the WebGL frustums and triangulation rays."""
    recon = P._load_json(C.RECON_DIR / "reconstruction.json")
    nominal = [v.to_dict() for v in C.camera_rig()]
    return {"nominal_rig": nominal,
            "reconstructed": (recon or {}).get("cameras", []),
            "method": (recon or {}).get("method"),
            "matrix": (recon or {}).get("stats", {}).get("matrix")}


# ==========================================================================
# Step 3 -- analysis + report
# ==========================================================================


@app.post("/api/analyse")
def api_analyse(with_raycast: bool = Query(True)):
    try:
        return P.step3_analyse(with_raycast=with_raycast)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.get("/api/detections")
def api_detections():
    return P._load_json(C.ANALYSIS_DIR / "detections.json") or {"detections": [], "per_view": []}


@app.get("/api/report")
def api_report():
    rep = P._load_json(C.ANALYSIS_DIR / "report.json")
    if not rep:
        raise HTTPException(404, "no report yet -- POST /api/analyse first")
    return rep


@app.get("/api/report.html", response_class=HTMLResponse)
def api_report_html():
    page = C.ANALYSIS_DIR / "report.html"
    if not page.exists():
        raise HTTPException(404, "no report yet -- POST /api/analyse first")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# ==========================================================================
# Step 4 -- ray back-projection payload for the WebGL twin
# ==========================================================================


@app.get("/api/twin")
def api_twin():
    """Defect clusters, camera-to-defect rays and geometry for the 3D viewer."""
    rc = P._load_json(C.TWIN_DIR / "raycast.json")
    recon = P._load_json(C.RECON_DIR / "reconstruction.json")
    if not rc:
        raise HTTPException(404, "no back-projection yet -- POST /api/analyse first")
    cameras = (recon or {}).get("cameras") or [v.to_dict() for v in C.camera_rig()]
    return {
        "generated": rc.get("generated"),
        "n_clusters": rc.get("n_clusters"),
        "global_defects": rc.get("global_defects", []),
        "hits": rc.get("hits", []),
        "stats": rc.get("stats", {}),
        "cameras": cameras,
        "scene": C.scene_summary(),
        "assets": {
            "point_cloud": "/model/digital_twin.ply",
            "mesh": "/model/digital_twin.obj",
            "defect_points": "/model/defect_points.ply",
            "texture": "/model/digital_twin_texture.jpg",
            "annotated_views": [f"/analysis/annotated_view_{i + 1:02d}.jpg"
                                for i in range(C.N_VIEWS)],
        },
    }


@app.post("/api/raycast")
def api_raycast():
    det = P._load_json(C.ANALYSIS_DIR / "detections.json") or {}
    recon = P._load_json(C.RECON_DIR / "reconstruction.json")
    caster = RC.RayCaster3D()
    return caster.run(det, recon, P.get_detector().masks_cache)


# ==========================================================================
# Pipeline control
# ==========================================================================


@app.post("/api/pipeline/run")
def api_pipeline_run(supersample: int = Query(1, ge=1, le=3)):
    """Kick off the full four-step run in the background."""
    return P.JOB.start(supersample=supersample)


@app.get("/api/pipeline/status")
def api_pipeline_status():
    st = P.JOB.status()
    st["result_summary"] = (
        {k: v for k, v in (P.JOB.result or {}).items() if k != "steps"}
        if P.JOB.result else None)
    return st


@app.post("/api/reset")
def api_reset(what: str = Query("all", pattern="^(all|captures|reconstruction|analysis)$")):
    """Wipe generated artefacts (never touches source code)."""
    import shutil
    targets = {
        "all": (C.CAPTURE_DIR, C.GT_DIR, C.RECON_DIR, C.ANALYSIS_DIR, C.TWIN_DIR,
                C.UPLOAD_DIR),
        "captures": (C.CAPTURE_DIR, C.GT_DIR),
        "reconstruction": (C.RECON_DIR, C.TWIN_DIR),
        "analysis": (C.ANALYSIS_DIR,),
    }[what]
    removed = []
    for d in targets:
        if d.exists() and d != C.DATA_DIR:
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            removed.append(str(d.relative_to(C.DATA_DIR)))
    return {"reset": what, "cleared": removed}


# ==========================================================================
# Artefacts
# ==========================================================================


@app.get("/api/files")
def api_files():
    return {"files": P.state_summary()["files"]}


@app.get("/api/download/{kind}")
def api_download(kind: str):
    mapping = {
        "twin_ply": (C.RECON_DIR / "digital_twin.ply", "application/octet-stream"),
        "twin_obj": (C.RECON_DIR / "digital_twin.obj", "text/plain"),
        "twin_mtl": (C.RECON_DIR / "digital_twin.mtl", "text/plain"),
        "twin_texture": (C.RECON_DIR / "digital_twin_texture.jpg", "image/jpeg"),
        "defect_ply": (C.RECON_DIR / "defect_points.ply", "application/octet-stream"),
        "cameras": (C.RECON_DIR / "cameras.txt", "text/plain"),
        "images": (C.RECON_DIR / "images.txt", "text/plain"),
        "points3d": (C.RECON_DIR / "points3D.txt", "text/plain"),
        "reconstruction_json": (C.RECON_DIR / "reconstruction.json", "application/json"),
        "report_html": (C.ANALYSIS_DIR / "report.html", "text/html"),
        "report_json": (C.ANALYSIS_DIR / "report.json", "application/json"),
        "detections": (C.ANALYSIS_DIR / "detections.json", "application/json"),
        "texture_atlas": (C.ANALYSIS_DIR / "pier_texture_atlas.jpg", "image/jpeg"),
        "defect_map": (C.ANALYSIS_DIR / "defect_map.jpg", "image/jpeg"),
        "run_log": (C.DATA_DIR / "last_run.json", "application/json"),
    }
    if kind not in mapping:
        raise HTTPException(404, f"unknown artefact '{kind}'. "
                                 f"Available: {', '.join(sorted(mapping))}")
    path, media = mapping[kind]
    if not path.exists():
        raise HTTPException(404, f"{kind} has not been generated yet")
    return FileResponse(str(path), media_type=media, filename=path.name)


@app.get("/api/captures")
def api_captures():
    sim = P.get_simulator(False)
    out = []
    for i, v in enumerate(sim.views):
        p = C.CAPTURE_DIR / v.image_name
        out.append({"view": v.label, "index": i, "compass": C.COMPASS[i],
                    "azimuth_deg": i * 45, "device_id": v.device_id,
                    "filename": v.image_name, "exists": p.exists(),
                    "bytes": p.stat().st_size if p.exists() else 0,
                    "url": f"/captures/{v.image_name}",
                    "annotated_url": f"/analysis/annotated_view_{i + 1:02d}.jpg",
                    "ground_truth_url": f"/ground_truth/gt_overlay_view_{i + 1:02d}.jpg"})
    return {"n_views": C.N_VIEWS, "views": out}


@app.get("/api/firmware", response_class=PlainTextResponse)
def api_firmware():
    """Serve the Arduino sketch straight from the repository."""
    p = ROOT / "esp32_cam_ov2640" / "esp32_cam_inspection.ino"
    if not p.exists():
        raise HTTPException(404, "firmware not found")
    return p.read_text(encoding="utf-8")


@app.get("/api/platformio", response_class=PlainTextResponse)
def api_platformio():
    p = ROOT / "esp32_cam_ov2640" / "platformio.ini"
    if not p.exists():
        raise HTTPException(404, "platformio.ini not found")
    return p.read_text(encoding="utf-8")


# ==========================================================================
# Developer entrypoint
# ==========================================================================


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
