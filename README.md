# AI 3D Structural Inspection System

An end-to-end structural-inspection platform: a fleet of **16-pin AI-Thinker ESP32-CAM (OV2640)**
nodes photographs a concrete pier from eight orbital views, the photos are turned into a metric
**3D digital twin**, a multi-stage OpenCV detector finds the four classic reinforced-concrete
defect mechanisms, and every defect is **back-projected into world coordinates** as a colour-coded
point on the twin. The whole thing runs as one FastAPI service with an interactive Three.js
console, and deploys to Render with a single click.

```
ESP32-CAM fleet  ──HTTPS──►  Step 1 capture  ──►  Step 2 reconstruction  ──►  Step 3 AI detection
  (16-pin OV2640)             8 orbital views      COLMAP + OpenCV SfM         crack / corrosion /
       ▲                      + ground-truth          cameras/images/            spalling / efflorescence
       │                          masks                 points3D + PLY/OBJ              │
       └──────── 3 s command poll ◄── job queue ◄───────────── Step 4 2D→3D ray back-projection ◄┘
                                                                   (K⁻¹, poses R|t, ray triangulation)
```

---

## 1. What it does

| Step | Module | Output |
|---|---|---|
| **1 — Capture** | `backend/esp32_simulator.py`, `esp32_cam_ov2640/esp32_cam_inspection.ino` | 8 × 1600×1200 OV2640 JPEGs on a 45° orbital rig, depth maps, pixel-accurate defect label maps, capture manifest |
| **2 — Reconstruct** | `backend/colmap_engine.py` | `cameras.txt`, `images.txt`, `points3D.txt`, `digital_twin.ply`, `digital_twin.obj` + texture, `reconstruction.json` |
| **3 — Analyse** | `backend/ai_inspector.py` | Per-view defect polygons, annotated images, `detection_quality.json`, `report.json` + printable `report.html`, Structural Health Index |
| **4 — Localise** | `backend/raycast_3d.py` | Multi-view defect clusters in world XYZ, `defect_points.ply`, class-stamped twin cloud, `raycast.json` |

The front end (`frontend/`) renders the twin, the mesh, the internal rebar cage, the eight camera
frustums, the triangulation rays and the pulsing defect markers in one WebGL scene, and drives all
four steps from the side panel. **Three.js is vendored locally** (`frontend/three.min.js`,
`OrbitControls.js`, `PLYLoader.js`, `OBJLoader.js`) — no CDN, so the console works on an air-gapped
site network.

---

## 2. Quick start (local)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000  ->  press "Run full pipeline"  (~40 s)
```

*(On a distribution that owns its Python you will need
`pip install --break-system-packages -r requirements.txt`, or the venv above.)*

No GPU, no external binaries and no network access are required: the ESP32 fleet is simulated in
process, and `colmap` is used automatically if it happens to be installed. The first run generates
everything under `data/`.

Every artefact is also reachable directly:

* `/` — interactive console · `/report` — printable engineering report
* `/api/docs` — OpenAPI/Swagger for the whole REST surface
* `/model/digital_twin.ply` · `/model/digital_twin.obj` · `/twin/raycast.json`

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data` | Root for captures, reconstruction, analysis and twin artefacts |
| `PORT` | `8000` | HTTP port (Render sets this automatically) |
| `ESP32_UPLOAD_TOKEN` | *(unset)* | When set, `POST /api/esp32/upload` requires a matching `X-Inspection-Token` header |
| `COLMAP_BINARY` | `colmap` | Path to the native COLMAP CLI used by the hybrid engine |
| `OPENCV_IO_MAX_IMAGE_PIXELS` | `0` | Lets OpenCV read the 4096×2048 texture atlas |

---

## 3. The ESP32-CAM node

### Hardware — 16-pin AI-Thinker ESP32-CAM with OV2640

Field wiring is deliberately minimal: **only 5 V and GND** reach the module. The OV2640 connects
over the module's own internal PCB traces, so no camera jumper wires exist to shake loose.

| Signal | GPIO | Notes |
|---|---|---|
| Countdown LED | **33** | 3 s pre-flash countdown, visible to the operator |
| Flash LED | **4** | 120 ms strobe, fired after the countdown |
| Battery sense | 35 | 2:1 divider on the ADC, reported with every upload |
| Camera bus | on-board | PWDN/XCLK/SIOD/SIOC/Y2–Y9/VSYNC/HREF/PCLK as per AI-Thinker |

### Firmware

`esp32_cam_ov2640/esp32_cam_inspection.ino` (Arduino) + `platformio.ini`:

1. A **3-second countdown** on GPIO 33, then a **120 ms flash** on GPIO 4, then the frame is
   captured at `FRAMESIZE_UXGA` / JPEG quality 12 (~200 KB).
2. The JPEG is streamed to `POST /api/esp32/upload` as `multipart/form-data` (`file`, `device_id`,
   `view`, `rssi`, `battery`, `firmware`) over `WiFiClientSecure` — raw-socket streaming, so the
   frame never has to fit in RAM twice. TLS validates against the ISRG Root X1 CA; `setInsecure()`
   is available for field recovery.
3. Every 3 s it polls `GET /api/esp32/command` and executes `capture`, `flash`, `set_interval`,
   `status` or `reboot`.
4. A 24 h watchdog restarts the node; a boot self-test reports to the serial console.

Build and flash:

```bash
pio run -e esp32cam -t upload     # USB-TTL on U0TXD/U0RXD with GPIO0 tied to GND
pio device monitor -b 115200
```

`GET /api/firmware` serves the sketch and `GET /api/platformio` the build file, so the console can
show the pin map without a separate download.

---

## 4. REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | One-poll dashboard state: scene, fleet, captures, reconstruction, analysis, raycast, files, job |
| `GET` | `/api/scene` | Pier geometry, rig definition, defect registry, camera intrinsics |
| `GET` | `/api/esp32/fleet` | All eight nodes: status, view, RSSI, battery, firmware, upload count |
| `POST` | `/api/esp32/capture` | Run the 8-view orbital sweep |
| `POST` | `/api/esp32/trigger/{view}` | Capture one view |
| `GET` | `/api/esp32/command` | 3 s long-poll command channel for the firmware |
| `POST` | `/api/esp32/queue` | Queue a command for a node |
| `POST` | `/api/esp32/upload` | **Firmware upload endpoint** (multipart, optional token) |
| `POST` | `/api/upload/browser` | Upload photos straight from a phone or laptop |
| `POST` | `/api/reconstruct` | Run step 2 (native COLMAP if present + OpenCV SfM) |
| `GET` | `/api/reconstruct/cameras` | Recovered poses, intrinsics and R/t per view |
| `POST` | `/api/analyse` | Run step 3 (+ step 4) — detections, SHI, report |
| `GET` | `/api/detections` · `/api/report` · `/api/report.html` | Raw detections / JSON report / printable report |
| `GET` | `/api/twin` · `POST` `/api/raycast` | Digital twin clusters / re-run back-projection |
| `POST` | `/api/pipeline/run` · `GET` `/api/pipeline/status` | Start the 4-step job, poll its progress log |
| `GET` | `/api/files` · `/api/download/{kind}` | Artefact inventory / zip download |
| `GET` | `/healthz` | Render health check |

---

## 5. Deploy to Render

`render.yaml` provisions everything (a `standard` instance is recommended — the reconstruction is
CPU-heavy):

```yaml
services:
  - type: web
    name: ai-3d-inspection-system
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /healthz
    envVars: [ PORT, DATA_DIR=/var/data, OPENCV_IO_MAX_IMAGE_PIXELS=0, ESP32_UPLOAD_TOKEN ]
    disk: { name: inspection-data, mountPath: /var/data, sizeGB: 5 }
```

Or push the included `Dockerfile` (Debian bookworm + `colmap` + OpenCV headless, `tini` as PID 1,
health check on `/healthz`). Either way the first request after deploy is a cold, empty install —
open the dashboard and press **Run full pipeline**; the 8 captures, the twin, the analysis and the
report are generated in about 40 seconds and stored on the persistent disk.

After deploy, set `INSPECTION_HOST` in `esp32_cam_inspection.ino` to your Render hostname and
re-flash the nodes.

---

## 6. Measured results

Numbers below are from a full `POST /api/pipeline/run` on the reference pier (`P-04 / C-102`,
600 mm diameter, 3.00 m exposed height) and are written to `data/` on every run.

### Reconstruction (step 2)

| Metric | Value |
|---|---|
| Registered views | 8 / 8 |
| SIFT keypoints | 48 002 across the capture set |
| Verified pair geometries | 28 pairs, 833 RANSAC inliers |
| Sparse points | 411 |
| Dense surface samples | 79 259 |
| Mean reprojection error | **0.486 px** (verified correspondences only) |
| Fitted cylinder radius | **0.2997 m** vs 0.3000 m nominal → **0.3 mm deviation**, 95.1 % inliers, 310° angular coverage |

The metric gauge is anchored to the commissioned survey of the rig. The monocular SfM bootstrap is
scale-free and, on a smooth cylinder photographed from a near-circular baseline, genuinely
degenerate — its fit to the survey is measured on every run (1.35 m RMS here) and when that residual
exceeds 150 mm the surveyed poses are kept as the reference and the deviation is reported in
`reconstruction.json` rather than silently trusted.

### Defect detection (step 3)

Detector quality is **measured against pixel-accurate ground-truth label maps on every run** — the
simulator emits them alongside the images — and persisted to `data/analysis/detection_quality.json`:

| Mechanism | Precision | Recall | F1 | Response |
|---|---|---|---|---|
| Diagonal shear crack | 0.705 | 0.381 | **0.494** | two-scale local-contrast band (`0.60·band41 + 0.40·band21`), skeleton length/width gate |
| Rebar corrosion | 0.995 | 0.849 | **0.916** | HSV hue gate (3–34°) × saturation × red dominance |
| Concrete spalling | 0.669 | 0.725 | **0.696** | additive blend of local texture energy and large-scale darkness |
| Efflorescence | 0.117 | 0.394 | 0.180 | bright/neutral contrast with an edge-preserving smooth — a genuinely diffuse mechanism, flagged *marginal* |
| **Macro** | | | **0.572** | |

The discriminating feature is **local contrast in a single scale band** (image minus its own local
mean). The pier's cylindrical shading and the 200 px-scale mottling of weathered concrete are both
low-frequency, so they cancel out of a local mean — that is what makes a 4 mm crack detectable at
all. Ridge/morphology operators (black-hat, Frangi) and multiplicative feature products were all
tried and measured *worse* (F1 0.067 and 0.042 respectively for the crack) and are not used.

### 3D localisation (step 4) — validated against the defect registry

The back-projected world centroid of each defect the pipeline finds is compared with its recorded
position. This is a measured accuracy statement, reproduced in `report.json` and in the HTML report:

| ID | Mechanism | Localisation error | Views | Result |
|---|---|---|---|---|
| D-01 | Diagonal shear crack | 46.9 mm | 2 | matched |
| D-02 | Rebar corrosion | 11.8 mm | 2 | matched |
| D-03 | Concrete spalling | 50.5 mm | 3 | matched |
| D-04 | Efflorescence | 167.9 mm | 3 | matched |

**4 of 4 known defects located** (250 mm match radius) · mean error **69.3 mm** · 16 multi-view
clusters from 23 rays (23/23 hit the reconstructed surface, mean consistency **6.5 mm**) · 246
defect points · 15 689 of 79 259 twin points class-stamped, giving a Structural Health Index of
**37.4 (grade E)** — a deliberately damaged reference pier.

---

## 7. Repository layout

```
main.py                       FastAPI app: REST surface, static mounts, firmware hosting
requirements.txt              pinned runtime dependencies
render.yaml / Dockerfile      one-click Render deployment (native or container)
backend/
  config.py                   pier geometry, 8-view rig, intrinsics, poses, defect registry
  esp32_simulator.py          physically-shaded OV2640 renderer + ground-truth masks + ingest
  colmap_engine.py            hybrid COLMAP-CLI / OpenCV SIFT-RANSAC-PnP reconstruction
  ai_inspector.py             4-class detector, quality scoring, SHI, ACI/EN report
  raycast_3d.py               K⁻¹/pose back-projection, ray triangulation, cluster fusion
  pipeline.py                 the four steps, job runner, dashboard state summary
frontend/
  index.html / styles.css     four-tab inspection console
  app.js                      Three.js scene: twin, mesh, rebar, frustums, rays, markers
  three.min.js, OrbitControls.js, PLYLoader.js, OBJLoader.js   vendored (no CDN)
esp32_cam_ov2640/
  esp32_cam_inspection.ino    16-pin AI-Thinker firmware (capture, flash, HTTPS upload, poll)
  platformio.ini              esp32cam + esp32cam-debug environments
data/                         generated at runtime (git-ignored)
```

---

## 8. Standards basis and limitations

The report is structured on **ACI 201.1R / ACI 364.1R** (condition assessment), **ACI 224R**
(cracking), **ACI 562** (repair) and **EN 1504-9 / EN 1992-1-1** (protection, repair, concrete
design), with **ISO 13822** for assessment of existing structures. Crack width thresholds follow
**EN 1992-1-1 Table 7.1N**; cover and corrosion assessment follow **EN 1504-9**.

This system is a survey and triage instrument, not a substitute for a licensed engineer:

* The detector is calibrated on the reference pier and reports its own measured quality on every
  run. The efflorescence class is deliberately reported as *marginal* — its response is broad and
  precise thresholds destroy its recall.
* One run measures **one facade wrap** of the element. Defects beyond the 310° covered arc are not
  observed.
* Back-projected positions inherit the reconstruction's accuracy: 0.486 px reprojection error and a
  0.3 mm radius deviation on the reference instrument, but absolute accuracy on a real structure
  depends on the survey of the camera rig.
* Every reported defect carries a severity, a residual uncertainty in millimetres and the views
  that corroborate it, so a reviewing engineer can see exactly how much to trust each callout.
