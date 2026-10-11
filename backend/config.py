"""
backend/config.py
=================
Single source of truth for the AI 3D Structural Inspection System.

Everything that describes *the structure under inspection* (metric geometry of
the concrete pier, the 8-camera orbital ESP32-CAM rig, camera intrinsics and
the four ground-truth structural defects) lives here so that the simulator,
the reconstruction engine, the AI detector and the ray back-projection stage
all speak exactly the same coordinate system.

World coordinate system
-----------------------
    +Y  ... vertical (up), y = 0.0 m is the foundation / floor level
    XZ  ... horizontal plane, column axis passes through the origin
    The column surface point at azimuth THETA and elevation Y is:

        P(theta, y) = ( R * sin(theta), y, R * cos(theta) )

    so theta = 0 points towards +Z ("south face" of the pier), theta grows
    counter-clockwise seen from above.

Camera pose convention (identical to COLMAP)
--------------------------------------------
    X_cam = R * X_world + t          (world -> camera)
    R ... 3x3 rotation,  t ... 3x1 translation
    projection:  p = K * (R * X + t),  u = px/pz, v = py/pz
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths / runtime configuration
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_NAME = "AI 3D Structural Inspection System"
PROJECT_VERSION = "1.0.0"
BUILD_STAMP = "2026-10-11"

# DATA_DIR is overridable so a Render persistent disk (e.g. /var/data) can be
# mounted without touching the code.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data")).resolve()
CAPTURE_DIR = DATA_DIR / "captures"          # Step 1 - 8 orbital frames
GT_DIR = DATA_DIR / "ground_truth"           # Step 1 - pixel label maps
RECON_DIR = DATA_DIR / "reconstruction"      # Step 2 - COLMAP text models + models
ANALYSIS_DIR = DATA_DIR / "analysis"         # Step 3 - annotated images + report
TWIN_DIR = DATA_DIR / "digital_twin"         # Step 4 - ray-cast defect clouds
UPLOAD_DIR = DATA_DIR / "uploads"            # raw ESP32 / browser uploads
LOG_DIR = DATA_DIR / "logs"

for _d in (DATA_DIR, CAPTURE_DIR, GT_DIR, RECON_DIR, ANALYSIS_DIR, TWIN_DIR,
           UPLOAD_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ESP32 upload secrets (set ESP32_UPLOAD_TOKEN on Render to lock the endpoint)
ESP32_UPLOAD_TOKEN = os.environ.get("ESP32_UPLOAD_TOKEN", "")

# --------------------------------------------------------------------------
# Structural geometry (metric, metres) -- a 600 mm diameter bridge pier
# --------------------------------------------------------------------------

COLUMN_RADIUS_M = 0.30          # 600 mm nominal diameter pier
COLUMN_HEIGHT_M = 3.00          # 3.0 m exposed height above foundation
COLUMN_LABEL = "PIER P-04 / COLUMN C-102"
COLUMN_DESCRIPTION = (
    "Reinforced concrete pier, 600 mm diameter, 3.0 m exposed height, "
    "C30/37 concrete, 40 mm nominal cover, epoxy-coated 8-D20 vertical "
    "reinforcement with D10@150 helical confinement (ACI 318 / EN 1992-1-1)"
)
CONCRETE_GRADE = "C30/37 (f'c = 30 MPa / 4350 psi)"
NOMINAL_COVER_MM = 40.0

# --------------------------------------------------------------------------
# The 8 x ESP32-CAM orbital capture rig
# --------------------------------------------------------------------------

N_VIEWS = 8
VIEW_LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

ORBIT_RADIUS_M = 2.60           # each ESP32-CAM node sits 2.6 m from the axis
ORBIT_HEIGHT_M = 1.35           # lens height above the foundation
LOOK_AT = np.array([0.0, 1.45, 0.0])
AZIMUTH_STEP_DEG = 360.0 / N_VIEWS   # 45 deg between the 8 nodes

# OV2640 sensor configuration used by the firmware (UXGA / JPEG)
IMAGE_W = 1600
IMAGE_H = 1200
SENSOR_NAME = "OV2640 2MP (UXGA 1600x1200)"
HORIZONTAL_FOV_DEG = 62.0       # OV2640 + standard M12 lens with 2x2 binning
JPEG_QUALITY = 88

# --------------------------------------------------------------------------
# Ground-truth structural defects
# --------------------------------------------------------------------------
# Four defect families, each physically anchored on the pier surface in
# (theta, y) surface coordinates so that the eight orbital frames see them
# from genuinely different parallax angles.

DEFECT_CLASSES = ["crack", "corrosion", "spalling", "efflorescence"]

DEFECT_META = {
    "crack": {
        "name": "Diagonal Shear Crack",
        "aci": "ACI 201.1R / ACI 224R",
        "en": "EN 1504-9 principle P4 (structural strengthening)",
        "color": "#ff2d2d",
        "rgb": (255, 45, 45),
        "severity_base": "High",
        "cause": ("Diagonal 45 deg shear cracking from principal tensile stress; "
                  "consistent with shear demand exceeding the diagonal cracking "
                  "capacity, or with a construction joint that was not "
                  "monolithically prepared."),
        "remediation": (
            "(1) Install a displacement/width gauge or crack comparator at the "
            "widest limb and monitor per ACI 224.1R for 30 days. "
            "(2) Crack widths > 0.30 mm that are dormant: pressure-inject low "
            "viscosity epoxy (ASTM C881 Type IV) full depth. "
            "(3) Active/widening cracks: route and seal with a flexible "
            "polyurethane (ASTM C920) so that movement does not tear the repair. "
            "(4) Where shear demand is the driver, externally bond UHM CFRP "
            "sheets oriented vertically to the crack direction or install a "
            "steel jacket per EN 1504-4, and re-verify the capacity in "
            "accordance with ACI 440.2R. "
            "(5) Instrument with a long-term crack monitor / strain gauge before "
            "closing out the repair."),
    },
    "corrosion": {
        "name": "Rebar Corrosion / Rust Staining",
        "aci": "ACI 201.1R / ACI 364.1R",
        "en": "EN 1504-9 principles P2, P3, P7, P10, P11",
        "color": "#ff8c00",
        "rgb": (255, 140, 0),
        "severity_base": "High",
        "cause": ("Carbonation front or chloride ingress has reached the depth "
                  "of the reinforcement, de-passivating the steel. Oxide volume "
                  "expansion (~6x) generates radial bursting stresses, hence the "
                  "longitudinal cracking and the rust bleeding observed on the "
                  "surface."),
        "remediation": (
            "(1) Profile the corroded zone with an electromagnetic cover meter "
            "and half-cell potential mapping (ASTM C876) to bound the anodic "
            "region -- repair 150 mm beyond any reading of -350 mV CSE. "
            "(2) Break out all delaminated and carbonated concrete to behind "
            "the rebar (min 25 mm clearance) using hydro-demolition or "
            "light chipping. "
            "(3) Grit-blast the reinforcement to Sa 2.5 (ISO 8501-1); replace "
            "any bar with > 10 % section loss or > 20 % of the cross-sectional "
            "area. "
            "(4) Apply a migrating corrosion inhibitor and a zinc-rich / "
            "cementitious passivating primer per EN 1504-7, then rebuild with "
            "an R4-class structural repair mortar (EN 1504-3). "
            "(5) Protect the surface with a breathable silane/siloxane "
            "water-repellent plus a carbonation-inhibiting coating "
            "(EN 1504-2, Class II) so the treatment reaches 40+ years of "
            "residual service life."),
    },
    "spalling": {
        "name": "Concrete Spalling / Delamination",
        "aci": "ACI 201.1R / ACI 364.1R / ACI 562",
        "en": "EN 1504-9 principles P3, P4, P7, P8",
        "color": "#ffd400",
        "rgb": (255, 212, 0),
        "severity_base": "High",
        "cause": ("Loss of concrete cover driven by expansion of the corroding "
                  "reinforcement behind the surface, aggravated by freeze-thaw "
                  "and by the fire/impact damage visible as unhydrated aggregate "
                  "exposure in the void."),
        "remediation": (
            "(1) Sound the full elevation with a hammer / chain drag and mark "
            "every drummy area; map delamination with impact-echo (ASTM C1383) "
            "or ultrasonic pulse-echo before any repair is designed. "
            "(2) Remove concrete to a square-edged, 25 mm minimum depth profile "
            "with no feathered edges; treat the exposed steel as per the "
            "corrosion remediation above. "
            "(3) Rebuild in lifts no greater than 40 mm with C60/75 R4 "
            "shrinkage-compensated repair mortar (EN 1504-3) and cure 7 days "
            "wet per ACI 308R. "
            "(4) Reinstate 40 mm cover, then apply a surface-applied "
            "carbonation barrier combined with a crack-bridging elastic "
            "coating at all construction joints. "
            "(5) Re-verify the section capacity; where > 15 % of the cover area "
            "of a single element is lost, uprate by FRP wrapping or a "
            "concrete jacket."),
    },
    "efflorescence": {
        "name": "Efflorescence / Moisture Seepage",
        "aci": "ACI 201.1R (cosmetic - indicator of transport)",
        "en": "EN 1504-9 principles P1, P2, P10",
        "color": "#4fc3f7",
        "rgb": (79, 195, 247),
        "severity_base": "Moderate",
        "cause": ("Soluble calcium hydroxide is being transported by water "
                  "through the cover and deposited as calcium carbonate on "
                  "evaporation. It is cosmetic in itself but is a direct "
                  "marker of a continuous water path (poor drainage, leaking "
                  "joint, or a cracked, non-waterproofed surface) which will "
                  "drive future corrosion."),
        "remediation": (
            "(1) Trace the moisture source and correct drainage / joint "
            "sealing first; the bloom will recur if the water path remains. "
            "(2) Dry-brush or low-pressure-wash the deposit (do NOT acid wash "
            "the surface - acid attacks the cement matrix). "
            "(3) Allow the concrete to dry to < 5 % moisture content, then "
            "impregnate with a silane/siloxane water repellent (EN 1504-2, "
            "Class I, >= 1.5 mm penetration) to break the capillary path. "
            "(4) Apply an anti-carbonation / anti-chloride coating on the "
            "affected elevation and re-inspect at 12 months."),
    },
}


@dataclass
class DefectSpec:
    """A single structural defect anchored on the pier surface."""
    id: str
    cls: str                 # one of DEFECT_CLASSES
    theta0: float            # start azimuth (rad)
    theta1: float            # end azimuth (rad)
    y0: float                # start elevation (m)
    y1: float                # end elevation (m)
    severity: str
    size_mm: float           # characteristic dimension (crack width / patch size)
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def theta_mid(self) -> float:
        return 0.5 * (self.theta0 + self.theta1)

    @property
    def y_mid(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    def surface_point(self, theta: float = None, y: float = None):
        th = self.theta_mid if theta is None else theta
        yy = self.y_mid if y is None else y
        return theta_y_to_xyz(th, yy)

    def surface_center(self) -> np.ndarray:
        """Ground-truth 3D centre of the defect on the pier surface."""
        return theta_y_to_xyz(self.theta_mid, self.y_mid)

    def to_dict(self) -> dict:
        c = self.surface_center()
        return {
            "id": self.id,
            "class": self.cls,
            "name": DEFECT_META[self.cls]["name"],
            "theta_deg": round(float(np.degrees(self.theta_mid)), 2),
            "elevation_m": round(float(self.y_mid), 3),
            "severity": self.severity,
            "size_mm": self.size_mm,
            "ground_truth_xyz": [round(float(v), 4) for v in c],
            "note": self.note,
        }


# --- the four defects the system has to find ------------------------------

def _default_defects():
    return [
        # 1 -----------------------------------------------------------------
        DefectSpec(
            id="D-01", cls="crack",
            theta0=np.radians(6.0), theta1=np.radians(35.5),
            y0=0.74, y1=1.72,
            severity="High", size_mm=1.8,
            note="Diagonal shear crack, 45 deg, branch network at mid-height.",
            extra=dict(
                # polyline in (azimuth deg, elevation m); the sim renders this
                # as a tapering crack so the CV detector has to work for it.
                poly=[(6.0, 0.74), (13.0, 1.02), (21.0, 1.30), (28.5, 1.55), (35.5, 1.72)],
                branches=[
                    [(13.0, 1.02), (18.5, 0.90), (24.0, 0.83)],
                    [(21.0, 1.30), (25.0, 1.48), (27.0, 1.68)],
                    [(28.5, 1.55), (32.0, 1.40), (35.0, 1.32)],
                ],
                width_mm=3.0, max_width_mm=9.5,
            ),
        ),
        # 2 -----------------------------------------------------------------
        DefectSpec(
            id="D-02", cls="corrosion",
            theta0=np.radians(-48.0), theta1=np.radians(-20.0),
            y0=0.34, y1=1.16,
            severity="High", size_mm=260.0,
            note="Exposed, section-loss rebar with active rust bleeding.",
            extra=dict(
                exposed_bar=True, bar_theta_deg=-34.0,
                stain_run_m=0.95, loss_pct=18.0,
            ),
        ),
        # 3 -----------------------------------------------------------------
        DefectSpec(
            id="D-03", cls="spalling",
            theta0=np.radians(49.0), theta1=np.radians(77.0),
            y0=1.66, y1=2.38,
            severity="High", size_mm=310.0,
            note="Cover loss, 45-60 mm deep void, aggregates exposed.",
            extra=dict(depth_mm=52.0),
        ),
        # 4 -----------------------------------------------------------------
        DefectSpec(
            id="D-04", cls="efflorescence",
            theta0=np.radians(-124.0), theta1=np.radians(-96.0),
            y0=0.14, y1=0.68,
            severity="Moderate", size_mm=290.0,
            note="Calcium carbonate bloom with vertical seepage drips.",
            extra=dict(drip_run_m=0.55),
        ),
    ]


DEFECTS = _default_defects()
DEFECTS_BY_ID = {d.id: d for d in DEFECTS}


# --------------------------------------------------------------------------
# Geometry helpers shared by every stage
# --------------------------------------------------------------------------

def theta_y_to_xyz(theta: float, y: float, radius: float = COLUMN_RADIUS_M) -> np.ndarray:
    """Pier surface point at azimuth THETA (rad) and elevation Y (m)."""
    return np.array([radius * np.sin(theta), y, radius * np.cos(theta)], dtype=float)


def xyz_to_theta_y(p: np.ndarray):
    """Inverse of :func:`theta_y_to_xyz` -> (azimuth rad, elevation, radius)."""
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.sqrt(x * x + z * z)
    return np.arctan2(x, z), y, r


def arc_length_to_theta(s_m: np.ndarray) -> np.ndarray:
    """Surface arc length (m) -> azimuth (rad)."""
    return np.asarray(s_m, dtype=float) / COLUMN_RADIUS_M


# --------------------------------------------------------------------------
# Camera rig
# --------------------------------------------------------------------------

def intrinsics(width: int = IMAGE_W, height: int = IMAGE_H,
               hfov_deg: float = HORIZONTAL_FOV_DEG) -> np.ndarray:
    """OV2640 pinhole intrinsics K (square pixels, principal point centred)."""
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = fx
    return np.array([[fx, 0.0, (width - 1) / 2.0],
                     [0.0, fy, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def look_at_pose(eye, target, up=(0.0, 1.0, 0.0)):
    """
    Build a world->camera pose (R, t) with the OpenCV/COLMAP convention
    (x right, y down, z forward along the optical axis).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(up, dtype=float)

    f = target - eye
    f = f / np.linalg.norm(f)                  # camera +Z (forward)
    r = np.cross(f, world_up)
    if np.linalg.norm(r) < 1e-9:               # degenerate: pick any up
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
    r = r / np.linalg.norm(r)                  # camera +X (right)
    u = np.cross(f, r)                         # camera +Y (down): r x u = f
    u = u / np.linalg.norm(u)

    R = np.stack([r, u, f], axis=0)            # rows: x_cam, y_cam, z_cam axes
    t = -R @ eye
    return R, t


@dataclass
class CameraView:
    """A single orbital ESP32-CAM node of the rig."""
    index: int
    label: str
    compass: str
    azimuth_deg: float
    position: np.ndarray
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    width: int = IMAGE_W
    height: int = IMAGE_H
    device_id: str = ""
    image_name: str = ""
    rssi_dbm: int = -55

    # -- convenience ------------------------------------------------------
    @property
    def center(self) -> np.ndarray:
        return (-self.R.T @ self.t).reshape(3)

    @property
    def forward(self) -> np.ndarray:
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    @property
    def down(self) -> np.ndarray:
        return self.R.T @ np.array([0.0, 1.0, 0.0])

    @property
    def right(self) -> np.ndarray:
        return self.R.T @ np.array([0.0, 1.0, 0.0])

    def project(self, points):
        """World points (N,3) -> pixel coords (N,2) and camera depths (N,)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        cam = pts @ self.R.T + self.t.reshape(1, 3)
        z = cam[:, 2]
        zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.K[0, 0] * cam[:, 0] / zz + self.K[0, 2]
        v = self.K[1, 1] * cam[:, 1] / zz + self.K[1, 2]
        return np.stack([u, v], axis=1), z

    def quaternion_wxyz(self):
        """COLMAP-style quaternion (world->camera rotation)."""
        return rotation_matrix_to_quaternion(self.R)

    def to_dict(self) -> dict:
        c = self.center
        return {
            "index": self.index,
            "label": self.label,
            "compass": self.compass,
            "device_id": self.device_id,
            "azimuth_deg": round(self.azimuth_deg, 3),
            "position": [round(float(v), 4) for v in c],
            "forward": [round(float(v), 4) for v in self.forward],
            "rssi_dbm": self.rssi_dbm,
            "image": self.image_name,
            "resolution": [self.width, self.height],
            "K": [[round(float(x), 4) for x in row] for row in self.K],
            "R": [[round(float(x), 6) for x in row] for row in self.R],
            "t": [round(float(x), 6) for x in self.t],
        }


def rotation_matrix_to_quaternion(R: np.ndarray):
    """3x3 rotation -> (qw, qx, qy, qz)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def camera_rig(include_empty: bool = False):
    """Build the 8-node orbital rig (deterministic, nominal survey poses)."""
    R = intrinsics()
    views = []
    for i in range(N_VIEWS):
        az = np.radians(i * AZIMUTH_STEP_DEG)
        # node spacing: alternate a small radius jitter so the baselines are
        # not perfectly regular (reconstruction must actually solve for them)
        rad = ORBIT_RADIUS_M + (0.06 if i % 3 == 0 else -0.03 if i % 3 == 1 else 0.0)
        eye = np.array([rad * np.sin(az), ORBIT_HEIGHT_M + 0.04 * ((i % 2) - 0.5),
                        rad * np.cos(az)])
        # each node aims at the pier, with a small aiming offset
        target = LOOK_AT + np.array([0.02 * np.cos(2 * i), 0.0, 0.02 * np.sin(2 * i)])
        Rm, t = look_at_pose(eye, target)
        views.append(CameraView(
            index=i, label=VIEW_LABELS[i], compass=COMPASS[i],
            azimuth_deg=i * AZIMUTH_STEP_DEG, position=eye,
            R=Rm, t=t, K=R, width=IMAGE_W, height=IMAGE_H,
            device_id=f"ESP32CAM-{i + 1:02d}",
            image_name=f"esp32_cam_{i + 1:02d}_view_{VIEW_LABELS[i]}.jpg",
            rssi_dbm=-41 - 3 * (i % 4),
        ))
    return views


def scene_summary() -> dict:
    return {
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "structure": {
            "label": COLUMN_LABEL,
            "description": COLUMN_DESCRIPTION,
            "concrete_grade": CONCRETE_GRADE,
            "diameter_m": 2 * COLUMN_RADIUS_M,
            "height_m": COLUMN_HEIGHT_M,
            "nominal_cover_mm": NOMINAL_COVER_MM,
        },
        "sensor": {
            "name": SENSOR_NAME,
            "resolution": [IMAGE_W, IMAGE_H],
            "hfov_deg": HORIZONTAL_FOV_DEG,
        },
        "rig": {
            "n_views": N_VIEWS,
            "orbit_radius_m": ORBIT_RADIUS_M,
            "orbit_height_m": ORBIT_HEIGHT_M,
            "azimuth_step_deg": AZIMUTH_STEP_DEG,
        },
        "defects": [d.to_dict() for d in DEFECTS],
        "data_dir": str(DATA_DIR),
    }
