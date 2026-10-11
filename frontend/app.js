/* ==========================================================================
   AI 3D Structural Inspection System -- WebGL digital twin front end
   --------------------------------------------------------------------------
   Renders, in a single Three.js scene:
     * the dense reconstruction point cloud (digital_twin.ply, class-coloured)
     * the textured surface mesh of the pier (digital_twin.obj + .mtl texture)
     * the internal reinforcement cage (8 x D20 verticals + D10 helix)
     * the eight ESP32-CAM 3D frustums at their reconstructed poses
     * the camera-to-defect triangulation rays from step 4
     * pulsing 3D defect markers coloured by defect class
   and drives the four-step pipeline from the side panel.
   ========================================================================== */
'use strict';

const CLASSES = ['crack', 'corrosion', 'spalling', 'efflorescence'];
const CLASS_COLOR = {crack: 0xff2d2d, corrosion: 0xff8c00, spalling: 0xffd400,
                     efflorescence: 0x4fc3f7};
const CLASS_LABEL = {crack: 'Diagonal shear crack', corrosion: 'Rebar corrosion',
                     spalling: 'Concrete spalling', efflorescence: 'Efflorescence'};
const clamp01 = v => Math.max(0, Math.min(1, v));
const $ = id => document.getElementById(id);

/* ========================================================================
   1. THREE.JS SCENE
   ======================================================================== */
const App = {
  scene: null, camera: null, renderer: null, controls: null,
  groups: {},      // cloud, mesh, rebar, cams, rays, markers, grid, axes
  markers: [],
  clock: null,
  state: null,
  spin: false,
  camCycle: 0,
};

function initThree() {
  const canvas = $('gl');
  App.scene = new THREE.Scene();
  App.scene.background = new THREE.Color(0x05080c);
  App.scene.fog = new THREE.Fog(0x05080c, 12, 30);

  App.camera = new THREE.PerspectiveCamera(
    46, canvas.clientWidth / canvas.clientHeight, 0.02, 200);
  App.camera.position.set(3.6, 2.4, 4.2);

  App.renderer = new THREE.WebGLRenderer({canvas, antialias: true,
                                          preserveDrawingBuffer: true});
  App.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  App.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  App.controls = new THREE.OrbitControls(App.camera, App.renderer.domElement);
  App.controls.enableDamping = true;
  App.controls.dampingFactor = 0.075;
  App.controls.target.set(0, 1.45, 0);
  App.controls.minDistance = 0.6;
  App.controls.maxDistance = 26;
  App.controls.maxPolarAngle = Math.PI * 0.92;

  // ---- lighting: key + sky fill + ground bounce + rim ------------------
  App.scene.add(new THREE.HemisphereLight(0xbcd4ff, 0x2a2f36, 0.85));
  const key = new THREE.DirectionalLight(0xfff3e0, 1.15);
  key.position.set(5.5, 8.5, 4.2);
  App.scene.add(key);
  const rim = new THREE.DirectionalLight(0x74a8ff, 0.45);
  rim.position.set(-6, 3.2, -5);
  App.scene.add(rim);
  App.scene.add(new THREE.AmbientLight(0x404a58, 0.55));

  // ---- groups ----------------------------------------------------------
  for (const g of ['grid', 'axes', 'pier', 'cloud', 'mesh', 'rebar', 'cams',
                   'rays', 'markers', 'ground'])
    App.groups[g] = new THREE.Group(), App.scene.add(App.groups[g]);
  // everything that belongs to the inspected element is parented to the
  // 'pier' group so that the spin control rotates the model as one object
  // while the ground, the axes and the camera rig stay put
  for (const g of ['cloud', 'mesh', 'rebar', 'markers', 'rays'])
    App.groups.pier.add(App.groups[g]);

  // ---- ground grid + axes ---------------------------------------------
  const grid = new THREE.GridHelper(14, 28, 0x2a3644, 0x18212c);
  grid.position.y = 0.001;
  App.groups.grid.add(grid);
  const axes = new THREE.AxesHelper(0.9);
  axes.position.set(0, 0.002, 0);
  App.groups.axes.add(axes);

  // subtle reflective floor for depth cueing
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(9, 64),
    new THREE.MeshStandardMaterial({color: 0x11161d, roughness: 0.95,
                                    metalness: 0.05}));
  floor.rotation.x = -Math.PI / 2;
  App.groups.ground.add(floor);

  App.clock = new THREE.Clock();
  addEventListener('resize', onResize);
  animate();
}

function onResize() {
  const c = $('gl');
  if (!c.clientWidth) return;
  App.camera.aspect = c.clientWidth / c.clientHeight;
  App.camera.updateProjectionMatrix();
  App.renderer.setSize(c.clientWidth, c.clientHeight, false);
}

function animate() {
  requestAnimationFrame(animate);
  const t = App.clock.getElapsedTime();
  App.controls.update();

  if (App.spin) App.groups.pier.rotation.y += 0.0035;   // whole model assembly

  // pulsing defect markers: a tapered spike + a glowing orbit ring
  if ($('t-pulse') && $('t-pulse').checked) {
    for (const m of App.markers) {
      const s = 1 + 0.28 * Math.sin(t * 2.6 + m.userData.phase);
      m.scale.setScalar(s);
      if (m.userData.ring) {
        m.userData.ring.scale.setScalar(1 + 0.55 * ((t * 0.55 + m.userData.phase) % 1));
        m.userData.ring.material.opacity =
          0.75 * (1 - ((t * 0.55 + m.userData.phase) % 1));
      }
    }
  }
  App.renderer.render(App.scene, App.camera);
}

/* ========================================================================
   2. STATE POLLING + PANEL RENDERING
   ======================================================================== */
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r.text();
}

function toast(msg, ms = 3200) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add('hidden'), ms);
}

async function refresh() {
  try {
    App.state = await api('/api/status');
  } catch (e) {
    toast('Status request failed: ' + e.message);
    return;
  }
  renderHeader();
  renderFleet();
  renderShots();
  renderRecon();
  renderAnalysis();
  renderTwin();
  renderFiles();
}

function renderHeader() {
  const s = App.state, sc = s.scene;
  $('hdr-structure').textContent = sc.structure.label;
  $('hdr-sensor').textContent = sc.sensor.name;
  $('hdr-mode').textContent = (s.reconstruction && s.reconstruction.method) ||
                              'hybrid COLMAP + OpenCV SfM';
  $('foot-build').textContent = `${sc.project} v${sc.version} · data ${sc.data_dir}`;

  const rep = s.report_meta;
  if (rep) {
    $('hdr-shi').textContent = rep.shi.shi.toFixed(1);
    $('hdr-grade').textContent = `grade ${rep.shi.shi_grade}`;
    $('hdr-shi').style.color = rep.shi.shi >= 78 ? '#5ee08d'
      : rep.shi.shi >= 62 ? '#ffce4d' : '#ff6b6b';
  } else {
    $('hdr-shi').textContent = '—';
    $('hdr-grade').textContent = 'not run';
  }
  const rstats = s.reconstruction && s.reconstruction.stats;
  if (rstats) {
    $('hdr-model').textContent = rstats.registered_images + '/' + sc.rig.n_views;
    $('hdr-model-sub').textContent =
      `${fmtInt(rstats.n_points3d)} sparse · ${fmtInt(rstats.dense_points)} dense`;
  } else {
    $('hdr-model').textContent = '—';
    $('hdr-model-sub').textContent = 'no reconstruction';
  }
  const a = s.analysis && s.analysis.step3;
  if (a) {
    $('hdr-defects').textContent = a.total_detections;
    $('hdr-defects-sub').textContent = CLASSES
      .filter(c => (a.per_view_counts[c] || 0) > 0)
      .map(c => `${c} ${a.per_view_counts[c]}`).join(' · ') || 'none';
  } else {
    $('hdr-defects').textContent = '—';
    $('hdr-defects-sub').textContent = 'no detections';
  }
  $('hdr-nodes').textContent = `${s.fleet.online}/${s.fleet.n_nodes}`;
  $('hdr-nodes-sub').textContent = s.job.running ? 'pipeline running' :
    (s.fleet.commands.pending ? `${s.fleet.commands.pending} queued` : 'idle');
  $('hud-cam').textContent = s.captures.filter(c => c.exists).length + ' views captured';
}

function renderFleet() {
  const el = $('fleet');
  el.innerHTML = App.state.fleet.devices.map(d => `
    <div class="dev">
      <div class="id">${d.device_id}
        <span class="badge ${d.status === 'IDLE' ? '' : 'busy'}">${d.status}</span>
      </div>
      <div class="meta">${d.view} · ${d.ip} · ${d.rssi_dbm} dBm</div>
      <div class="meta">GPIO${d.flash_led_gpio} flash · GPIO${d.countdown_led_gpio} led
        · ${d.uploads} up</div>
      <div class="meta">${d.battery_v.toFixed(2)} V · fw ${d.firmware}</div>
    </div>`).join('');
}

function renderShots() {
  const el = $('shots');
  el.innerHTML = App.state.captures.map(c => `
    <div class="shot ${c.exists ? '' : 'empty'}" title="${c.device} · azimuth ${c.azimuth_deg}°">
      ${c.exists ? `<img loading="lazy" src="${c.annotated_url}?t=${Date.now()}"
                    onerror="this.src='/captures/${c.filename}?t=${Date.now()}'"/>` : ''}
      <div class="cap">${c.view} · ${c.compass} ${c.azimuth_deg}°</div>
    </div>`).join('');
}

function fmtInt(n) { return n == null ? '—' : n.toLocaleString(); }

function kv(target, rows) {
  $(target).innerHTML = rows.map(([k, v, wide]) =>
    `<div class="cell ${wide ? 'wide' : ''}"><span class="k">${k}</span>
     <span class="v">${v}</span></div>`).join('');
}

function renderRecon() {
  const r = App.state.reconstruction;
  if (!r) {
    kv('recon-kv', [['status', 'not reconstructed'],
                    ['action', 'click <b>Reconstruct</b>']]);
    $('pair-matrix').innerHTML = '';
    return;
  }
  const s = r.stats;
  kv('recon-kv', [
    ['method', r.method],
    ['registered views', `${s.registered_images}/${App.state.scene.rig.n_views}`],
    ['sparse points', fmtInt(s.n_points3d)],
    ['dense samples', fmtInt(s.dense_points)],
    ['verified pairs', `${s.pairs_with_geometry}/${s.n_pairs}`],
    ['RANSAC inliers', fmtInt(s.total_inliers)],
    ['reproj. error', `${s.mean_reprojection_error_px} px`],
    ['mean track', `${s.mean_track_length} obs/pt`],
    ['fitted radius', s.fitted_cylinder_radius_m
      ? `${(s.fitted_cylinder_radius_m * 1000).toFixed(1)} mm` : '—'],
    ['nominal radius', `${(App.state.scene.structure.diameter_m * 500).toFixed(1)} mm`],
    ['colmap binary', s.colmap_binary || 'not installed (OpenCV hybrid)'],
    ['total time', `${r.timing.total_s} s`],
  ]);
  renderMatrix(s.matrix);
}

function renderMatrix(m) {
  if (!m) { $('pair-matrix').innerHTML = ''; return; }
  const t = ['<tr><th></th>' + m.map((_, j) => `<th>V${j + 1}</th>`).join('') + '</tr>'];
  const max = Math.max(1, ...m.flat());
  m.forEach((row, i) => {
    let tr = `<tr><th>V${i + 1}</th>`;
    row.forEach((v, j) => {
      const cls = i === j ? '' : v === 0 ? '' : (v > max * 0.6 ? 'hot' : 'warm');
      tr += `<td class="${cls}" title="V${i + 1}↔V${j + 1}: ${v} RANSAC inliers">`
          + (i === j ? '·' : v) + '</td>';
    });
    t.push(tr + '</tr>');
  });
  $('pair-matrix').innerHTML = t.join('');
}

function renderAnalysis() {
  const rep = App.state.report_meta;
  const an = App.state.analysis;
  if (!rep) {
    $('shi-value').textContent = '—';
    $('shi-grade').textContent = 'not run';
    $('shi-action').textContent = 'Run the analysis to compute the index.';
    $('shi-bar-fill').style.width = '0%';
    kv('analysis-kv', [['status', 'no analysis yet'],
                       ['action', 'click <b>Run analysis</b>']]);
    $('findings').innerHTML = '';
    kv('quality-kv', []);
    return;
  }
  const shi = rep.shi;
  $('shi-value').textContent = shi.shi.toFixed(1);
  $('shi-grade').textContent = `Grade ${shi.shi_grade}`;
  $('shi-action').textContent = shi.recommended_action;
  $('shi-bar-fill').style.width = shi.shi + '%';
  const deg = 360 * (shi.shi / 100);
  const col = shi.shi >= 78 ? '#2fd07a' : shi.shi >= 62 ? '#ffb300' : '#ff4646';
  document.querySelector('.shi-dial').style.background =
    `conic-gradient(${col} ${deg}deg, #1a2431 ${deg}deg)`;

  const st = an.step3, s4 = an.step4;
  kv('analysis-kv', [
    ['detections', st.total_detections],
    ['detector time', `${st.duration_s} s`],
    ['rays hit', `${s4.rays_hit}/${s4.rays_cast}`],
    ['3D clusters', s4.n_clusters],
    ['defect points', fmtInt(s4.n_defect_points)],
    ['multi-view err', `${(s4.mean_multiview_consistency_m * 1000).toFixed(2)} mm`],
    ['class penalty', shi.total_penalty.toFixed(1)],
    ['worst class', worstClass(shi), true],
  ]);

  $('findings').innerHTML = (rep.findings || []).map(f => {
    const m = App.state.scene.defects.find(d => d.class === f.class) || {};
    return `<div class="finding">
      <div class="head"><span class="swatch" style="background:${colHex(f.class)}"></span>
        ${f.name}<span class="sev ${f.worst_severity}">${f.worst_severity}</span></div>
      <div class="body">${f.count} instance(s) · affected surface
        <b>${(f.total_area_m2 * 1e4).toFixed(0)} cm²</b>
        ${f.max_width_mm ? `· max opening <b>${f.max_width_mm.toFixed(2)} mm</b>` : ''}
        · per-view detections ${f.evidence.per_view_detections}
        (mean confidence ${(f.evidence.mean_confidence * 100).toFixed(0)}%)</div>
      <div class="rem">${f.remediation}</div>
    </div>`;
  }).join('') || '<p class="muted">No defects above the reporting threshold.</p>';

  const q = (an.step3.quality && an.step3.quality.per_class) || null;
  kv('quality-kv', q ? CLASSES.map(c => {
    const v = q[c]; if (!v) return null;
    return [c, `P ${(v.precision * 100).toFixed(0)}% · R ${(v.recall * 100).toFixed(0)}% ·
      F1 ${v.f1.toFixed(2)}`, true];
  }).filter(Boolean) : [['quality', 'not scored']]);
}

function worstClass(shi) {
  let w = '—', p = -1;
  for (const c of CLASSES) {
    const v = shi.per_class[c];
    if (v && v.penalty > p) { p = v.penalty; w = `${c} (${v.worst})`; }
  }
  return w;
}
function colHex(cls) {
  return '#' + CLASS_COLOR[cls].toString(16).padStart(6, '0');
}

function renderTwin() {
  const rc = App.state.raycast;
  const s = rc && rc.stats;
  if (!rc) {
    kv('twin-kv', [['status', 'no back-projection yet'],
                   ['action', 'run the analysis']]);
    $('cluster-list').innerHTML = '';
    return;
  }
  kv('twin-kv', [
    ['clusters', rc.n_clusters],
    ['defect points', fmtInt(rc.n_defect_points)],
    ['mean ray error', `${(s.mean_ray_error_m * 1000).toFixed(2)} mm`],
    ['raycast time', `${rc.duration_s} s`],
    ['surface radius', `${(s.surface_model.radius_m * 1000).toFixed(1)} mm`],
    ['axis (x,z)', `${s.surface_model.center_xz.map(v => v.toFixed(3)).join(', ')}`],
    ['twin stamped', s.twin_stamp ? fmtInt(s.twin_stamp.stamped_points) : '—'],
  ]);
  $('cluster-list').innerHTML = (rc.global_defects || []).map(c => `
    <div class="cl" data-id="${c.id}" data-xyz="${c.world_xyz.join(',')}">
      <div class="top"><span class="swatch" style="background:${c.color};
        width:10px;height:10px;border-radius:50%"></span>${c.id} · ${c.name}
        <span class="sev ${c.severity}">${c.severity}</span></div>
      <div class="pos">${c.views.join(' ')} · θ ${c.surface_theta_deg.toFixed(1)}° ·
        y ${c.surface_elevation_m.toFixed(2)} m ·
        XYZ (${c.world_xyz.map(v => v.toFixed(3)).join(', ')}) ·
        ±${(c.uncertainty_m * 1000).toFixed(1)} mm · conf
        ${(c.confidence * 100).toFixed(0)}%</div>
    </div>`).join('') || '<p class="muted">No clusters resolved.</p>';

  document.querySelectorAll('.cl').forEach(el => el.addEventListener('click', () => {
    const [x, y, z] = el.dataset.xyz.split(',').map(Number);
    flyTo(new THREE.Vector3(x, y, z));
  }));
}

function renderFiles() {
  $('file-strip').innerHTML = App.state.files.map(f =>
    `<span class="file-chip ${f.exists ? 'ok' : ''}" title="${f.name} ·
      ${f.bytes} bytes">${f.name}</span>`).join('');
}

function flyTo(target) {
  const start = App.controls.target.clone();
  const dir = App.camera.position.clone().sub(start).normalize().multiplyScalar(1.9);
  const end = target.clone();
  const camEnd = end.clone().add(dir);
  const t0 = performance.now();
  const step = () => {
    const k = clamp01((performance.now() - t0) / 750);
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    App.controls.target.lerpVectors(start, end, e);
    App.camera.position.lerpVectors(start.clone().add(
      App.camera.position.clone().sub(start)), camEnd, e);
    if (k < 1) requestAnimationFrame(step);
  };
  step();
}

/* ========================================================================
   3. 3D CONTENT BUILDERS
   ======================================================================== */
const pierGroupOf = name => App.groups[name];

function clearGroup(name) {
  const g = App.groups[name];
  if (!g) return;
  for (const c of [...g.children]) {
    g.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) (Array.isArray(c.material) ? c.material : [c.material])
      .forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
  }
}

async function loadPointCloud(url, colourByClass) {
  clearGroup('cloud');
  try {
    const geo = await new Promise((res, rej) =>
      new THREE.PLYLoader().load(url + '?t=' + Date.now(), res, undefined, rej));
    geo.computeBoundingBox();
    const hasClass = geo.hasAttribute('class') && colourByClass;
    let colors = geo.getAttribute('color');
    if (hasClass) {
      const cls = geo.getAttribute('class');
      const arr = new Float32Array(cls.count * 3);
      for (let i = 0; i < cls.count; i++) {
        const c = Math.round(cls.getX(i));
        const col = c >= 1 && c <= 4
          ? new THREE.Color(CLASS_COLOR[CLASSES[c - 1]])
          : new THREE.Color(colors.getX(i), colors.getY(i), colors.getZ(i))
              .lerp(new THREE.Color(0x9aa7b6), 0.35);
        arr[i * 3] = col.r; arr[i * 3 + 1] = col.g; arr[i * 3 + 2] = col.b;
      }
      geo.setAttribute('color', new THREE.BufferAttribute(arr, 3));
    }
    const size = 0.028;
    const mat = new THREE.PointsMaterial({
      size, vertexColors: true, sizeAttenuation: true,
      transparent: true, opacity: 0.92, depthWrite: false,
    });
    const pts = new THREE.Points(geo, mat);
    App.groups.cloud.add(pts);
    $('hud-points').textContent = geo.attributes.position.count.toLocaleString();
    return geo.attributes.position.count;
  } catch (e) {
    $('hud-points').textContent = 'unavailable';
    return 0;
  }
}

async function loadSurfaceMesh(url, fallbackRadius) {
  clearGroup('mesh');
  // ---- 1. try the reconstructed OBJ (with its MTL texture) -------------
  try {
    const obj = await new Promise((res, rej) =>
      new THREE.OBJLoader().load(url + '?t=' + Date.now(), res, undefined, rej));
    let textured = false;
    obj.traverse(child => {
      if (!child.isMesh) return;
      child.material = new THREE.MeshStandardMaterial({
        color: 0x8d949c, roughness: 0.94, metalness: 0.04,
        flatShading: false, side: THREE.DoubleSide});
      textured = true;
    });
    if (textured) {
      App.groups.mesh.add(obj);
      // the OBJ carries the rebar cage as line segments: move them out
      return true;
    }
  } catch (e) { /* fall through to the analytic surface */ }

  // ---- 2. analytic fallback: the fitted/nominal cylinder ----------------
  const R = fallbackRadius || 0.30, H = 3.0;
  const g = new THREE.CylinderGeometry(R, R, H, 96, 1, true);
  const m = new THREE.MeshStandardMaterial({
    color: 0x8f959d, roughness: 0.95, metalness: 0.03,
    side: THREE.DoubleSide});
  const mesh = new THREE.Mesh(g, m);
  mesh.position.y = H / 2;
  const tex = new THREE.TextureLoader().load(
    '/model/digital_twin_texture.jpg?t=' + Date.now(),
    t => { t.wrapS = THREE.RepeatWrapping; mesh.material.map = t;
           mesh.material.needsUpdate = true; },
    undefined, () => {});
  App.groups.mesh.add(mesh);
  return true;
}

function buildRebarCage(radius) {
  clearGroup('rebar');
  const R = (radius || 0.30) - 0.05, H = 3.0;
  const mat = new THREE.MeshStandardMaterial({color: 0xb0743a, roughness: 0.42,
                                              metalness: 0.75});
  const barR = 0.011;
  // 8 vertical D20 bars
  for (let i = 0; i < 8; i++) {
    const a = i / 8 * Math.PI * 2;
    const bar = new THREE.Mesh(
      new THREE.CylinderGeometry(barR, barR, H - 0.06, 10), mat);
    bar.position.set(Math.sin(a) * R, H / 2, Math.cos(a) * R);
    App.groups.rebar.add(bar);
  }
  // D10@150 helical confinement in 20 turns
  const curve = new THREE.CatmullRomCurve3(
    Array.from({length: 400}, (_, k) => {
      const f = k / 399, a = f * Math.PI * 2 * 20;
      return new THREE.Vector3(Math.sin(a) * R, 0.05 + f * (H - 0.1), Math.cos(a) * R);
    }));
  App.groups.rebar.add(new THREE.Mesh(
    new THREE.TubeGeometry(curve, 900, 0.0055, 7, false), mat));
  App.groups.rebar.visible = $('t-rebar').checked;
}

function buildCameras(cameras, viewRadius) {
  clearGroup('cams');
  const R = viewRadius || 0.30, H = 3.0;
  cameras.forEach((cam, i) => {
    const pos = new THREE.Vector3(...(cam.center || cam.position));
    const grp = new THREE.Group();
    grp.position.copy(pos);

    // node body
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.09, 0.07, 0.032),
      new THREE.MeshStandardMaterial({color: 0x1b2836, roughness: 0.6,
                                      metalness: 0.35}));
    grp.add(body);

    // LED indicators: GPIO4 flash + GPIO33 countdown
    const led = (col, x) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.0075, 10, 8),
        new THREE.MeshBasicMaterial({color: col}));
      m.position.set(x, 0.026, 0.019);
      grp.add(m);
      return m;
    };
    const flash = led(0xffffff, -0.018);
    const count = led(0xff3b30, 0.018);

    // optical frustum wireframe, oriented at the pier
    const aim = new THREE.Vector3(0, H * 0.48, 0);
    const depth = Math.max(0.35, pos.distanceTo(aim));
    const aspect = 1600 / 1200;
    const fov = 62 * Math.PI / 180;
    const hh = Math.tan(fov / 2) * depth, hw = hh * aspect;
    const apex = new THREE.Vector3(0, 0, 0);
    const corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]
      .map(([x, y]) => new THREE.Vector3(x, y, depth));
    const fg = new THREE.BufferGeometry();
    const verts = [];
    corners.forEach(c => {
      verts.push(apex.x, apex.y, apex.z, c.x, c.y, c.z);
      const nx = corners[(corners.indexOf(c) + 1) % 4];
      verts.push(c.x, c.y, c.z, nx.x, nx.y, nx.z);
    });
    fg.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const lines = new THREE.LineSegments(fg,
      new THREE.LineBasicMaterial({color: 0x7cf7c0, transparent: true, opacity: 0.55}));
    const imPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(hw * 2, hh * 2),
      new THREE.MeshBasicMaterial({color: 0x7cf7c0, transparent: true,
                                   opacity: 0.045, side: THREE.DoubleSide}));
    imPlane.position.z = depth;
    const frustum = new THREE.Group();
    frustum.add(lines, imPlane);

    // orient +Z of the frustum towards the pier
    const dir = aim.clone().sub(pos).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
    frustum.quaternion.copy(q);
    grp.add(frustum);

    // caption sprite
    const sprite = makeLabel(`V${i + 1} ${cam.view || ''}`, '#7cf7c0');
    sprite.position.set(0, 0.075, 0);
    grp.add(sprite);

    grp.userData = {flash, count, phase: i * 0.4, view: cam.view || `V${i + 1}`};
    App.groups.cams.add(grp);
  });
  App.groups.cams.visible = $('t-cams').checked;
}

function makeLabel(text, color) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(6,10,15,0.82)';
  g.roundRect ? (g.beginPath(), g.roundRect(2, 12, 252, 40, 10), g.fill())
              : g.fillRect(2, 12, 252, 40);
  g.font = 'bold 30px system-ui, sans-serif';
  g.fillStyle = color; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, 128, 33);
  const tex = new THREE.CanvasTexture(c);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, transparent: true,
                                                       depthWrite: false}));
  s.scale.set(0.20, 0.05, 1);
  return s;
}

function buildRays(clusters) {
  clearGroup('rays');
  clusters.forEach(c => {
    (c.rays || []).forEach(r => {
      const a = new THREE.Vector3(...r.origin);
      const b = new THREE.Vector3(...c.world_xyz);
      const g = new THREE.BufferGeometry().setFromPoints([a, b]);
      const line = new THREE.Line(g, new THREE.LineBasicMaterial({
        color: CLASS_COLOR[c.class] || 0xb388ff, transparent: true, opacity: 0.34}));
      App.groups.rays.add(line);
      // arrow head at the hit point
      const dir = b.clone().sub(a).normalize();
      const head = new THREE.Mesh(
        new THREE.ConeGeometry(0.012, 0.038, 8),
        new THREE.MeshBasicMaterial({color: CLASS_COLOR[c.class] || 0xb388ff,
                                     transparent: true, opacity: 0.6}));
      head.position.copy(b).addScaledVector(dir, -0.02);
      head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      App.groups.rays.add(head);
    });
  });
  App.groups.rays.visible = $('t-rays').checked;
}

function buildMarkers(clusters) {
  clearGroup('markers');
  App.markers = [];
  clusters.forEach((c, i) => {
    const col = CLASS_COLOR[c.class] || 0xffffff;
    const p = new THREE.Vector3(...c.world_xyz);

    // spike pointing radially out of the pier surface
    const dir = new THREE.Vector3(p.x, 0, p.z).normalize();
    const grp = new THREE.Group();
    grp.position.copy(p);

    const spike = new THREE.Mesh(
      new THREE.ConeGeometry(0.022, 0.10, 14),
      new THREE.MeshBasicMaterial({color: col}));
    spike.position.copy(dir.clone().multiplyScalar(0.055));
    spike.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    grp.add(spike);

    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.020, 16, 12),
      new THREE.MeshBasicMaterial({color: col}));
    grp.add(core);

    // pulsing ring on the surface plane
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.030, 0.041, 26),
      new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: 0.7,
                                   side: THREE.DoubleSide}));
    ring.position.copy(dir.clone().multiplyScalar(0.006));
    ring.lookAt(p.clone().add(dir));
    grp.add(ring);

    grp.userData = {phase: i * 0.7, ring, cluster: c};
    App.groups.markers.add(grp);
    App.markers.push(grp);
  });
}

/* ========================================================================
   4. ACTIONS
   ======================================================================== */
async function runPipeline() {
  const modal = $('modal');
  modal.classList.remove('hidden');
  $('btn-modal-close').disabled = true;
  $('modal-log').textContent = 'starting…';
  try {
    await api('/api/pipeline/run?supersample=1', {method: 'POST'});
  } catch (e) {
    $('modal-log').textContent = 'failed to start: ' + e.message;
    return;
  }
  const poll = setInterval(async () => {
    let st;
    try { st = await api('/api/pipeline/status'); } catch (e) { return; }
    $('modal-log').textContent = st.progress.join('\n') || 'starting…';
    $('modal-log').scrollTop = $('modal-log').scrollHeight;
    if (!st.running) {
      clearInterval(poll);
      $('btn-modal-close').disabled = false;
      if (st.error) {
        $('modal-log').textContent += '\n\nERROR: ' + st.error;
      } else {
        $('modal-log').textContent += `\n\ncompleted in ${st.duration_s}s`;
      }
      await refresh();
      await loadScene();
    }
  }, 1200);
}

async function loadScene() {
  const st = App.state;
  const scene = st.scene;
  const rad = (st.reconstruction && st.reconstruction.stats &&
               st.reconstruction.stats.fitted_cylinder_radius_m) ||
              scene.structure.diameter_m / 2;

  // one fetch of the twin cloud per rebuild: the classifier stamp travels in
  // the PLY itself as a per-vertex 'class' property
  const n = await loadPointCloud('/model/digital_twin.ply', $('t-color').checked);
  if (n) $('hud-points').textContent = n.toLocaleString();
  await loadSurfaceMesh('/model/digital_twin.obj', rad);
  buildRebarCage(rad);

  const cams = (st.reconstruction && st.reconstruction.cameras) ||
               (st.raycast && st.raycast.cameras) || [];
  if (cams.length) buildCameras(cams, rad);

  const clusters = (st.raycast && st.raycast.global_defects) || [];
  buildRays(clusters);
  buildMarkers(clusters);
  $('hud-rays').textContent = clusters.length
    ? `${clusters.length} clusters` : 'no clusters';
}

async function runStep(button, path, label) {
  const b = $(button);
  const old = b.textContent;
  b.disabled = true;
  b.textContent = 'working…';
  $('hud-stage').textContent = label;
  try {
    await api(path, {method: 'POST'});
    toast(`${label} complete`);
  } catch (e) {
    toast(`${label} failed: ${e.message}`, 6000);
  }
  b.disabled = false;
  b.textContent = old;
  $('hud-stage').textContent = 'idle';
  await refresh();
  await loadScene();
}

/* ========================================================================
   5. WIRING
   ======================================================================== */
function wire() {
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tabpane').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('pane-' + t.dataset.tab).classList.add('active');
    onResize();
  }));

  $('btn-run').addEventListener('click', runPipeline);
  $('btn-sweep').addEventListener('click', () =>
    runStep('btn-sweep', '/api/esp32/capture', 'capture sweep'));
  $('btn-recon').addEventListener('click', () =>
    runStep('btn-recon', '/api/reconstruct', 'reconstruction'));
  $('btn-analyse').addEventListener('click', () =>
    runStep('btn-analyse', '/api/analyse', 'defect analysis'));
  $('btn-refresh').addEventListener('click', async () => { await refresh();
                                                          await loadScene(); });
  $('btn-reset').addEventListener('click', async () => {
    if (!confirm('Delete every generated artefact (captures, model, analysis, twin)?'))
      return;
    await api('/api/reset?what=all', {method: 'POST'});
    clearGroup('cloud'); clearGroup('mesh'); clearGroup('rebar');
    clearGroup('cams'); clearGroup('rays'); clearGroup('markers');
    App.markers = [];
    toast('artefacts cleared');
    await refresh();
  });

  // ---- uploads ---------------------------------------------------------
  $('btn-upload').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', async ev => {
    const files = ev.target.files;
    if (!files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    $('hud-stage').textContent = 'uploading photos';
    try {
      const r = await api('/api/upload/browser', {method: 'POST', body: fd});
      toast(`${r.uploaded} photo(s) ingested`);
    } catch (e) { toast('upload failed: ' + e.message); }
    $('hud-stage').textContent = 'idle';
    ev.target.value = '';
    await refresh();
  });

  // ---- twin toggles ----------------------------------------------------
  $('t-cloud').addEventListener('change', e =>
    App.groups.cloud.visible = e.target.checked);
  $('t-mesh').addEventListener('change', e =>
    App.groups.mesh.visible = e.target.checked);
  $('t-rebar').addEventListener('change', e =>
    App.groups.rebar.visible = e.target.checked);
  $('t-cams').addEventListener('change', e =>
    App.groups.cams.visible = e.target.checked);
  $('t-rays').addEventListener('change', e =>
    App.groups.rays.visible = e.target.checked);
  $('t-pulse').addEventListener('change', e => {
    App.groups.markers.visible = e.target.checked;
    if (!e.target.checked) App.markers.forEach(m => m.scale.setScalar(1));
  });
  $('t-grid').addEventListener('change', e =>
    App.groups.grid.visible = e.target.checked);
  $('t-axes').addEventListener('change', e =>
    App.groups.axes.visible = e.target.checked);
  $('t-color').addEventListener('change', () => loadScene());

  // ---- camera helpers --------------------------------------------------
  $('btn-fit').addEventListener('click', () => {
    App.controls.target.set(0, 1.45, 0);
    App.camera.position.set(3.4, 2.3, 3.9);
  });
  $('btn-spin').addEventListener('click', () => App.spin = !App.spin);
  $('btn-camcycle').addEventListener('click', () => {
    const cams = App.groups.cams.children;
    if (!cams.length) return;
    const g = cams[App.camCycle % cams.length];
    App.camCycle++;
    const p = g.position.clone();
    App.controls.target.set(0, 1.45, 0);
    App.camera.position.copy(p.clone().multiplyScalar(1.35)
      .add(new THREE.Vector3(0, 0.35, 0)));
    App.groups.cams.children.forEach(c => c.scale.setScalar(1));
    g.scale.setScalar(1.5);            // highlight the selected node
    $('hud-cam').textContent = 'camera ' + g.userData.view;
  });

  // ---- modals ----------------------------------------------------------
  $('btn-help').addEventListener('click', () => $('help').classList.remove('hidden'));
  $('btn-help-close').addEventListener('click', () => $('help').classList.add('hidden'));
  $('btn-modal-close').addEventListener('click', () => $('modal').classList.add('hidden'));
}

/* ========================================================================
   6. BOOT
   ======================================================================== */
(async function boot() {
  initThree();
  wire();

  // firmware preview
  try {
    const fw = await api('/api/firmware');
    const lines = fw.split('\n');
    const head = lines.slice(0, 8).join('\n');
    const pins = lines.filter(l => /#define (PWDN|XCLK|SIOD|SIOC|Y[0-9]|VSYNC|HREF|PCLK|PIN_)/.test(l));
    $('fw-snippet').textContent = head + '\n\n// camera pin map (16-pin AI-Thinker)\n'
      + pins.slice(0, 22).join('\n') + '\n\n// … full sketch: /api/firmware';
  } catch (e) { $('fw-snippet').textContent = 'firmware unavailable'; }

  await refresh();
  await loadScene();
  setInterval(async () => {           // keep the fleet/HUD fresh
    try { const s = await api('/api/status'); App.state = s;
          renderHeader(); renderFleet(); } catch (e) {}
  }, 15000);
})();
