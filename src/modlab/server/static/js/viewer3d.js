// Minimal three.js mesh viewer (global THREE from the UMD build loaded in index.html).
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function viewer(host) {
  const T = window.THREE;
  if (!T) { host.innerHTML = '<p class="cap" style="padding:16px">The 3D viewer needs three.js, which did not load. Check the network connection.</p>'; return null; }
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);
  const hint = document.createElement('span'); hint.className = 'hint'; hint.textContent = 'drag to rotate · scroll to zoom'; host.appendChild(hint);
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(35, 4 / 3, 0.01, 1000);
  scene.add(new T.HemisphereLight(0xffffff, 0x445566, 0.85));
  const key = new T.DirectionalLight(0xffffff, 0.75); key.position.set(2, 3, 4); camera.add(key); scene.add(camera);
  const controls = new T.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  let mesh = null, geo = null, framed = false;
  const material = new T.MeshStandardMaterial({ color: 0x8fa3bf, roughness: 0.65, metalness: 0.05, side: T.DoubleSide, flatShading: false });

  function size() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(size).observe(host);

  function frame(positions) {
    geo.computeBoundingSphere();
    const s = geo.boundingSphere;
    controls.target.copy(s.center);
    camera.position.copy(s.center).add(new T.Vector3(0.6, 0.35, 1).normalize().multiplyScalar(s.radius * 3.2));
    camera.near = s.radius / 100; camera.far = s.radius * 50; camera.updateProjectionMatrix();
    framed = true;
  }

  const api = {
    /* set(vertices flat, faces flat, {colors flat rgb 0..1, refit}) */
    set(vertices, faces, { colors = null, refit = false } = {}) {
      const pos = new Float32Array(vertices);
      if (!geo || faces) {
        if (mesh) { scene.remove(mesh); geo.dispose(); }
        geo = new T.BufferGeometry();
        geo.setAttribute('position', new T.BufferAttribute(pos, 3));
        geo.setIndex(Array.from(faces));
        mesh = new T.Mesh(geo, material); scene.add(mesh);
        framed = false;
      } else {
        geo.attributes.position.array.set(pos); geo.attributes.position.needsUpdate = true;
      }
      if (colors) {
        geo.setAttribute('color', new T.BufferAttribute(new Float32Array(colors), 3));
        material.vertexColors = true; material.color.set(0xffffff);
      } else if (material.vertexColors) {
        geo.deleteAttribute('color'); material.vertexColors = false; material.color.set(0x8fa3bf);
      }
      material.needsUpdate = true;
      geo.computeVertexNormals();
      if (!framed || refit) frame();
      size();
    },
  };
  (function loop() { requestAnimationFrame(loop); if (host.offsetParent !== null) { controls.update(); renderer.render(scene, camera); } })();
  return api;
}

/* Colour helpers for per-vertex scalars. */
export function diverging(values, lim) {
  const out = new Float32Array(values.length * 3);
  const neg = [0.16, 0.47, 0.84], mid = [0.86, 0.87, 0.89], pos = [0.89, 0.29, 0.28];
  values.forEach((v, i) => {
    const t = Math.max(-1, Math.min(1, (v || 0) / (lim || 1)));
    const c = t < 0 ? neg : pos, a = Math.abs(t);
    for (let j = 0; j < 3; j++) out[i * 3 + j] = mid[j] + (c[j] - mid[j]) * a;
  });
  return out;
}

export function categorical(labels) {
  const pal = [[0.16, 0.47, 0.84], [0.92, 0.41, 0.2], [0.11, 0.69, 0.48], [0.93, 0.63, 0], [0.91, 0.48, 0.64], [0, 0.51, 0], [0.29, 0.23, 0.65], [0.89, 0.29, 0.28], [0.45, 0.5, 0.56], [0.6, 0.4, 0.2]];
  const out = new Float32Array(labels.length * 3);
  labels.forEach((l, i) => { const c = pal[l % pal.length]; out.set(c, i * 3); });
  return out;
}
