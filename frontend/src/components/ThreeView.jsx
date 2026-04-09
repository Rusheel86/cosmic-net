import React, { useRef, useEffect } from "react";
import * as THREE from "three";

export default function ThreeView({ csvData, explainResult, selectedNode, onSelectNode }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const nodeObjectsRef = useRef([]);
  const edgeObjectsRef = useRef([]);

  useEffect(() => {
    if (!mountRef.current || !csvData) return;

    const W = mountRef.current.clientWidth;
    const H = mountRef.current.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050510);
    sceneRef.current = scene;

    // Stars background
    const starGeo = new THREE.BufferGeometry();
    const starVerts = [];
    for (let i = 0; i < 2000; i++) {
      starVerts.push((Math.random() - 0.5) * 400);
      starVerts.push((Math.random() - 0.5) * 400);
      starVerts.push((Math.random() - 0.5) * 400);
    }
    starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starVerts, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.4, transparent: true, opacity: 0.6 });
    scene.add(new THREE.Points(starGeo, starMat));

    // Camera
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
    camera.position.set(0, 0, 60);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    scene.add(new THREE.AmbientLight(0x223355, 2));
    const pointLight = new THREE.PointLight(0x4a9eff, 3, 200);
    pointLight.position.set(20, 20, 20);
    scene.add(pointLight);

    // Parse CSV
    const rows = csvData.trim().split("\n");
    const headers = rows[0].split(",").map(h => h.trim());
    const data = rows.slice(1).map(r => {
      const vals = r.split(",");
      return Object.fromEntries(headers.map((h, i) => [h, parseFloat(vals[i])]));
    });

    // Normalize positions to scene space
    const xs = data.map(d => d.pos_x);
    const ys = data.map(d => d.pos_y);
    const zs = data.map(d => d.pos_z);
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const cz = (Math.max(...zs) + Math.min(...zs)) / 2;
    const spread = Math.max(Math.max(...xs) - Math.min(...xs), 1);
    const scale = 40 / spread;

    const positions = data.map(d => ({
      x: (d.pos_x - cx) * scale,
      y: (d.pos_y - cy) * scale,
      z: (d.pos_z - cz) * scale,
    }));

    // Mass range for node sizing
    const masses = data.map(d => d.stellar_mass);
    const minM = Math.min(...masses);
    const maxM = Math.max(...masses);
    const nodeSize = m => 0.4 + ((m - minM) / (maxM - minM + 1e-8)) * 1.2;

    // Draw nodes
    nodeObjectsRef.current = [];
    data.forEach((row, i) => {
      const geo = new THREE.SphereGeometry(nodeSize(row.stellar_mass), 16, 16);
      const isSelected = selectedNode === i;
      const mat = new THREE.MeshPhongMaterial({
        color: isSelected ? 0xff4757 : 0x4a9eff,
        emissive: isSelected ? 0x991122 : 0x001133,
        shininess: 80,
        transparent: true,
        opacity: 0.9,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(positions[i].x, positions[i].y, positions[i].z);
      mesh.userData = { index: i, data: row };
      scene.add(mesh);
      nodeObjectsRef.current.push(mesh);
    });

    // Draw edges
    edgeObjectsRef.current = [];
    const edgeImportances = explainResult?.edge_importances || [];

    if (edgeImportances.length > 0) {
      edgeImportances.forEach(e => {
        const src = positions[e.source];
        const dst = positions[e.target];
        if (!src || !dst) return;
        const pts = [new THREE.Vector3(src.x, src.y, src.z), new THREE.Vector3(dst.x, dst.y, dst.z)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const color = e.importance > 0.6 ? 0xff4757 : e.importance > 0.3 ? 0xffb800 : 0x2a4a7a;
        const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 + e.importance * 0.5 });
        const line = new THREE.Line(geo, mat);
        scene.add(line);
        edgeObjectsRef.current.push(line);
      });
    } else {
      // Default k=3 nearest edges
      data.forEach((a, i) => {
        const dists = data.map((b, j) => ({
          j,
          d: Math.sqrt((a.pos_x - b.pos_x) ** 2 + (a.pos_y - b.pos_y) ** 2 + (a.pos_z - b.pos_z) ** 2)
        })).filter(x => x.j !== i).sort((a, b) => a.d - b.d).slice(0, 3);
        dists.forEach(({ j }) => {
          const pts = [
            new THREE.Vector3(positions[i].x, positions[i].y, positions[i].z),
            new THREE.Vector3(positions[j].x, positions[j].y, positions[j].z),
          ];
          const geo = new THREE.BufferGeometry().setFromPoints(pts);
          const mat = new THREE.LineBasicMaterial({ color: 0x2a4a7a, transparent: true, opacity: 0.3 });
          scene.add(new THREE.Line(geo, mat));
        });
      });
    }

    // Mouse click raycasting
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onClick = (e) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(nodeObjectsRef.current);
      if (hits.length > 0) {
        onSelectNode && onSelectNode(hits[0].object.userData.index, hits[0].object.userData.data);
      }
    };
    renderer.domElement.addEventListener("click", onClick);

    // Auto-rotate
    let angle = 0;
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      angle += 0.003;
      camera.position.x = Math.sin(angle) * 60;
      camera.position.z = Math.cos(angle) * 60;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameRef.current);
      renderer.domElement.removeEventListener("click", onClick);
      if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [csvData, explainResult, selectedNode]);

  return (
    <div
      ref={mountRef}
      style={{ width: "100%", height: "340px", borderRadius: "8px", overflow: "hidden", cursor: "crosshair" }}
    />
  );
}
