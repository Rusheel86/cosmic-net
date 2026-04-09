import React, { useState, useCallback, useRef, useEffect } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import Plot from "react-plotly.js";
import axios from "axios";
import * as THREE from "three";

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";
const api = axios.create({ baseURL: BASE });

// ─── Sample clusters ────────────────────────────────────────────────────────
const SAMPLES = {
  "Virgo-class": `pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,stellar_mass,vel_dispersion,half_mass_radius,metallicity
12.3,45.6,78.9,-120.5,340.2,89.1,10.2,450.3,12.5,0.012
14.1,43.2,80.1,200.3,-150.4,310.2,11.1,620.1,18.3,0.018
11.8,47.1,77.3,-80.2,210.6,-200.3,9.8,380.5,9.1,0.009
13.5,44.8,79.5,150.8,-90.3,180.7,10.9,510.2,15.7,0.015
12.9,46.3,78.1,-200.1,120.5,250.4,10.5,490.7,13.2,0.013
15.2,42.7,81.3,310.2,-280.6,90.8,11.4,670.4,21.4,0.021
11.2,48.5,76.8,90.4,180.3,-310.2,9.4,350.2,7.8,0.007
13.1,45.1,79.9,-170.3,260.7,140.5,10.7,530.8,16.1,0.016
12.5,46.8,78.5,240.6,-130.2,220.9,10.3,465.1,11.9,0.011
14.8,43.9,80.7,-290.4,310.8,-160.3,11.2,640.3,19.6,0.019
10.9,49.2,76.2,180.7,-220.5,380.1,9.1,320.6,6.4,0.006
13.8,44.4,79.2,130.2,90.6,-270.8,11.0,555.9,17.3,0.017
12.1,47.6,77.8,-310.8,170.4,150.6,10.1,425.3,10.8,0.010
15.5,42.1,81.8,270.5,-360.3,100.2,11.6,700.2,23.1,0.023
11.5,48.0,77.1,380.1,140.8,-190.4,9.6,365.7,8.2,0.008`,

  "Coma-class": `pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,stellar_mass,vel_dispersion,half_mass_radius,metallicity
8.1,22.3,41.5,-280.4,520.1,190.3,11.8,890.2,28.4,0.028
9.4,20.1,43.2,410.5,-380.2,520.1,12.3,1050.4,35.2,0.034
7.6,24.7,39.8,-190.3,380.6,-410.5,11.2,750.8,22.1,0.021
10.2,19.4,44.6,340.8,-290.4,380.2,12.1,980.3,31.8,0.031
8.8,23.1,41.9,-350.2,290.8,450.6,11.6,840.5,26.3,0.025
11.1,18.2,45.9,480.3,-480.1,280.4,12.5,1120.6,38.9,0.038
7.2,25.8,38.4,220.6,410.3,-520.8,10.9,680.2,18.7,0.018
9.7,21.6,42.8,-420.4,350.7,390.1,11.9,920.8,30.1,0.029
8.4,23.8,40.6,360.1,-310.5,430.7,11.4,800.3,24.5,0.023
10.8,19.9,44.1,-490.2,430.4,-340.6,12.2,1010.5,33.4,0.032
7.9,25.2,39.2,290.7,260.8,-480.3,11.1,720.1,20.3,0.019
9.1,22.7,42.3,-310.6,480.2,360.9,11.7,870.4,27.6,0.026
8.6,24.1,41.1,450.3,-350.8,310.2,11.5,820.7,25.1,0.024
10.5,20.4,43.7,-380.9,320.5,470.8,12.0,960.2,32.6,0.030
7.4,26.1,38.9,510.4,280.3,-390.7,10.8,660.5,17.4,0.016`,

  "Perseus-class": `pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,stellar_mass,vel_dispersion,half_mass_radius,metallicity
5.2,11.4,23.8,-180.3,620.4,230.1,12.6,1280.5,42.3,0.041
6.1,9.8,25.6,520.8,-490.3,610.2,13.1,1480.2,52.1,0.048
4.8,12.9,22.4,-310.5,480.7,-590.4,12.1,1090.8,36.2,0.035
6.8,8.6,26.9,450.2,-580.1,480.6,13.4,1580.4,58.4,0.054
5.6,11.9,24.5,-420.4,350.9,570.3,12.8,1360.1,46.8,0.044
7.2,7.8,28.1,590.6,-420.8,350.5,13.7,1690.3,65.2,0.061
4.4,13.8,21.7,240.8,530.4,-640.2,11.8,980.5,30.4,0.029
6.4,10.4,25.1,-560.3,390.6,420.8,13.0,1440.7,49.6,0.046
5.0,12.5,23.3,490.1,-360.5,510.7,12.4,1190.3,40.1,0.038
6.9,9.2,26.4,-630.7,450.2,-380.4,13.3,1540.6,55.7,0.051
4.6,13.3,22.1,320.5,280.9,-560.3,11.5,870.2,25.8,0.024
5.8,11.1,24.2,-390.2,560.4,330.6,12.9,1400.4,48.2,0.045
5.3,12.2,23.6,610.4,-410.7,290.8,12.5,1240.8,43.9,0.042
6.6,9.6,25.8,-470.8,310.3,590.5,13.2,1510.1,53.8,0.049
4.9,13.1,22.7,380.6,490.1,-450.9,11.9,1020.6,32.7,0.031`,
};

const parseCsv = (text) => {
  const rows = text.trim().split("\n");
  const headers = rows[0].split(",").map(h => h.trim());
  return rows.slice(1).map(r => {
    const vals = r.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, parseFloat(vals[i])]));
  });
};

// ─── Equation computation ────────────────────────────────────────────────────
const computeEquation = (df) => {
  if (!df || df.length === 0) return null;
  const sorted = [...df].sort((a, b) => b.stellar_mass - a.stellar_mass);
  const cen = sorted[0];
  const sats = sorted.slice(1);
  const logMcen   = cen.stellar_mass;
  const logSigCen = Math.log10(Math.max(cen.vel_dispersion, 1e-8));
  const logRcen   = Math.log10(Math.max(cen.half_mass_radius, 1e-8));
  const logSigSat = sats.length > 0 ? Math.log10(sats.reduce((s,r) => s + r.vel_dispersion, 0) / sats.length) : 0;
  const logNsat   = Math.log10(Math.max(df.length - 1, 1));
  const mSatTot   = sats.reduce((s,r) => s + Math.pow(10, r.stellar_mass), 0);
  const logMsat   = mSatTot > 0 ? Math.log10(mSatTot) : 0;
  const vals = [logMcen, logSigCen, logRcen, logSigSat, logNsat, logMsat];
  const coeffs = [0.022, 0.288, 0.079, 0.024, 0.001, -0.001];
  const labels = ["log(M★_cen)", "log(σ_cen)", "log(R½_cen)", "log(σ_sat)", "log(N_sat)", "log(M★_sat)"];
  const dominant = [false, true, false, false, false, false];
  const terms = coeffs.map((c, i) => ({ coef: c, label: labels[i], rawVal: vals[i], contrib: c * vals[i], dominant: dominant[i] }));
  const total = 12.968 + terms.reduce((s, t) => s + t.contrib, 0);
  const fj = 4 * logSigCen + 8.5;
  return { terms, total, fj, logSigCen, cen };
};

// ─── Three.js component ──────────────────────────────────────────────────────
function ThreeCanvas({ parsedDf, explainResult, selectedIdx, onSelect }) {
  const mountRef = useRef(null);
  useEffect(() => {
    if (!mountRef.current || !parsedDf) return;
    const W = mountRef.current.clientWidth, H = mountRef.current.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0F172A); // matches slate-900
    const starGeo = new THREE.BufferGeometry();
    const sv = []; for (let i=0;i<3000;i++){sv.push((Math.random()-.5)*500,(Math.random()-.5)*500,(Math.random()-.5)*500);}
    starGeo.setAttribute("position", new THREE.Float32BufferAttribute(sv,3));
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({color:0xaabbdd,size:.3,transparent:true,opacity:.5})));
    const cam = new THREE.PerspectiveCamera(55, W/H, .1, 1000);
    cam.position.set(0,0,65);
    const renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(W,H); renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0x334466,2));
    const pl = new THREE.PointLight(0x4a9eff,4,200); pl.position.set(30,30,30); scene.add(pl);
    const xs=parsedDf.map(d=>d.pos_x),ys=parsedDf.map(d=>d.pos_y),zs=parsedDf.map(d=>d.pos_z);
    const cx=(Math.max(...xs)+Math.min(...xs))/2,cy=(Math.max(...ys)+Math.min(...ys))/2,cz=(Math.max(...zs)+Math.min(...zs))/2;
    const sp=Math.max(Math.max(...xs)-Math.min(...xs),1),sc=42/sp;
    const pos=parsedDf.map(d=>({x:(d.pos_x-cx)*sc,y:(d.pos_y-cy)*sc,z:(d.pos_z-cz)*sc}));
    const masses=parsedDf.map(d=>d.stellar_mass),minM=Math.min(...masses),maxM=Math.max(...masses);
    const nsize=m=>1.0+((m-minM)/(maxM-minM+1e-8))*2.0; // slightly thicker nodes
    const meshes=[];
    parsedDf.forEach((row,i)=>{
      const geo=new THREE.SphereGeometry(nsize(row.stellar_mass),24,24);
      const isSel=selectedIdx===i;
      const mat=new THREE.MeshPhongMaterial({
        color:isSel?0xF59E0B:0x3B82F6, // amber-500 or blue-500
        emissive:isSel?0x664400:0x001040,
        shininess:100,transparent:true,opacity:.92
      });
      const m=new THREE.Mesh(geo,mat); m.position.set(pos[i].x,pos[i].y,pos[i].z); m.userData={i,row};
      scene.add(m); meshes.push(m);
    });
    const ei=explainResult?.edge_importances||[];
    if(ei.length>0){
      ei.forEach(e=>{
        const s=pos[e.source],d=pos[e.target]; if(!s||d === undefined) return;
        const pts=[new THREE.Vector3(s.x,s.y,s.z),new THREE.Vector3(d.x,d.y,d.z)];
        const c=e.importance>.6?0xEF4444:e.importance>.3?0xF59E0B:0x1E293B; // red/amber/slate
        const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:c,transparent:true,opacity:.4+e.importance*.6, linewidth: 2}));
        scene.add(line);
      });
    } else {
      parsedDf.forEach((a,i)=>{
        const dists=parsedDf.map((b,j)=>({j,d:Math.sqrt((a.pos_x-b.pos_x)**2+(a.pos_y-b.pos_y)**2+(a.pos_z-b.pos_z)**2)})).filter(x=>x.j!==i).sort((a,b)=>a.d-b.d).slice(0,3);
        dists.forEach(({j})=>{
          const pts=[new THREE.Vector3(pos[i].x,pos[i].y,pos[i].z),new THREE.Vector3(pos[j].x,pos[j].y,pos[j].z)];
          scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x334155,transparent:true,opacity:.4})));
        });
      });
    }
    const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
    const onClick=e=>{
      const rect=renderer.domElement.getBoundingClientRect();
      mouse.x=((e.clientX-rect.left)/rect.width)*2-1;
      mouse.y=-((e.clientY-rect.top)/rect.height)*2+1;
      ray.setFromCamera(mouse,cam);
      const hits=ray.intersectObjects(meshes);
      if(hits.length>0) onSelect&&onSelect(hits[0].object.userData.i,hits[0].object.userData.row);
    };
    renderer.domElement.addEventListener("click",onClick);
    let angle=0,raf;
    const animate=()=>{ raf=requestAnimationFrame(animate); angle+=.004; cam.position.x=Math.sin(angle)*65; cam.position.z=Math.cos(angle)*65; cam.lookAt(0,0,0); renderer.render(scene,cam); };
    animate();
    return ()=>{ cancelAnimationFrame(raf); renderer.domElement.removeEventListener("click",onClick); if(mountRef.current?.contains(renderer.domElement)) mountRef.current.removeChild(renderer.domElement); renderer.dispose(); };
  },[parsedDf,explainResult,selectedIdx]);
  return <div ref={mountRef} className="w-full h-full" style={{borderRadius:8,overflow:"hidden",cursor:"crosshair"}}/>;
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [csvText,setCsvText]=useState(null);
  const [df,setDf]=useState(null);
  const [loadedName,setLoadedName]=useState(null);
  const [overview,setOverview]=useState(null);
  const [pred,setPred]=useState(null);
  const [virial,setVirial]=useState(null);
  const [explain,setExplain]=useState(null);
  const [loading,setLoading]=useState(false);
  const [explaining,setExplaining]=useState(false);
  const [error,setError]=useState(null);
  const [selNode,setSelNode]=useState(null);
  const [selEdge,setSelEdge]=useState(null);
  const [view,setView]=useState("2d");
  const [sampleOpen,setSampleOpen]=useState(false);

  const load = useCallback(async(text,name)=>{
    setCsvText(text); setDf(parseCsv(text)); setLoadedName(name);
    setError(null); setPred(null); setVirial(null); setExplain(null); setSelNode(null); setSelEdge(null);
    try {
      const blob=new Blob([text],{type:"text/csv"});
      const form=new FormData(); form.append("file",blob,name);
      const res=await api.post("/overview",form);
      setOverview(res.data);
    } catch { setOverview(null); }
  },[]);

  const handleFile=async e=>{ const f=e.target.files[0]; if(!f) return; load(await f.text(),f.name); };
  const handleSample=name=>{ load(SAMPLES[name],`${name}.csv`); setSampleOpen(false); };

  const handlePredict=async()=>{
    if(!csvText) return; setLoading(true); setError(null);
    try {
      const mk=()=>{ const b=new Blob([csvText],{type:"text/csv"}); const f=new FormData(); f.append("file",b,"c.csv"); return f; };
      const [p,v]=await Promise.all([api.post("/predict",mk()),api.post("/virial",mk())]);
      setPred(p.data); setVirial(v.data);
    } catch(err){ setError(err.response?.data?.detail||"Prediction failed."); }
    finally { setLoading(false); }
  };

  const handleExplain=async()=>{
    if(!csvText) return; setExplaining(true);
    try {
      const blob=new Blob([csvText],{type:"text/csv"}); const form=new FormData(); form.append("file",blob,"c.csv");
      const res=await api.post("/explain",form); setExplain(res.data);
    } catch { setError("PGExplainer failed."); }
    finally { setExplaining(false); }
  };

  // Cytoscape elements
  const cyElements=()=>{
    if(!df) return [];
    const masses=df.map(r=>r.stellar_mass),minM=Math.min(...masses),maxM=Math.max(...masses);
    const sz=m=>30+((m-minM)/(maxM-minM+1e-8))*40;
    const nodes=df.map((row,i)=>({
      data:{id:`n${i}`,label:`${i}`,size:sz(row.stellar_mass),stellar_mass:row.stellar_mass?.toFixed(3),vel_dispersion:row.vel_dispersion?.toFixed(1),half_mass_radius:row.half_mass_radius?.toFixed(2),metallicity:row.metallicity?.toFixed(5)},
      position:{x:(row.pos_x-Math.min(...df.map(r=>r.pos_x)))*85+50,y:(row.pos_y-Math.min(...df.map(r=>r.pos_y)))*85+50},
    }));
    let edges=[];
    if(explain?.edge_importances){
      edges=explain.edge_importances.map((e,i)=>({data:{id:`e${i}`,source:`n${e.source}`,target:`n${e.target}`,importance:e.importance,width:2+e.importance*6,color:e.importance>.6?"#EF4444":e.importance>.3?"#F59E0B":"#334155"}}));
    } else {
      df.forEach((a,i)=>{
        df.map((b,j)=>({j,d:Math.hypot(a.pos_x-b.pos_x,a.pos_y-b.pos_y)})).filter(x=>x.j!==i).sort((a,b)=>a.d-b.d).slice(0,3).forEach(({j})=>edges.push({data:{id:`e${i}_${j}`,source:`n${i}`,target:`n${j}`,width:2,color:"#334155"}}));
      });
    }
    return [...nodes,...edges];
  };

  const anchorSet=new Set(explain?.top_anchor_indices?.map(i=>`n${i}`)||[]);
  const cyStyle=[
    {selector:"node",style:{width:"data(size)",height:"data(size)",backgroundColor:"#1E293B",borderWidth:2,borderColor:"#3B82F6",label:"data(label)",color:"#F8FAFC",fontSize:13,textValign:"center",fontFamily:"'IBM Plex Mono', monospace"}},
    {selector:anchorSet.size>0?[...anchorSet].map(id=>`node[id="${id}"]`).join(","):".x",style:{backgroundColor:"#7F1D1D",borderColor:"#EF4444",borderWidth:4}},
    {selector:`node[id="n${selNode?.idx}"]`,style:{backgroundColor:"#78350F",borderColor:"#F59E0B",borderWidth:4}},
    {selector:"edge",style:{width:"data(width)",lineColor:"data(color)",opacity:.8,curveStyle:"bezier"}},
  ];

  const plotCfg={displayModeBar:false, responsive: true};
  const plotLay=(title,xa,ya)=>({
    title:{text:title,font:{color:"#E2E8F0",size:14,family:"Inter"}},
    xaxis:{title:xa,color:"#94A3B8",gridcolor:"#334155",tickfont:{size:11,color:"#94A3B8"}},
    yaxis:{title:ya,color:"#94A3B8",gridcolor:"#334155",tickfont:{size:11,color:"#94A3B8"}},
    paper_bgcolor:"transparent",plot_bgcolor:"transparent",
    font:{color:"#F8FAFC",size:12,family:"Inter"},
    margin:{t:40,b:40,l:50,r:20},
    showlegend:false,
  });
  const eqData=computeEquation(df);

  return (
    <>
      {/* ── Starfield bg ── */}
      <div style={{position:"fixed",inset:0,backgroundImage:"radial-gradient(1px 1px at 20% 30%, rgba(59,130,246,0.1) 0%, transparent 100%), radial-gradient(1px 1px at 80% 10%, rgba(59,130,246,0.15) 0%, transparent 100%), radial-gradient(1px 1px at 50% 70%, rgba(96,165,250,0.1) 0%, transparent 100%)",pointerEvents:"none",zIndex:0}}/>

      {/* ── Header ── */}
      <header className="solid-header flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-xl font-bold text-blue tracking-wide w-full" style={{letterSpacing: "0.1em"}}>✦ COSMIC-NET</div>
            <div className="text-xs text-muted tracking-wide mt-1">DARK MATTER HALO MASS PREDICTOR</div>
          </div>
          <div style={{width:1,height:40,background:"var(--border-subtle)"}}/>
          <div className="flex gap-2">
            {["GNN R²=0.924","MC-Dropout","PGExplainer","PySR"].map(b=>(
              <span key={b} className="badge">{b}</span>
            ))}
          </div>
        </div>
        <div className="text-sm text-muted font-medium">IllustrisTNG-100 · Snapshot 99 · z=0</div>
      </header>

      {/* ── Toolbar ── */}
      <div className="toolbar">
        {/* Sample dropdown */}
        <div style={{position:"relative"}}>
          <button onClick={()=>setSampleOpen(o=>!o)} className="btn btn-secondary text-sm">
            ⚡ LOAD SAMPLE
            <span className="opacity-60 text-xs">▼</span>
          </button>
          {sampleOpen && (
            <div className="dropdown-menu">
              {Object.keys(SAMPLES).map(name=>(
                <button key={name} onClick={()=>handleSample(name)} className="dropdown-item">
                  {name}
                </button>
              ))}
            </div>
          )}
        </div>

        <label className="btn btn-outline text-sm" style={{borderStyle: 'dashed'}}>
          <input type="file" accept=".csv" onChange={handleFile} style={{display:"none"}}/>
          📂 {loadedName||"UPLOAD CSV"}
        </label>

        <div style={{width:1,height:32,background:"var(--border-subtle)"}}/>

        <button
          onClick={handlePredict}
          disabled={!csvText||loading}
          className="btn btn-primary"
        >
          {loading?"⟳ RUNNING...":"▶ PREDICT"}
        </button>

        <button
          onClick={handleExplain}
          disabled={!csvText||explaining}
          className="btn btn-secondary"
          style={{color: "var(--accent-red)"}}
        >
          {explaining?"⟳ EXPLAINING...":"❓ WHY?"}
        </button>

        {error && <span className="text-sm font-semibold text-red">⚠ {error}</span>}
        {loadedName && !error && <span className="text-sm text-blue ml-2">{df?.length} subhalos · {loadedName}</span>}
      </div>

      {/* ── Main layout ── */}
      <div className="dashboard-grid">

        {/* ── LEFT PANEL ── */}
        <div className="left-sidebar">
          {/* Graph header */}
          <div className="panel-header flex items-center justify-between" style={{padding: "16px 24px", flexShrink: 0}}>
            <div className="flex gap-2">
              {[["2d","2D GRAPH"],["3d","3D SPACE"]].map(([k,l])=>(
                <button key={k} onClick={()=>setView(k)} className={`btn text-xs ${view === k ? 'btn-primary' : 'btn-secondary'}`}>
                  {l}
                </button>
              ))}
            </div>
            {explain && (
              <div className="flex gap-4 text-xs font-semibold">
                <span className="flex items-center gap-1 text-red"><span className="status-indicator bg-red"></span> HIGH</span>
                <span className="flex items-center gap-1 text-amber"><span className="status-indicator bg-amber"></span> MED</span>
                <span className="flex items-center gap-1 text-muted"><span className="status-indicator" style={{background: 'var(--border-subtle)'}}></span> LOW</span>
                <span className="flex items-center gap-1 text-red">★ ANCHOR</span>
              </div>
            )}
          </div>

          {/* Graph view */}
          <div className="w-full flex-grow relative" style={{background: "var(--bg-main)"}}>
            {!csvText && (
              <div className="absolute w-full h-full flex flex-col items-center justify-center gap-4">
                <div className="text-5xl opacity-60 text-blue">✦</div>
                <div className="text-lg font-bold tracking-widest text-muted">LOAD A CLUSTER TO BEGIN</div>
                <div className="text-sm text-muted">SELECT FROM SAMPLES OR UPLOAD CSV</div>
              </div>
            )}
            {csvText && view==="2d" && (
              <CytoscapeComponent
                elements={cyElements()}
                stylesheet={cyStyle}
                style={{width:"100%",height:"100%",background:"transparent"}}
                cy={cy=>{
                  cy.on("tap","node",e=>{const d=e.target.data();const idx=parseInt(d.id.replace("n",""));setSelNode({idx,data:df[idx]});setSelEdge(null);});
                  cy.on("tap","edge",e=>{setSelEdge(e.target.data());setSelNode(null);});
                  cy.on("tap",e=>{if(e.target===cy){setSelNode(null);setSelEdge(null);}});
                }}
              />
            )}
            {csvText && view==="3d" && (
              <ThreeCanvas parsedDf={df} explainResult={explain} selectedIdx={selNode?.idx} onSelect={(i,row)=>{setSelNode({idx:i,data:row});setSelEdge(null);}}/>
            )}
          </div>

          {/* Selected item detail */}
          {(selNode||selEdge) && (
            <div className="panel-header" style={{borderTop: "1px solid var(--border-subtle)", borderBottom: "none", flexShrink: 0}}>
              {selNode && (
                <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
                  <div style={{gridColumn:"1/-1",display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4}}>
                    <span className="text-sm font-bold text-amber tracking-wide">SUBHALO #{selNode.idx}</span>
                    <button onClick={()=>setSelNode(null)} className="btn btn-secondary" style={{padding: "4px 8px"}}>✕</button>
                  </div>
                  {[["log(M★)",selNode.data?.stellar_mass?.toFixed(4)],["σ_vel km/s",selNode.data?.vel_dispersion?.toFixed(1)],["R½ kpc",selNode.data?.half_mass_radius?.toFixed(2)],["Metallicity Z",selNode.data?.metallicity?.toFixed(5)],["pos_x",selNode.data?.pos_x?.toFixed(2)],["pos_y",selNode.data?.pos_y?.toFixed(2)]].map(([k,v])=>(
                    <div key={k} className="stat-card">
                      <div className="stat-label">{k}</div>
                      <div className="stat-value">{v||"—"}</div>
                    </div>
                  ))}
                </div>
              )}
              {selEdge && !selNode && (
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-4">
                    <span className="text-sm font-bold text-amber tracking-wide">EDGE</span>
                    <span className="text-lg font-mono font-semibold">{selEdge.source} → {selEdge.target}</span>
                    {selEdge.importance!=null && (
                      <span className={`text-sm font-bold ${selEdge.importance>.6?'text-red':selEdge.importance>.3?'text-amber':'text-muted'}`}>
                        {(selEdge.importance*100).toFixed(1)}% importance
                      </span>
                    )}
                  </div>
                  <button onClick={()=>setSelEdge(null)} className="btn btn-secondary" style={{padding: "4px 8px"}}>✕</button>
                </div>
              )}
            </div>
          )}

          {/* Data overview charts */}
          {overview && (
            <div className="panel-header" style={{borderTop:"1px solid var(--border-subtle)", borderBottom: "none", paddingBottom: "24px"}}>
              <div className="text-xs font-semibold text-muted tracking-wide mb-4">DATA OVERVIEW — {overview.n_subhalos} SUBHALOS</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16}}>
                <Plot data={[{x:overview.stellar_mass_log,type:"histogram",marker:{color:"#3B82F6",line:{color:"#60A5FA",width:1}}}]} layout={plotLay("Stellar Mass","log(M★)","n")} style={{width:"100%",height:160}} config={plotCfg}/>
                <Plot data={[{x:overview.stellar_mass_log,y:overview.velocity_dispersion,mode:"markers",type:"scatter",marker:{color:"#10B981",size:6,line:{color:"#34D399",width:1}}}]} layout={plotLay("σ vs M★","log(M★)","σ")} style={{width:"100%",height:160}} config={plotCfg}/>
                <Plot data={[{z:overview.correlation_matrix,x:overview.correlation_labels,y:overview.correlation_labels,type:"heatmap",colorscale:[[0,"#0F172A"],[.5,"#1E293B"],[1,"#3B82F6"]],zmin:-1,zmax:1,showscale:false}]} layout={plotLay("Correlation","","")} style={{width:"100%",height:160}} config={plotCfg}/>
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div className="right-sidebar">

          {/* GNN Prediction */}
          <div className="sidebar-section">
            <div className="text-sm font-bold text-muted tracking-widest mb-4">GNN PREDICTION</div>
            {!pred && !loading && (
              <div className="text-muted text-center py-6">RUN PREDICTION TO SEE RESULTS</div>
            )}
            {loading && <div className="text-blue font-semibold animate-pulse py-4">⟳ Running 30 MC-Dropout passes...</div>}
            {pred && (
              <div className="card">
                <div className="mb-4">
                  <div className="text-sm text-muted mb-1">log₁₀(M_halo / M☉)</div>
                  <div className="text-5xl font-bold font-mono text-blue leading-none py-2">
                    {pred.gnn_log_mass_mean.toFixed(3)}
                    <span className="text-2xl text-muted font-sans font-medium ml-2">dex</span>
                  </div>
                  <div className="text-xs text-muted mt-2 font-mono">
                    95% CI [{pred.gnn_log_mass_lower.toFixed(3)}, {pred.gnn_log_mass_upper.toFixed(3)}] · σ={pred.gnn_log_mass_std.toFixed(4)}
                  </div>
                </div>

                {/* Confidence bar */}
                <div style={{background:"var(--bg-main)",borderRadius:4,height:8,marginBottom:16,overflow:"hidden", border: "1px solid var(--border-subtle)"}}>
                  <div style={{height:"100%",background:"linear-gradient(90deg, #3B82F6, #60A5FA)",width:`${Math.min(100,Math.max(10,(1-pred.gnn_log_mass_std*5)*100))}%`,transition:"width .6s"}}/>
                </div>

                {virial && (
                  <div className={`p-3 rounded border mb-4 flex items-center gap-3 ${virial.status === 'red' ? 'bg-red-dim' : virial.status === 'amber' ? 'bg-amber-dim' : 'bg-green-dim'}`}>
                    <div className={`status-indicator flex-shrink-0 ${virial.status === 'red' ? 'bg-red' : virial.status === 'amber' ? 'bg-amber' : 'bg-green'}`}/>
                    <div>
                      <div className="text-xs font-bold uppercase tracking-wide mb-1" style={{color: `var(--accent-${virial.status === 'amber' ? 'amber' : virial.status === 'red' ? 'red' : 'emerald'})`}}>
                        {virial.status} · Q = {virial.virial_ratio.toFixed(3)}
                      </div>
                      <div className="text-xs text-muted leading-relaxed">
                        {virial.message.slice(0,80)}{virial.message.length>80?"...":""}
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex gap-4 text-xs font-semibold text-muted pt-4 border-t border-solid" style={{borderColor: "var(--border-subtle)"}}>
                  <span className="bg-main px-2 py-1 rounded">{pred.n_subhalos} subhalos</span>
                  <span className="bg-main px-2 py-1 rounded">{pred.n_edges} edges</span>
                  <span className="bg-main px-2 py-1 rounded">30 MC passes</span>
                </div>
              </div>
            )}
          </div>

          {/* Equation Calculator */}
          <div className="sidebar-section">
            <div className="text-sm font-bold text-muted tracking-widest mb-1">SYMBOLIC EQUATION · LIVE CALCULATOR</div>
            <div className="text-xs text-muted mb-6">full_physics linear model · PySR discovered</div>

            {!eqData && (
              <div className="text-muted text-center py-6">LOAD A CLUSTER TO SEE BREAKDOWN</div>
            )}

            {eqData && (
              <div className="card">
                {/* Intercept */}
                <div className="flex justify-between items-center py-2 text-sm font-medium border-b border-solid" style={{borderColor: "var(--border-subtle)", marginBottom: 8}}>
                  <span className="text-muted">intercept</span>
                  <span className="font-mono">12.968</span>
                </div>

                {/* Terms */}
                {eqData.terms.map((t,i)=>(
                  <div key={i} className={`eq-row ${t.dominant ? 'dominant' : ''}`}>
                    <div className={`text-sm font-semibold ${t.dominant ? 'text-blue' : 'text-muted'}`}>
                      {t.dominant && <span className="text-blue mr-2">★</span>}
                      {t.label}
                    </div>
                    <div className={`text-sm font-mono text-right ${t.coef >= 0 ? 'text-emerald' : 'text-red'}`} style={{minWidth: 50}}>
                      {t.coef >= 0 ? "+" : ""}{t.coef.toFixed(3)}
                    </div>
                    <div className="text-sm font-mono text-muted text-right" style={{minWidth: 50}}>
                      {t.rawVal.toFixed(2)}
                    </div>
                    <div className={`text-sm font-mono text-right font-bold ${t.dominant ? 'text-blue' : t.contrib >= 0 ? 'text-main' : 'text-red'}`} style={{minWidth: 60}}>
                      {t.contrib >= 0 ? "+" : ""}{t.contrib.toFixed(3)}
                    </div>
                  </div>
                ))}

                <div className="divider"/>

                {/* Totals */}
                <div className="flex justify-between items-center p-3 rounded mb-3" style={{background: "var(--bg-main)"}}>
                  <span className="text-sm font-semibold text-muted">Equation total</span>
                  <span className="text-xl font-bold font-mono text-blue">{eqData.total.toFixed(3)}</span>
                </div>

                {pred && (
                  <>
                    <div className="flex justify-between items-center px-3 py-1">
                      <span className="text-sm text-muted">GNN prediction</span>
                      <span className="text-md font-bold font-mono text-emerald">{pred.gnn_log_mass_mean.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between items-center px-3 py-1 mb-2">
                      <span className="text-sm text-muted">
                        Difference {Math.abs(eqData.total-pred.gnn_log_mass_mean)<.1?"✓ good summary":Math.abs(eqData.total-pred.gnn_log_mass_mean)<.2?"⚠ marginal":"✗ unusual cluster"}
                      </span>
                      <span className={`text-md font-bold font-mono ${Math.abs(eqData.total-pred.gnn_log_mass_mean)<.1?'text-emerald':Math.abs(eqData.total-pred.gnn_log_mass_mean)<.2?'text-amber':'text-red'}`}>
                        {Math.abs(eqData.total-pred.gnn_log_mass_mean).toFixed(3)} dex
                      </span>
                    </div>
                  </>
                )}

                {/* Faber-Jackson */}
                <div className="mt-4 p-4 rounded border border-solid" style={{background: "rgba(16, 185, 129, 0.05)", borderColor: "rgba(16, 185, 129, 0.2)"}}>
                  <div className="text-xs font-bold text-emerald tracking-wide mb-2">FABER-JACKSON CONTEXT</div>
                  <div className="text-sm text-muted leading-relaxed font-mono">
                    σ_cen = {Math.pow(10,eqData.logSigCen).toFixed(0)} km/s<br/>
                    F-J alone → <strong className="text-emerald">{eqData.fj.toFixed(3)}</strong> dex<br/>
                    Full eq → <strong className="text-blue">{eqData.total.toFixed(3)}</strong> dex<br/>
                    <span className={`mt-2 block ${Math.abs(eqData.total-eqData.fj)<.05?'text-emerald':'text-blue'}`}>
                      Satellite correction = {(eqData.total-eqData.fj).toFixed(3)} dex
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Cluster stats */}
          {df && (
            <div className="sidebar-section border-none">
              <div className="text-sm font-bold text-muted tracking-widest mb-4">CLUSTER STATISTICS</div>
              <div className="grid w-full gap-4" style={{display: 'grid', gridTemplateColumns: '1fr 1fr'}}>
                {[
                  ["Subhalos",df.length],
                  ["M★ range",`${Math.min(...df.map(d=>d.stellar_mass)).toFixed(1)}–${Math.max(...df.map(d=>d.stellar_mass)).toFixed(1)}`],
                  ["Mean σ",`${(df.reduce((s,d)=>s+d.vel_dispersion,0)/df.length).toFixed(0)} km/s`],
                  ["Mean R½",`${(df.reduce((s,d)=>s+d.half_mass_radius,0)/df.length).toFixed(1)} kpc`],
                ].map(([k,v])=>(
                  <div key={k} className="stat-card">
                    <div className="stat-label">{k}</div>
                    <div className="stat-value">{v}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
