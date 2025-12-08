import { goToCheckout } from "./checkout";
import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import * as posedetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

// ★ chart.js
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Legend,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";
ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip);

// スケルトン描画用の接続ペア
const LINE_PAIRS = [
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
  ["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"]
];

// ---------- utility ----------
function mid(A, B) {
  return { x: (A.x + B.x) / 2, y: (A.y + B.y) / 2, score: Math.min(A.score ?? 1, B.score ?? 1) };
}
function angle(A, B, C) {
  const ab = { x: A.x - B.x, y: A.y - B.y };
  const cb = { x: C.x - B.x, y: C.y - B.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const m1 = Math.hypot(ab.x, ab.y) || 1e-6;
  const m2 = Math.hypot(cb.x, cb.y) || 1e-6;
  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}
function movingAvg(buf, val, max = 5) {
  if (val != null && isFinite(val)) buf.push(val);
  while (buf.length > max) buf.shift();
  if (!buf.length) return null;
  return buf.reduce((a, b) => a + b, 0) / buf.length;
}

// 骨格を描画（空色系で見やすく）
function drawKeypoints(ctx, keypoints) {
  const byName = Object.fromEntries(keypoints.map(k => [k.name, k]));
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#2A6EBB";
  ctx.fillStyle   = "#2A6EBB";
  keypoints.forEach((k) => {
    if (k.score != null && k.score > 0.3) {
      ctx.beginPath(); ctx.arc(k.x, k.y, 4, 0, 2*Math.PI); ctx.fill();
    }
  });
  LINE_PAIRS.forEach(([a,b])=>{
    const p1 = byName[a], p2 = byName[b];
    if (p1?.score>0.3 && p2?.score>0.3) {
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
  });
  ctx.restore();
}

export default function App() {
  const videoRef = useRef(null);
  const fileVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const rafRef = useRef(null);

  // 状態
  const [metrics, setMetrics] = useState({kneeL:true, kneeR:true, hipL:false, hipR:false, trunk:false});
  const [cycleNormalize, setCycleNormalize] = useState(true); // 「動きを1回分に揃えて比較（平均フォーム）」
  const [compareResult, setCompareResult] = useState(null);
  const [compareStats, setCompareStats]   = useState(null);
  const [compareRmse, setCompareRmse]     = useState({});
  const [refSamples, setRefSamples] = useState(null); // お手本
  const [cmpSamples, setCmpSamples] = useState(null); // 比較

  const [chartTick, setChartTick] = useState(0);
  const [useCamera, setUseCamera] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const runningRef = useRef(false);

  // スムージングバッファ
  const kneeLBufRef  = useRef([]);  const kneeRBufRef  = useRef([]);
  const hipLBufRef   = useRef([]);  const hipRBufRef   = useRef([]);
  const trunkBufRef  = useRef([]);

  // 記録
  const [recording, setRecording] = useState(false);
  const recordingRef = useRef(false);
  useEffect(() => { recordingRef.current = recording; }, [recording]);

  const samplesRef = useRef([]); // {t, kneeL,kneeR,hipL,hipR,trunk,dKnee,dHip}
  const startTimeRef = useRef(0);
  const lastSampleTimeRef = useRef(0);
  const SAMPLE_INTERVAL_MS = 100; // 10Hz

  // TFJS + Detector init
  useEffect(() => {
    (async () => {
      await import("@tensorflow/tfjs-backend-webgl");
      await tf.setBackend("webgl");
      await tf.ready();
      detectorRef.current = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        { modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );
      console.log("✅ Detector ready");
    })();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 共通ボタンスタイル（SoraLab 空色）
  const buttonStyle = {
    background:"#6DBFF2",
    color:"#fff",
    border:"none",
    borderRadius:"8px",
    padding:"8px 16px",
    cursor:"pointer",
    fontSize:"14px",
    margin:"2px",
    boxShadow:"0 2px 4px rgba(0,0,0,0.1)",
    transition:"background 0.2s"
  };

  // カメラ開始
  const startCamera = async () => {
    setUseCamera(true);
    const v = videoRef.current;

    const tryGet = async (constraints) => {
      try { return await navigator.mediaDevices.getUserMedia(constraints); }
      catch(e){ console.warn("getUserMedia failed:", e?.name||e); return null; }
    };

    let stream = await tryGet({ video:{ facingMode:{ideal:"environment"}, width:{ideal:960}, height:{ideal:540} }, audio:false });
    if (!stream) stream = await tryGet({ video:{ facingMode:{ideal:"user"},        width:{ideal:960}, height:{ideal:540} }, audio:false });
    if (!stream) stream = await tryGet({ video:true, audio:false });

    if (!stream) { alert("カメラにアクセスできませんでした。まずは『動画ファイル読込』で確認してください。"); return; }

    v.srcObject = stream;
    v.onloadedmetadata = async () => { await v.play(); setPlaying(true); startLoop(v); };
  };

  // 動画ファイル読み込み
  const loadFile = async (e) => {
    setUseCamera(false);
    const file = e.target.files?.[0];
    if (!file) return;
    const v = fileVideoRef.current;
    v.src = URL.createObjectURL(file);
    v.muted = true;
    v.playsInline = true;
    v.playbackRate = speed;
    v.onloadedmetadata = async () => { await v.play(); setPlaying(true); startLoop(v); };
  };

  // 停止
  const stop = () => {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    setPlaying(false);
    const v = videoRef.current;
    if (v?.srcObject) { v.srcObject.getTracks().forEach(t=>t.stop()); v.srcObject = null; }
  };

  // 推論ループ開始
  const startLoop = (videoEl) => {
    if (!detectorRef.current) return;
    runningRef.current = true;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const render = async () => {
      if (!runningRef.current) return;

      if (videoEl.readyState < 2) {
        rafRef.current = requestAnimationFrame(render);
        return;
      }

      // キャンバスを動画に合わせる
      if (canvas.width !== videoEl.videoWidth || canvas.height !== videoEl.videoHeight) {
        canvas.width = videoEl.videoWidth || 960;
        canvas.height = videoEl.videoHeight || 540;
      }

      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

      try {
        const poses = await detectorRef.current.estimatePoses(videoEl, { maxPoses: 1, flipHorizontal: false });
        if (poses[0]?.keypoints?.length) {
          // 骨格描画
          drawKeypoints(ctx, poses[0].keypoints);

          // 角度算出
          const kp = Object.fromEntries(poses[0].keypoints.map(k => [k.name, k]));
          const LHIP = kp["left_hip"],  LKN = kp["left_knee"],  LAN = kp["left_ankle"];
          const RHIP = kp["right_hip"], RKN = kp["right_knee"], RAN = kp["right_ankle"];
          const LSH  = kp["left_shoulder"],  RSH = kp["right_shoulder"];

          if ([LHIP,LKN,LAN,RHIP,RKN,RAN,LSH,RSH].every(p => p?.score > 0.3)) {
            const shoulderMid = mid(LSH, RSH);
            const hipMid      = mid(LHIP, RHIP);

            const kneeL = angle(LHIP, LKN, LAN);
            const kneeR = angle(RHIP, RKN, RAN);
            const hipL  = angle(shoulderMid, LHIP, LKN);
            const hipR  = angle(shoulderMid, RHIP, RKN);
            const trunk = angle(shoulderMid, hipMid, {x:hipMid.x, y:hipMid.y-100});

            const kneeLSm = movingAvg(kneeLBufRef.current, kneeL, 5);
            const kneeRSm = movingAvg(kneeRBufRef.current, kneeR, 5);
            const hipLSm  = movingAvg(hipLBufRef.current,  hipL,  5);
            const hipRSm  = movingAvg(hipRBufRef.current,  hipR,  5);
            const trunkSm = movingAvg(trunkBufRef.current, trunk, 5);

            const dKnee = (kneeLSm!=null && kneeRSm!=null) ? Math.abs(kneeLSm - kneeRSm) : null;
            const dHip  = (hipLSm!=null  && hipRSm!=null ) ? Math.abs(hipLSm  - hipRSm ) : null;

            // HUD（明るめ背景に馴染む淡色）
            ctx.save();
            ctx.fillStyle = "rgba(255,255,255,0.90)";
            ctx.strokeStyle = "rgba(42,110,187,0.15)";
            ctx.lineWidth = 1;
            ctx.fillRect(10, 10, 320, 112);
            ctx.strokeRect(10, 10, 320, 112);

            ctx.fillStyle = "#2A2A2A";
            ctx.font = "16px system-ui, sans-serif";
            const f = (v)=> v==null ? "-" : v.toFixed(1);
            ctx.fillText(`左膝: ${f(kneeLSm)}°   右膝: ${f(kneeRSm)}°   差: ${f(dKnee)}°`, 20, 38);
            ctx.fillText(`左股: ${f(hipLSm)}°    右股: ${f(hipRSm)}°    差: ${f(dHip)}°`,   20, 62);
            ctx.fillText(`体幹前傾: ${f(trunkSm)}°`, 20, 86);
            ctx.restore();

            // 記録（10Hz）
            if (recordingRef.current) {
              const now = performance.now();
              if (now - lastSampleTimeRef.current >= SAMPLE_INTERVAL_MS) {
                const t = (now - startTimeRef.current) / 1000; // sec
                samplesRef.current.push({
                  t: +t.toFixed(2),
                  kneeL: kneeLSm, kneeR: kneeRSm,
                  hipL:  hipLSm,  hipR:  hipRSm,
                  trunk: trunkSm,
                  dKnee, dHip,
                });
                lastSampleTimeRef.current = now;
                setChartTick(n => n + 1); // グラフ更新
              }
            }
          }
        }
      } catch (e) {
        console.warn("estimatePoses error:", e?.message || e);
      }

      rafRef.current = requestAnimationFrame(render);
    };
    render();
  };

  // 再生コントロール（アップロード動画向け）
  const playPause = () => {
    const v = fileVideoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else { v.pause(); setPlaying(false); }
  };
  const replay = () => {
    const v = fileVideoRef.current;
    if (!v) return;
    v.currentTime = 0;
    v.play();
    setPlaying(true);
  };
  const changeSpeed = (s) => {
    setSpeed(s);
    const v = fileVideoRef.current;
    if (v) v.playbackRate = s;
  };

  // 今の samplesRef.current を保存
  const saveCurrentAs = (role) => {
    if (!samplesRef.current.length) {
      alert("記録データがありません。先に『記録開始 → 停止』してください。");
      return;
    }
    const copy = samplesRef.current.map(s => ({...s}));
    if (role === "ref") setRefSamples(copy);
    if (role === "cmp") setCmpSamples(copy);
  };

  // 記録開始/停止/クリア/CSV
  const toggleRecord = () => {
    setRecording((r) => {
      const next = !r;
      if (next) {
        samplesRef.current = [];
        startTimeRef.current = performance.now();
        lastSampleTimeRef.current = 0;
        setChartTick((n) => n + 1);
      }
      return next;
    });
  };
  const clearRecord = () => {
    samplesRef.current = [];
    startTimeRef.current = 0;
    lastSampleTimeRef.current = 0;
    setChartTick(n => n+1);
  };
  const downloadCSV = () => {
    const rows = [["t(s)","kneeL","kneeR","hipL","hipR","trunk","dKnee","dHip"]];
    for (const s of samplesRef.current) {
      rows.push([
        s.t,
        n3(s.kneeL), n3(s.kneeR),
        n3(s.hipL),  n3(s.hipR),
        n3(s.trunk), n3(s.dKnee), n3(s.dHip)
      ]);
    }
    const csv = rows.map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `soralab_form_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };
  const n3 = (v)=> v==null ? "" : v.toFixed(3);

  // ★ 実況グラフ（色分け）
  const chartData = useMemo(() => {
    const s = samplesRef.current;
    return {
      labels: s.map(x => x.t),
      datasets: [
        { label: "左膝角度 (°)",   data: s.map(x => x.kneeL ?? null), borderWidth: 2, pointRadius: 0, borderColor:"#2A6EBB" },
        { label: "右膝角度 (°)",   data: s.map(x => x.kneeR ?? null), borderWidth: 2, pointRadius: 0, borderColor:"#00A8E8" },
        { label: "左股関節角度 (°)", data: s.map(x => x.hipL  ?? null), borderWidth: 2, pointRadius: 0, borderColor:"#7CC5EB" },
        { label: "右股関節角度 (°)", data: s.map(x => x.hipR  ?? null), borderWidth: 2, pointRadius: 0, borderColor:"#8FD3FF" },
        { label: "体幹前傾 (°)",   data: s.map(x => x.trunk ?? null),  borderWidth: 2, pointRadius: 0, borderColor:"#4F9FD8" },
      ],
    };
  }, [chartTick]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { title: { display: true, text: "時間 (秒)" } },
      y: { title: { display: true, text: "角度 (°)" } },
    },
    plugins: { legend: { position: "top" } },
  };

  // 比較ロジック
  const runCompareMulti = useCallback(() => {
    if (!refSamples || !cmpSamples) return;

    const metricsList = Object.keys(metrics).filter(k => metrics[k]);
    const res = { labels: [], datasets: [] };
    const rmseRes = {};
    const stats = { mode: cycleNormalize ? "cycle" : "time" };

    if (cycleNormalize) {
      // 「動きを1回分に揃えて、平均したフォームで比較」
      const refT = refSamples.map(s=>s.t), cmpT = cmpSamples.map(s=>s.t);

      for (const key of metricsList) {
        const refYraw = refSamples.map(s=>s[key] ?? null).filter(v=>v!=null);
        const cmpYraw = cmpSamples.map(s=>s[key] ?? null).filter(v=>v!=null);
        if (!refYraw.length || !cmpYraw.length) continue;

        const refPeaks = findLocalMinima(refT, refYraw, { prominence: 5, minGapSec: 0.30 });
        const cmpPeaks = findLocalMinima(cmpT, cmpYraw, { prominence: 5, minGapSec: 0.30 });

        const refC = cyclesNormalize(refT, refYraw, refPeaks);
        const cmpC = cyclesNormalize(cmpT, cmpYraw, cmpPeaks);
        if (!(refC.length && cmpC.length)) continue;

        const N = refC[0].normT.length;
        const avgRef = Array(N).fill(0), avgCmp = Array(N).fill(0);
        for (const c of refC) c.normV.forEach((v,i)=> avgRef[i] += v/refC.length);
        for (const c of cmpC) c.normV.forEach((v,i)=> avgCmp[i] += v/cmpC.length);

        rmseRes[key] = rmse(avgRef, avgCmp);

        res.labels = refC[0].normT.map(x => (x*100).toFixed(0));
        // 色割り当て
        const color = metricColor(key);
        res.datasets.push({ label:`お手本:${labelJP(key)}`, data:avgRef, borderWidth:2, pointRadius:0, borderColor:color });
        res.datasets.push({ label:`比較:${labelJP(key)}`,   data:avgCmp, borderWidth:2, pointRadius:0, borderColor:color, borderDash:[6,4] });

        // サイクル統計（揃えた後の各区間の秒数）
        stats.ref = {
          count: refC.length,
          avg:  avg(refC.map(c=>c.dur)),
          sd:   stdev(refC.map(c=>c.dur)),
          min:  Math.min(...refC.map(c=>c.dur)),
          max:  Math.max(...refC.map(c=>c.dur)),
          cadence: 60 / avg(refC.map(c=>c.dur))
        };
        stats.cmp = {
          count: cmpC.length,
          avg:  avg(cmpC.map(c=>c.dur)),
          sd:   stdev(cmpC.map(c=>c.dur)),
          min:  Math.min(...cmpC.map(c=>c.dur)),
          max:  Math.max(...cmpC.map(c=>c.dur)),
          cadence: 60 / avg(cmpC.map(c=>c.dur))
        };
      }
    } else {
      // 時間ベースで比較（ref の時刻に cmp を補間）
      const refT = refSamples.map(s=>s.t);
      const cmpT = cmpSamples.map(s=>s.t);
      res.labels = refT.map(t => t.toFixed(2));

      for (const key of metricsList) {
        const refY   = refSamples.map(s => s[key] ?? null);
        const cmpYraw= cmpSamples.map(s => s[key] ?? null);
        const cmpYseries = fillNaLinear(cmpT, cmpYraw);
        const cmpY = refT.map(t => linInterp(t, cmpT, cmpYseries));

        rmseRes[key] = rmse(
          refY.filter(v => v != null),
          cmpY.filter(v => v != null)
        );

        const color = metricColor(key);
        res.datasets.push({ label:`お手本:${labelJP(key)}`, data: refY, borderWidth:2, pointRadius:0, borderColor:color });
        res.datasets.push({ label:`比較:${labelJP(key)}`,   data: cmpY, borderWidth:2, pointRadius:0, borderColor:color, borderDash:[6,4] });
      }
    }

    if (!res.datasets.length) {
      setCompareResult(null);
      setCompareStats(null);
      alert("比較に必要なデータが得られませんでした。記録時間を少し長くするか、指標を減らして再試行してください。");
      return;
    }
    setCompareRmse(rmseRes);
    setCompareResult({ chartData: res });
    setCompareStats(stats);
  }, [refSamples, cmpSamples, metrics, cycleNormalize]);

  const labelJP = (key) => ({
    kneeL:"左膝", kneeR:"右膝", hipL:"左股", hipR:"右股", trunk:"体幹前傾"
  }[key] || key);

  const metricColor = (key) => ({
    kneeL:"#2A6EBB",
    kneeR:"#00A8E8",
    hipL:"#7CC5EB",
    hipR:"#8FD3FF",
    trunk:"#4F9FD8"
  }[key] || "#6DBFF2");

  // ------------------- UI -------------------
  return (
    <div style={{
      fontFamily:"'Segoe UI','Hiragino Sans',sans-serif",
      padding:16,
      background:"linear-gradient(to bottom, #E6F6FF, #FFFFFF)",
      minHeight:"100vh"
    }}>
      <h1 style={{ color:"#2A6EBB", textAlign:"center", marginBottom:20 }}>
        ☁️ SORA LAB フォーム可視化
      </h1>

      {/* カメラ/動画読込 */}
      <div style={{ display:"flex", gap:12, flexWrap:"wrap", alignItems:"center" }}>
        <button style={buttonStyle} onClick={startCamera} disabled={useCamera}>カメラ開始</button>
        <label style={{ ...buttonStyle, cursor:"pointer" }}>
          動画ファイル読込
          <input type="file" accept="video/*" onChange={loadFile} style={{ display:"none" }} />
        </label>
        <button style={buttonStyle} onClick={stop}>停止</button>
      </div>

      {/* アップロード動画の再生コントロール */}
      {!useCamera && (
        <div style={{ marginTop:10, display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
          <button style={buttonStyle} onClick={playPause}>{playing ? "⏸ 一時停止" : "▶ 再生"}</button>
          <button style={buttonStyle} onClick={replay}>⟲ リプレイ</button>
          <span>速度:</span>
          {[0.25, 0.5, 0.75, 1].map(s => (
           <button
             key={s}
             onClick={() => changeSpeed(s)}
             style={{
               marginRight: 4,
               padding: "4px 10px",
               borderRadius: 6,
               border: "1px solid #6DBFF2",
               cursor: "pointer",
               background: speed === s ? "#2A6EBB" : "#6DBFF2", // 選択中は濃い空色
               color: "white",
               fontWeight: speed === s ? "bold" : "normal"
             }}
           >
             {s}x
           </button>
         ))}
        </div>
      )}

      {/* 記録系UI */}
      <div style={{ marginTop:10, display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
        <button style={{...buttonStyle, fontWeight:700}} onClick={toggleRecord}>
          {recording ? "■ 記録停止" : "● 記録開始"}
        </button>
        <button style={buttonStyle} onClick={clearRecord} disabled={!samplesRef.current.length}>記録クリア</button>
        <button style={buttonStyle} onClick={downloadCSV} disabled={!samplesRef.current.length}>CSVダウンロード</button>
        <span style={{ color:"#333" }}>サンプル数: {samplesRef.current.length}</span>
      </div>

      {/* 保存 */}
      <div style={{ marginTop:10, display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
        <button style={buttonStyle} onClick={()=>saveCurrentAs("ref")} disabled={!samplesRef.current.length}>この記録を「お手本」に保存</button>
        <button style={buttonStyle} onClick={()=>saveCurrentAs("cmp")} disabled={!samplesRef.current.length}>この記録を「比較」に保存</button>
        <span style={{marginLeft:8, color:"#333"}}>
          保存状況：お手本 {refSamples? "✅": "❌"} / 比較 {cmpSamples? "✅": "❌"}
        </span>
      </div>

      {/* 比較パネル */}
      <div style={{
        marginTop:12, padding:12,
        border:"1px solid #CCE7F5",
        borderRadius:12,
        background:"#fff",
        boxShadow:"0 2px 6px rgba(0,0,0,0.05)"
      }}>
        {/* 指標選択 & オプション */}
        <div style={{display:'flex', gap:12, flexWrap:'wrap', alignItems:'center'}}>
          {[{key:'kneeL',label:'左膝'},{key:'kneeR',label:'右膝'},{key:'hipL',label:'左股'},{key:'hipR',label:'右股'},{key:'trunk',label:'体幹前傾'}]
            .map(m=>(
              <label key={m.key}><input type="checkbox" checked={metrics[m.key]??true}
              onChange={e=>setMetrics(v=>({...v,[m.key]:e.target.checked}))}/> {m.label}</label>
          ))}
          <label style={{marginLeft:8}}>
            <input type="checkbox" checked={cycleNormalize} onChange={e=>setCycleNormalize(e.target.checked)}/>
            動きを1回分に揃えて比較（平均フォーム）
          </label>
          <button style={buttonStyle} onClick={runCompareMulti} disabled={!refSamples||!cmpSamples}>比較（グラフ）</button>

          {compareResult && (
            <span style={{marginLeft:8}}>
              {Object.entries(compareRmse).map(([k,v])=>(
                <span key={k} style={{marginRight:10}}>{labelJP(k)}: RMSE {v?.toFixed(2)}°</span>
              ))}
            </span>
          )}
        </div>

        {/* サイクル統計（正規化ONのとき表示） */}
        {compareStats?.mode === 'cycle' && (
          <div style={{marginTop:6}}>
            <table style={{fontSize:14}}>
              <thead>
                <tr>
                  <th></th><th>サイクル数</th><th>平均(s)</th><th>SD</th>
                  <th>最短</th><th>最長</th><th>ケイデンス(回/分)</th>
                </tr>
              </thead>
              <tbody>
                {['ref','cmp'].map(tag=>{
                  const s=compareStats[tag]; if(!s) return null;
                  return (
                    <tr key={tag}>
                      <td>{tag==='ref'?'お手本':'比較'}</td>
                      <td>{s.count}</td><td>{s.avg.toFixed(2)}</td><td>{s.sd.toFixed(2)}</td>
                      <td>{s.min.toFixed(2)}</td><td>{s.max.toFixed(2)}</td><td>{s.cadence.toFixed(1)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* 比較グラフ */}
        {compareResult && (
          <div style={{
            height: 280, marginTop: 8, background:"#FFFFFF",
            border:"1px solid #CCE7F5", borderRadius:12, padding:8,
            boxShadow:"0 2px 6px rgba(0,0,0,0.05)"
          }}>
            <Line
              data={compareResult.chartData}
              options={{
                responsive:true, maintainAspectRatio:false, animation:false,
                scales:{
                  x:{ title:{display:true, text: cycleNormalize ? 'サイクル(%)' : '時間(秒)'} },
                  y:{ title:{display:true, text:'角度(°)'} }
                },
                plugins:{ legend:{ position:'top' } }
              }}
            />
          </div>
        )}
      </div>

      {/* 隠しvideo */}
      <video ref={videoRef} playsInline muted style={{ display:"none" }} />
      <video ref={fileVideoRef} controls playsInline muted style={{ display:"none" }} />

      {/* 右側プレビュー（動画＋骨格） */}
      <div style={{ marginTop:12 }}>
        <canvas ref={canvasRef} style={{
          width:"100%", maxWidth:960, background:"#fff", borderRadius:12,
          boxShadow:"0 2px 6px rgba(0,0,0,0.05)"
        }} />
      </div>

      {/* ライブグラフ */}
      <div style={{
        height: 260, marginTop: 12, background:"#FFFFFF",
        border:"1px solid #CCE7F5", borderRadius:12, padding:8,
        boxShadow:"0 2px 6px rgba(0,0,0,0.05)"
      }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* 補足 */}
      <p style={{ marginTop:20, color:"#555", textAlign:"center" }}>
        🌸 コツ：横から全身が入るように撮影すると、より正確に分析できます。<br/>
        そっと寄り添う可視化で、あなたのフォームを応援します。
      </p>

      {/* ▼ 指標の説明を追加 ▼ */}
      <div style={{ marginTop:20, padding:12, background:"#f9f9f9", border:"1px solid #eee", borderRadius:8, fontSize:14, lineHeight:1.6 }}>
        <h3 style={{ marginTop:0, fontSize:16, color:"#333" }}>📘 数値の意味（ソララボ式）</h3>
        <ul style={{ paddingLeft:18, margin:0 }}>
          <li><b>1歩のリズム（平均秒数）</b>：1歩にかかる時間の目安です。</li>
          <li><b>リズムの安定度（SD）</b>：数字が小さいほど、動きが揃っていて安定しています。</li>
          <li><b>いちばん速い動き（最短）</b>：最も速く脚が動いたときのリズムです。</li>
          <li><b>いちばんゆっくりの動き（最長）</b>：最もゆっくりだったときのリズムです。</li>
          <li><b>テンポ（ケイデンス）</b>：1分あたりの歩数。音楽のBPMのように走るテンポを表します。</li>
        </ul>
      </div>

      {/* ▼ Stripe 決済ボタンを追加 ▼ */}
      <button 
        style={{
          marginTop: 20,
          padding: "10px 20px",
          background: "#2A6EBB",
          color: "white",
          border: "none",
          borderRadius: 8,
          fontSize: 16,
          cursor: "pointer",
          fontWeight: "bold",
          boxShadow: "0 2px 6px rgba(0,0,0,0.15)"
        }}
        onClick={goToCheckout}
      >
        🌟 SORA LAB プレミアム（月額プラン）に登録する
      </button>
    </div>
  );
}

// ---------- 補助関数（下にまとめて定義） ----------

// 線形補間
function linInterp(x, xp, yp) {
  if (!xp || !yp || xp.length === 0 || yp.length === 0) return null;
  if (x <= xp[0]) return yp[0];
  if (x >= xp[xp.length - 1]) return yp[yp.length - 1];
  let i = 1;
  while (i < xp.length && xp[i] < x) i++;
  const x0 = xp[i - 1], x1 = xp[i];
  const y0 = yp[i - 1], y1 = yp[i];
  if (x1 === x0) return y0;
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

// null を簡易補間（端は最近傍、内部は線形）
function fillNaLinear(times, values) {
  const y = values.slice();
  const n = y.length;
  if (n === 0) return y;

  let i = 0;
  while (i < n && (y[i] == null || !isFinite(y[i]))) i++;
  if (i > 0 && i < n) for (let k = 0; k < i; k++) y[k] = y[i];

  let j = n - 1;
  while (j >= 0 && (y[j] == null || !isFinite(y[j]))) j--;
  if (j >= 0 && j < n - 1) for (let k = j + 1; k < n; k++) y[k] = y[j];

  let a = 0;
  while (a < n) {
    if (y[a] == null || !isFinite(y[a])) {
      let b = a;
      while (b < n && (y[b] == null || !isFinite(y[b]))) b++;
      if (a > 0 && b < n) {
        const y0 = y[a - 1], y1 = y[b];
        const x0 = times[a - 1], x1 = times[b];
        const dx = (x1 - x0) || 1e-9;
        for (let k = a; k < b; k++) y[k] = y0 + (y1 - y0) * (times[k] - x0) / dx;
      }
      a = b;
    } else {
      a++;
    }
  }
  return y;
}

// 極小値（谷）の検出
function findLocalMinima(times, values, { prominence = 8, minGapSec = 0.35 } = {}) {
  const idxs = [];
  for (let i = 1; i < values.length - 1; i++) {
    if (values[i] <= values[i - 1] && values[i] <= values[i + 1]) idxs.push(i);
  }
  const kept = [];
  let lastKeepT = -1e12;
  for (const i of idxs) {
    const left = Math.max(0, i - 10);
    const right = Math.min(values.length - 1, i + 10);
    const leftMax = Math.max(...values.slice(left, i));
    const rightMax = Math.max(...values.slice(i + 1, right + 1));
    const prom = Math.min(leftMax - values[i], rightMax - values[i]);
    if (prom >= prominence && (times[i] - lastKeepT) >= minGapSec) {
      kept.push(i);
      lastKeepT = times[i];
    }
  }
  return kept;
}

// サイクル正規化(0-100%)
function cyclesNormalize(times, values, peaks, N = 100) {
  const cycles = [];
  if (!peaks || peaks.length < 2) return cycles;
  for (let c = 0; c < peaks.length - 1; c++) {
    const i0 = peaks[c], i1 = peaks[c + 1];
    const t0 = times[i0], t1 = times[i1];
    if (t1 <= t0) continue;
    const normT = Array.from({ length: N }, (_, i) => i / (N - 1));
    const normV = normT.map(frac => {
      const targetT = t0 + frac * (t1 - t0);
      return linInterp(targetT, times, values);
    });
    cycles.push({ normT, normV, dur: t1 - t0 });
  }
  return cycles;
}

// RMSE
function rmse(arr1, arr2) {
  const n = Math.min(arr1.length, arr2.length);
  if (n === 0) return null;
  let s = 0, c = 0;
  for (let i = 0; i < n; i++) {
    const a = arr1[i], b = arr2[i];
    if (a == null || b == null || !isFinite(a) || !isFinite(b)) continue;
    const d = a - b;
    s += d * d;
    c++;
  }
  return c ? Math.sqrt(s / c) : null;
}

function avg(arr) {
  const v = arr.filter(x => x != null && isFinite(x));
  if (!v.length) return 0;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function stdev(arr) {
  const v = arr.filter(x => x != null && isFinite(x));
  if (v.length <= 1) return 0;
  const m = avg(v);
  const s2 = v.reduce((acc, x) => acc + (x - m) ** 2, 0) / v.length;
  return Math.sqrt(s2);
}
