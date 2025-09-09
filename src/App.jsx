import { useEffect, useRef, useState, useMemo } from "react";
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

// 骨格を描く「だけ」
function drawKeypoints(ctx, keypoints) {
  const byName = Object.fromEntries(keypoints.map(k => [k.name, k]));
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,0,0,0.9)";
  ctx.fillStyle   = "rgba(0,0,0,0.9)";
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

  // 記録データの保存先（お手本 / 比較）
  const [refSamples, setRefSamples] = useState(null); // お手本
  const [cmpSamples, setCmpSamples] = useState(null); // 比較
  const [compareSide, setCompareSide] = useState("left"); // "left" or "right"
  const [useMinPeak, setUseMinPeak] = useState(true);     // 最小値ピークで同期
  const [compareResult, setCompareResult] = useState(null); // {shift, rmse, chartData}

  const [chartTick, setChartTick] = useState(0);
  const [useCamera, setUseCamera] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const runningRef = useRef(false);

  // スムージング用
  const kneeLBufRef  = useRef([]);  const kneeRBufRef  = useRef([]);
  const hipLBufRef   = useRef([]);  const hipRBufRef   = useRef([]);
  const trunkBufRef  = useRef([]);

  // ★ 記録（時系列）関連
  const [recording, setRecording] = useState(false);
  const recordingRef = useRef(false);                       // ← 名前を recordingRef に
  useEffect(() => { recordingRef.current = recording; }, [recording]);  // ← 正しい同期

  const samplesRef = useRef([]); // {t, kneeL,kneeR,hipL,hipR,trunk,dKnee,dHip}
  const startTimeRef = useRef(0);
  const lastSampleTimeRef = useRef(0);
  const SAMPLE_INTERVAL_MS = 100; // 10Hzで記録

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

  const stop = () => {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    setPlaying(false);
    const v = videoRef.current;
    if (v?.srcObject) { v.srcObject.getTracks().forEach(t=>t.stop()); v.srcObject = null; }
  };

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

      if (canvas.width !== videoEl.videoWidth || canvas.height !== videoEl.videoHeight) {
        canvas.width = videoEl.videoWidth || 960;
        canvas.height = videoEl.videoHeight || 540;
      }

      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

      try {
        const poses = await detectorRef.current.estimatePoses(videoEl, { maxPoses: 1, flipHorizontal: false });
        if (poses[0]?.keypoints?.length) {
          drawKeypoints(ctx, poses[0].keypoints);

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

            // HUD
            ctx.save();
            ctx.fillStyle = "rgba(255,255,255,0.88)";
            ctx.fillRect(10, 10, 300, 112);
            ctx.fillStyle = "#111";
            ctx.font = "16px system-ui, sans-serif";
            const f = (v)=> v==null ? "-" : v.toFixed(1);
            ctx.fillText(`左膝: ${f(kneeLSm)}°   右膝: ${f(kneeRSm)}°   差: ${f(dKnee)}°`, 20, 36);
            ctx.fillText(`左股: ${f(hipLSm)}°    右股: ${f(hipRSm)}°    差: ${f(dHip)}°`,   20, 58);
            ctx.fillText(`体幹前傾: ${f(trunkSm)}°`, 20, 82);
            ctx.restore();

            // ★ 記録（10Hz）
            // 置き換え（記録部分）
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
                  lastSampleTimeRef.current = now;     // ← now を保存
                  setChartTick(n => n + 1);            // 再描画
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

  // ----- 再生コントロール（アップロード動画向け） -----
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

// 今の samplesRef.current をディープコピーして保存
const saveCurrentAs = (role) => {
  if (!samplesRef.current.length) {
    alert("記録データがありません。先に『記録開始 → 停止』してください。");
    return;
  }
  const copy = samplesRef.current.map(s => ({...s}));
  if (role === "ref") setRefSamples(copy);
  if (role === "cmp") setCmpSamples(copy);
};

  // ★ 記録の開始/停止/クリア/CSV保存
  // 置き換え
  const toggleRecord = () => {
  setRecording((r) => {
    const next = !r;
    if (next) {
      // ← 記録を始める瞬間に一度だけ初期化
      samplesRef.current = [];
      startTimeRef.current = performance.now();
      lastSampleTimeRef.current = 0;
      setChartTick((n) => n + 1); // 表示も初期化
    }
    return next;
  });
};

  const clearRecord = () => { samplesRef.current = []; startTimeRef.current = 0; lastSampleTimeRef.current = 0;
    setChartTick(n => n+1);
   };
  const downloadCSV = () => {
    const rows = [
      ["t(s)","kneeL","kneeR","hipL","hipR","trunk","dKnee","dHip"]
    ];
    for (const s of samplesRef.current) {
      rows.push([
        s.t,
        n(s.kneeL), n(s.kneeR),
        n(s.hipL),  n(s.hipR),
        n(s.trunk), n(s.dKnee), n(s.dHip)
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
  const n = (v)=> v==null ? "" : v.toFixed(3);

// ユーティリティ（すでにあるなら流用OK）
const interp1d = (tx, yx, grid) => {
  const out = new Array(grid.length);
  for (let i = 0, j = 0; i < grid.length; i++) {
    const g = grid[i];
    while (j+1 < tx.length && tx[j+1] < g) j++;
    if (j+1 >= tx.length) { out[i] = yx[tx.length-1]; continue; }
    const t0 = tx[j], t1 = tx[j+1];
    const y0 = yx[j], y1 = yx[j+1];
    const r = (g - t0) / Math.max(1e-9, (t1 - t0));
    out[i] = y0 + (y1 - y0) * r;
  }
  return out;
};
const rmseOnOverlap = (t1, y1, t2s, y2) => {
  const tmin = Math.max(t1[0], t2s[0]);
  const tmax = Math.min(t1[t1.length-1], t2s[t2s.length-1]);
  if (!(tmax > tmin)) return NaN;
  const N = 200;
  const grid = Array.from({length:N}, (_,i)=> tmin + (tmax-tmin)*i/(N-1));
  const a = interp1d(t1, y1, grid);
  const b = interp1d(t2s, y2, grid);
  let s = 0; for (let i=0;i<N;i++){ const d=a[i]-b[i]; s+=d*d; }
  return Math.sqrt(s/N);
};
// 最小 or 最大の“ピーク”インデックス（簡易）
const findPeakIndex = (y, useMin=true) => {
  let best=0, val=y[0];
  for (let i=1;i<y.length;i++){
    if (useMin ? y[i] < val : y[i] > val) { val=y[i]; best=i; }
  }
  return best;
};

const runCompare = () => {
  if (!refSamples || !cmpSamples) {
    alert("お手本と比較、両方の記録を保存してください。");
    return;
  }
  const sideKey = compareSide === "left" ? {k:"kneeL", label:"kneeL"} : {k:"kneeR", label:"kneeR"};

  const t1 = refSamples.map(s => s.t);
  const y1 = refSamples.map(s => s[sideKey.k]);
  const t2 = cmpSamples.map(s => s.t);
  const y2 = cmpSamples.map(s => s[sideKey.k]);

  if (!y1.length || !y2.length) { alert("角度データが不足しています。"); return; }

  const i1 = findPeakIndex(y1, useMinPeak);
  const i2 = findPeakIndex(y2, useMinPeak);
  const shift = t1[i1] - t2[i2];      // 比較データを +shift で合わせる
  const t2s = t2.map(v => v + shift);

  // 表示都合：実測（お手本）の時刻上に比較データを補間して重ねる
  const y2_on_t1 = interp1d(t2s, y2, t1);
  const rmse = rmseOnOverlap(t1, y1, t2s, y2);

  setCompareResult({
    shift, rmse,
    chartData: {
      labels: t1,
      datasets: [
        { label:`お手本: ${sideKey.label}`, data:y1, borderWidth:2, pointRadius:0 },
        { label:`比較(shift済): ${sideKey.label}`, data:y2_on_t1, borderWidth:2, borderDash:[6,4], pointRadius:0 },
      ]
    },
    peakX: t1[i1],
  });
};

  // ★ グラフ用データ（差分中心）
  const chartData = useMemo(() => {
  const s = samplesRef.current;
  return {
    labels: s.map(x => x.t),
    datasets: [
      { label: "左膝角度 (°)",   data: s.map(x => x.kneeL ?? null), borderWidth: 2, pointRadius: 0 },
      { label: "右膝角度 (°)",   data: s.map(x => x.kneeR ?? null), borderWidth: 2, pointRadius: 0 },
      { label: "左股関節角度 (°)", data: s.map(x => x.hipL  ?? null), borderWidth: 2, pointRadius: 0 },
      { label: "右股関節角度 (°)", data: s.map(x => x.hipR  ?? null), borderWidth: 2, pointRadius: 0 },
      { label: "体幹前傾 (°)",   data: s.map(x => x.trunk ?? null),  borderWidth: 2, pointRadius: 0 },
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

  return (
    <div style={{ fontFamily:"system-ui, sans-serif", padding:16 }}>
      <h1>SORA LAB フォーム可視化（PoC）</h1>

      <div style={{ display:"flex", gap:12, flexWrap:"wrap", alignItems:"center" }}>
        <button onClick={startCamera} disabled={useCamera}>カメラ開始</button>
        <label style={{ border:"1px solid #ccc", padding:"8px 12px", cursor:"pointer" }}>
          動画ファイル読込
          <input type="file" accept="video/*" onChange={loadFile} style={{ display:"none" }} />
        </label>
        <button onClick={stop}>停止</button>
      </div>

      {/* アップロード動画の再生コントロール */}
      {!useCamera && (
        <div style={{ marginTop:10, display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
          <button onClick={playPause}>{playing ? "⏸ 一時停止" : "▶ 再生"}</button>
          <button onClick={replay}>⟲ リプレイ</button>
          <span>速度:</span>
          {[0.5, 1, 1.5, 2].map(s => (
            <button key={s} onClick={()=>changeSpeed(s)} disabled={speed===s}>{s}x</button>
          ))}
        </div>
      )}

      {/* ★ 記録系のUI */}
      <div style={{ marginTop:10, display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
        <button onClick={toggleRecord} style={{ fontWeight: 700 }}>
          {recording ? "■ 記録停止" : "● 記録開始"}
        </button>
        <button onClick={clearRecord} disabled={!samplesRef.current.length}>記録クリア</button>
        <button onClick={downloadCSV} disabled={!samplesRef.current.length}>CSVダウンロード</button>
        <span style={{ color:"#666" }}>
          サンプル数: {samplesRef.current.length}
        </span>
      </div>

      {/* ここを既存UIの下あたりに追加 */}
<div style={{ marginTop:10, display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
  <button onClick={()=>saveCurrentAs("ref")} disabled={!samplesRef.current.length}>この記録を「お手本」に保存</button>
  <button onClick={()=>saveCurrentAs("cmp")} disabled={!samplesRef.current.length}>この記録を「比較」に保存</button>

  <span style={{marginLeft:8, color:"#333"}}>
    保存状況：お手本 {refSamples? "✅": "❌"} / 比較 {cmpSamples? "✅": "❌"}
  </span>
</div>

{/* 比較パネル */}
<div style={{marginTop:12, padding:12, border:"1px solid #eee", borderRadius:8}}>
  <div style={{display:"flex", gap:12, alignItems:"center", flexWrap:"wrap"}}>
    <div>膝側：
      <select value={compareSide} onChange={e=>setCompareSide(e.target.value)}>
        <option value="left">左</option>
        <option value="right">右</option>
      </select>
    </div>
    <label>
      <input type="checkbox" checked={useMinPeak} onChange={e=>setUseMinPeak(e.target.checked)} />
      最小値ピークで同期（外すと最大）
    </label>
    <button onClick={runCompare} disabled={!refSamples || !cmpSamples}>比較（ピーク同期）</button>
    {compareResult && (
      <span style={{marginLeft:8}}>
        RMSE: {compareResult.rmse?.toFixed(2)}° / shift: {compareResult.shift?.toFixed(3)}s
      </span>
    )}
  </div>

  {compareResult && (
    <div style={{ height: 260, marginTop: 8, background:"#fafafa", border:"1px solid #eee", borderRadius:8, padding:8 }}>
      <Line
        data={compareResult.chartData}
        options={{
          responsive:true, maintainAspectRatio:false, animation:false,
          scales:{ x:{ title:{display:true, text:"時間 (秒)"}}, y:{ title:{display:true, text:"角度 (°)"}}},
          plugins:{ legend:{ position:"top" } }
        }}
      />
    </div>
  )}
</div>

      {/* 隠しvideo */}
      <video ref={videoRef} playsInline muted style={{ display:"none" }} />
      <video ref={fileVideoRef} controls playsInline muted style={{ display:"none" }} />

      <div style={{ marginTop:12 }}>
        <canvas ref={canvasRef} style={{ width:"100%", maxWidth:960, background:"#eee", borderRadius:8 }} />
      </div>

      {/* ★ グラフ領域 */}
      <div style={{ height: 260, marginTop: 12, background:"#fafafa", border:"1px solid #eee", borderRadius:8, padding:8 }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      <p style={{ marginTop:12, color:"#555" }}>
        コツ：横から全身が入るように撮影（30fps以上）。明るい場所で。
      </p>
    </div>
  );
}
