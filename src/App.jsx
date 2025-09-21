// App.jsx
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
ChartJS.defaults.color = "#222"; // 全体トーン

const LINE_PAIRS = [
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
  ["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"]
];

// 見やすいカラーパレット
const COLOR_MAP = {
  kneeL:  "#e53935", // 赤
  kneeR:  "#1e88e5", // 青
  hipL:   "#43a047", // 緑
  hipR:   "#fb8c00", // オレンジ
  trunk:  "#8e24aa", // 紫
};

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
  const [metrics, setMetrics] = useState({kneeL:true, kneeR:true, hipL:false, hipR:false, trunk:false});
  const [cycleNormalize, setCycleNormalize] = useState(true);
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

  // スムージング用
  const kneeLBufRef  = useRef([]);  const kneeRBufRef  = useRef([]);
  const hipLBufRef   = useRef([]);  const hipRBufRef   = useRef([]);
  const trunkBufRef  = useRef([]);

  // ★ 記録（時系列）関連
  const [recording, setRecording] = useState(false);
  const recordingRef = useRef(false);
  useEffect(() => { recordingRef.current = recording; }, [recording]);

  const samplesRef = useRef([]); // {t, kneeL,kneeR,hipL,hipR,trunk,dKnee,dHip}
  const startTimeRef = useRef(0);
  const lastSampleTimeRef = useRef(0);
  const SAMPLE_INTERVAL_MS = 100; // 10Hzで記録

  // 🧡 コーチ（生成コメント）関連
  const [autoCoach, setAutoCoach] = useState(true);
  const [coachNotes, setCoachNotes] = useState(null);
  const [coachLoading, setCoachLoading] = useState(false);
  const [coachError, setCoachError] = useState(null);

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
                setChartTick(n => n + 1);
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
  const toggleRecord = () => {
    setRecording((r) => {
      const next = !r;
      if (next) {
        // 記録を始める瞬間に一度だけ初期化
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

  // ★ 下段の記録グラフ（色付き）
  const chartData = useMemo(() => {
    const s = samplesRef.current;
    return {
      labels: s.map(x => x.t),
      datasets: [
        { label: "左膝角度 (°)",     data: s.map(x => x.kneeL ?? null), borderWidth: 2, pointRadius: 0,
          borderColor: COLOR_MAP.kneeL, backgroundColor: COLOR_MAP.kneeL },
        { label: "右膝角度 (°)",     data: s.map(x => x.kneeR ?? null), borderWidth: 2, pointRadius: 0,
          borderColor: COLOR_MAP.kneeR, backgroundColor: COLOR_MAP.kneeR },
        { label: "左股関節角度 (°)", data: s.map(x => x.hipL  ?? null), borderWidth: 2, pointRadius: 0,
          borderColor: COLOR_MAP.hipL,  backgroundColor: COLOR_MAP.hipL },
        { label: "右股関節角度 (°)", data: s.map(x => x.hipR  ?? null), borderWidth: 2, pointRadius: 0,
          borderColor: COLOR_MAP.hipR,  backgroundColor: COLOR_MAP.hipR },
        { label: "体幹前傾 (°)",     data: s.map(x => x.trunk ?? null),  borderWidth: 2, pointRadius: 0,
          borderColor: COLOR_MAP.trunk, backgroundColor: COLOR_MAP.trunk },
      ],
    };
  }, [chartTick]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { title: { display: true, text: "時間 (秒)" }, grid: { color: "#eee" }, ticks:{ color:"#333" } },
      y: { title: { display: true, text: "角度 (°)"   }, grid: { color: "#eee" }, ticks:{ color:"#333" } },
    },
    plugins: {
      legend: { position: "top", labels: { usePointStyle: true, boxWidth: 10 } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.formattedValue}°` } }
    },
  };

  // ---------- 比較ロジック（コールバック） ----------
  const runCompareMulti = useCallback(() => {
    if (!refSamples || !cmpSamples) return;

    const metricsList = Object.keys(metrics).filter(k => metrics[k]);
    const res = { labels: [], datasets: [] };
    const rmseRes = {};
    const stats = { mode: cycleNormalize ? "cycle" : "time" };

    if (cycleNormalize) {
      // サイクル正規化（複数サイクル→平均波形）
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
        const col = COLOR_MAP[key] || "#666";
        res.datasets.push({
          label:`お手本:${key}`, data:avgRef, borderWidth:2.5, pointRadius:0,
          borderColor: col, backgroundColor: col
        });
        res.datasets.push({
          label:`比較:${key}`,   data:avgCmp, borderWidth:2.5, pointRadius:0,
          borderColor: col, backgroundColor: col, borderDash:[6,4]
        });

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
      // 時系列そのまま比較（ref の時刻に cmp を補間）
      const refT = refSamples.map(s=>s.t);
      const cmpT = cmpSamples.map(s=>s.t);
      res.labels = refT.map(t => t.toFixed(2));

      for (const key of metricsList) {
        const refY   = refSamples.map(s => s[key] ?? null);
        const cmpYraw= cmpSamples.map(s => s[key] ?? null);

        const cmpYseries = fillNaLinear(cmpT, cmpYraw);            // 欠損軽補間
        const cmpY = refT.map(t => linInterp(t, cmpT, cmpYseries)); // 時間合わせ

        rmseRes[key] = rmse(
          refY.filter(v => v != null),
          cmpY.filter(v => v != null)
        );

        const col = COLOR_MAP[key] || "#666";
        res.datasets.push({
          label:`お手本:${key}`, data: refY, borderWidth:2.5, pointRadius:0,
          borderColor: col, backgroundColor: col
        });
        res.datasets.push({
          label:`比較:${key}`,   data: cmpY, borderWidth:2.5, pointRadius:0,
          borderColor: col, backgroundColor: col, borderDash:[6,4]
        });
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

    if (autoCoach) generateCoachNotes(rmseRes, stats, metrics, cycleNormalize);
  }, [refSamples, cmpSamples, metrics, cycleNormalize, autoCoach]);

  // ---- 生成AIコメント（まずはローカルのヒューリスティックで） ----
  const generateCoachNotes = useCallback(
    (rmse = compareRmse, stats = compareStats, enabled = metrics, mode = cycleNormalize) => {
      try {
        setCoachError(null);
        setCoachLoading(true);
        const text = buildHeuristicNotes(rmse, stats, enabled, mode);
        setCoachNotes(text);
      } catch (e) {
        setCoachError(e?.message || String(e));
      } finally {
        setCoachLoading(false);
      }
    },
    [compareRmse, compareStats, metrics, cycleNormalize]
  );

  // ---- UI ----
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
          {[0.25, 0.5, 0.75, 1].map(s => (
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

      {/* 保存ボタン */}
      <div style={{ marginTop:10, display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
        <button onClick={()=>saveCurrentAs("ref")} disabled={!samplesRef.current.length}>この記録を「お手本」に保存</button>
        <button onClick={()=>saveCurrentAs("cmp")} disabled={!samplesRef.current.length}>この記録を「比較」に保存</button>

        <span style={{marginLeft:8, color:"#333"}}>
          保存状況：お手本 {refSamples? "✅": "❌"} / 比較 {cmpSamples? "✅": "❌"}
        </span>
      </div>

      {/* 比較パネル */}
      <div style={{marginTop:12, padding:12, border:"1px solid #eee", borderRadius:8}}>
        {/* 指標の選択 & オプション */}
        <div style={{display:'flex', gap:12, flexWrap:'wrap', alignItems:'center'}}>
          {[
            {key:'kneeL', label:'左膝'}, {key:'kneeR', label:'右膝'},
            {key:'hipL',  label:'左股'}, {key:'hipR',  label:'右股'},
            {key:'trunk', label:'体幹前傾'},
          ].map(m => (
            <label key={m.key}>
              <input
                type="checkbox"
                checked={metrics[m.key] ?? true}
                onChange={e => setMetrics(v => ({ ...v, [m.key]: e.target.checked }))}
              />
              {m.label}
            </label>
          ))}

          <label style={{marginLeft:8}}>
            <input
              type="checkbox"
              checked={cycleNormalize}
              onChange={e => setCycleNormalize(e.target.checked)}
            />
            動きを1回分に揃えて、平均したフォームで比較
            <span style={{ fontSize:"0.85em", color:"#666", marginLeft:4 }}>
              （速さの違いを気にせず、フォームそのものを比べられます）
            </span>
          </label>

          <button onClick={runCompareMulti} disabled={!refSamples || !cmpSamples}>
            比較（グラフ）
          </button>

          <label style={{marginLeft:8}}>
            <input type="checkbox" checked={autoCoach} onChange={e=>setAutoCoach(e.target.checked)} />
            比較後にAIコメントを自動生成
          </label>
          <button onClick={()=>generateCoachNotes()} disabled={!compareResult && !compareStats} >
            AIコメントを生成
          </button>

          {compareResult && (
            <span style={{marginLeft:8}}>
              {Object.entries(compareRmse).map(([k,v]) => (
                <span key={k} style={{marginRight:10}}>{k}: RMSE {v?.toFixed(2)}°</span>
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
          <div style={{ height: 280, marginTop: 8, background:"#fafafa", border:"1px solid #eee", borderRadius:8, padding:8 }}>
            <Line
              data={compareResult.chartData}
              options={{
                responsive:true, maintainAspectRatio:false, animation:false,
                scales:{
                  x:{ title:{display:true, text: cycleNormalize ? 'サイクル(%)' : '時間(秒)'}, grid:{color:"#eee"}, ticks:{color:"#333"} },
                  y:{ title:{display:true, text:'角度(°)'}, grid:{color:"#eee"}, ticks:{color:"#333"} }
                },
                plugins:{ legend:{ position:'top', labels:{ usePointStyle:true, boxWidth:10 } } }
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

      {/* ★ グラフ領域（記録可視化） */}
      <div style={{ height: 260, marginTop: 12, background:"#fafafa", border:"1px solid #eee", borderRadius:8, padding:8 }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* 生成AIコメント表示 */}
      {(coachLoading || coachError || coachNotes) && (
        <div style={{marginTop:12, padding:12, border:"1px solid #eee", borderRadius:8, background:"#fffef8"}}>
          <div style={{fontWeight:700, marginBottom:6}}>コーチからのひとこと</div>
          {coachLoading && <div>考え中です… ⏳</div>}
          {coachError && <div style={{color:"#b00020"}}>エラー: {coachError}</div>}
          {coachNotes && coachNotes.split("\n").map((line,i)=>(
            <p key={i} style={{margin:"6px 0"}}>{line}</p>
          ))}
        </div>
      )}

      <p style={{ marginTop:12, color:"#555" }}>
        コツ：横から全身が入るように撮影（30fps以上）。明るい場所で。
      </p>
    </div>
  );
}

// ---------- 比較ロジック関数（下請け） ----------

// 線形補間
function linInterp(x, xp, yp) {
  if (!xp.length || !yp.length) return null;
  if (x <= xp[0]) return yp[0];
  if (x >= xp[xp.length-1]) return yp[yp.length-1];
  let i = 1;
  while (i < xp.length && xp[i] < x) i++;
  const x0 = xp[i-1], x1 = xp[i];
  const y0 = yp[i-1], y1 = yp[i];
  return y0 + (y1-y0) * (x-x0)/(x1-x0);
}

// ★ 補間用に null を軽く埋める（端は最近傍、内部は線形）
function fillNaLinear(xp, yp) {
  const y = yp.slice();
  let i = 0; while (i < y.length && y[i] == null) i++;
  if (i > 0 && i < y.length) for (let k = 0; k < i; k++) y[k] = y[i];
  let j = y.length - 1; while (j >= 0 && y[j] == null) j--;
  if (j >= 0 && j < y.length - 1) for (let k = j + 1; k < y.length; k++) y[k] = y[j];
  for (let a = 0; a < y.length; a++) if (y[a] == null) {
    let b = a; while (b < y.length && y[b] == null) b++;
    const y0 = y[a - 1], y1 = y[b], x0 = xp[a - 1], x1 = xp[b];
    for (let k = a; k < b; k++) y[k] = y0 + (y1 - y0) * (xp[k] - x0) / (x1 - x0);
    a = b;
  }
  return y;
}

// 極小値（谷）の検出：時間ベース
function findLocalMinima(t, y, {prominence=8, minGapSec=0.35} = {}) {
  const idxs = [];
  for (let i = 1; i < y.length - 1; i++) {
    if (y[i] <= y[i-1] && y[i] <= y[i+1]) idxs.push(i);
  }
  const kept = [];
  let lastKeepT = -1e9;
  for (const i of idxs) {
    const left = Math.max(0, i-10), right = Math.min(y.length-1, i+10);
    const leftMax  = Math.max(...y.slice(left, i));
    const rightMax = Math.max(...y.slice(i+1, right+1));
    const prom = Math.min(leftMax - y[i], rightMax - y[i]);
    if (prom >= prominence && (t[i] - lastKeepT) >= minGapSec) {
      kept.push(i);
      lastKeepT = t[i];
    }
  }
  return kept; // 返り値はインデックス配列
}

// サイクルごとの正規化 (0-100%)
function cyclesNormalize(times, values, peaks, N=100) {
  const cycles = [];
  for (let c=0; c<peaks.length-1; c++) {
    const t0 = times[peaks[c]], t1 = times[peaks[c+1]];
    const normT = Array.from({length:N}, (_,i)=>i/(N-1));
    const normV = normT.map(frac => {
      const targetT = t0 + frac*(t1-t0);
      return linInterp(targetT, times, values);
    });
    cycles.push({normT, normV, dur:t1-t0});
  }
  return cycles;
}

// RMSE
function rmse(arr1, arr2) {
  const n = Math.min(arr1.length, arr2.length);
  if (n===0) return null;
  let s=0; for (let i=0;i<n;i++){const d=arr1[i]-arr2[i]; s+=d*d;}
  return Math.sqrt(s/n);
}

// 平均と標準偏差
function avg(arr){return arr.reduce((a,b)=>a+b,0)/arr.length;}
function stdev(arr){const m=avg(arr);return Math.sqrt(avg(arr.map(v=>(v-m)**2)));}

// ---- 生成AIコメント用ヘルパ ----
function formatDeg(v) { return (v==null || isNaN(v)) ? "-" : `${v.toFixed(1)}°`; }
function levelFromRmse(v){
  if (v == null) return "info";
  if (v < 10) return "good";
  if (v < 20) return "ok";
  if (v < 30) return "warn";
  return "alert";
}
function bullet(prefix, text){ return `${prefix} ${text}`; }

// stats: {mode:'cycle'|'time', ref:{count,avg,sd,min,max,cadence}, cmp:{...}}
function buildHeuristicNotes(rmse, stats, enabledKeys, mode) {
  const lines = [];
  lines.push("いつもおつかれさまです。今日の計測をもとに、やさしく振り返ってみましょう。");

  const keys = ["kneeL","kneeR","hipL","hipR","trunk"].filter(k=>enabledKeys[k]);
  const map = { kneeL:"左膝", kneeR:"右膝", hipL:"左股関節", hipR:"右股関節", trunk:"体幹前傾" };
  for (const k of keys) {
    const v = rmse?.[k]; if (v == null) continue;
    const level = levelFromRmse(v);
    if (level === "good")   lines.push(bullet("✅", `${map[k]}はお手本に近い動き（RMSE ${formatDeg(v)}）。この調子！`));
    if (level === "ok")     lines.push(bullet("☑️", `${map[k]}はまずまず一致（RMSE ${formatDeg(v)}）。リズムを揃えるとさらに良くなりそう。`));
    if (level === "warn")   lines.push(bullet("⚠️", `${map[k]}の差がやや大きめ（RMSE ${formatDeg(v)}）。可動域のピーク付近で左右差が出ている可能性。`));
    if (level === "alert")  lines.push(bullet("❗", `${map[k]}の差が大きい傾向（RMSE ${formatDeg(v)}）。フォームの要点を一つに絞って練習しましょう。`));
  }

  if (stats?.mode === "cycle" && stats.ref && stats.cmp) {
    const cadRef = stats.ref.cadence, cadCmp = stats.cmp.cadence;
    const diff = (cadRef && cadCmp) ? Math.abs(cadRef - cadCmp) : null;
    if (diff != null) {
      if (diff < 2) lines.push(bullet("🎵", `ケイデンスはお手本に近いです（${cadCmp.toFixed(1)} 回/分）。リズムは良好！`));
      else lines.push(bullet("🫧", `ケイデンスに差があります（お手本 ${cadRef.toFixed(1)} / 比較 ${cadCmp.toFixed(1)} 回/分）。テンポ合わせを意識しましょう。`));
    }
    lines.push(bullet("⏱️", `1サイクルの平均：お手本 ${stats.ref.avg.toFixed(2)}s / 比較 ${stats.cmp.avg.toFixed(2)}s`));
  } else if (mode === false) {
    lines.push(bullet("ℹ️", "今回は時間比較です。必要に応じて『動きを1回分に揃えて、平均したフォームで比較』をONにすると、リズム差を除いた形で波形比較ができます。"));
  }

  lines.push("次回のおすすめ：");
  if (keys.includes("kneeL") || keys.includes("kneeR")) lines.push(bullet("・", "膝角度は“曲げピークの深さとタイミング”を合わせる意識で。"));
  if (keys.includes("trunk")) lines.push(bullet("・", "体幹は“胸の向き”を固定し、股関節で前傾を作ると安定します。"));
  lines.push(bullet("・", "1～2項目に絞って撮影し、5～10サイクルほど計測してみましょう。"));
  lines.push("無理せず、少しずつ。今日もよくできました！");

  return lines.join("\n");
}
