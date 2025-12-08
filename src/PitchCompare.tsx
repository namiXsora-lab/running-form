// src/PitchCompare.tsx
import React, { useEffect, useRef, useState } from "react";
import { createDetector, elbowAngle, trunkLeanDeg, stepLength, EMA, computePitchKPI, coachTips } from "./poseUtils";
import type { Detector } from "./poseUtils";
import { drawSkeleton, drawAngles } from "./poseUtils";


// 追加：動画の実サイズにキャンバスを合わせる
const syncCanvasSize = (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
};

type Side = "left" | "right";

export default function PitchCompare() {

  // 追加：オーバーレイ用キャンバス
  const leftCanvasRef  = useRef<HTMLCanvasElement>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement>(null);

  const [detector, setDetector] = useState<Detector | null>(null);

  const leftVideoRef  = useRef<HTMLVideoElement>(null);
  const rightVideoRef = useRef<HTMLVideoElement>(null);

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [powerSave, setPowerSave] = useState(false);

  const [leftFileName, setLeftFileName] = useState<string>("");
  const [rightFileName, setRightFileName] = useState<string>("");

  // 記録
  const leftElbowSeq  = useRef<number[]>([]);
  const leftTrunkSeq  = useRef<number[]>([]);
  const leftStepSeq   = useRef<number[]>([]);
  const rightElbowSeq = useRef<number[]>([]);
  const rightTrunkSeq = useRef<number[]>([]);
  const rightStepSeq  = useRef<number[]>([]);

  // 平滑化
  const lElbowEMA = useRef(new EMA(0.35));
  const lTrunkEMA = useRef(new EMA(0.35));
  const lStepEMA  = useRef(new EMA(0.35));
  const rElbowEMA = useRef(new EMA(0.35));
  const rTrunkEMA = useRef(new EMA(0.35));
  const rStepEMA  = useRef(new EMA(0.35));

  const [kpiLeft, setKpiLeft]   = useState<any>(null);
  const [kpiRight, setKpiRight] = useState<any>(null);

  // Detector 初期化
  useEffect(() => {
    (async () => {
      const det = await createDetector();
      setDetector(det);
    })();
  }, []);

  // 推定ループ
  useEffect(() => {
    let raf = 0;
    let lastTs = 0;

    async function step(ts: number) {
      raf = requestAnimationFrame(step);
      if (!detector) return;

      // 省エネ：フレーム間隔をあける（~20fps程度）
      if (powerSave && ts - lastTs < 50) return;
      lastTs = ts;

      await Promise.all([
        analyzeOne(leftVideoRef.current, "left"),
        analyzeOne(rightVideoRef.current, "right"),
      ]);
    }

    async function analyzeOne(video: HTMLVideoElement | null, which: Side) {
      if (!video || video.paused || video.ended) return;
      const poses = await detector!.estimatePoses(video, { flipHorizontal: false });
      if (!poses?.[0]) return;
      const pose = poses[0];

      // 右投げ想定で右肘角を採用（左投げチームなら切替可）
      const e = elbowAngle(pose, "right");
      const t = trunkLeanDeg(pose);
      const s = stepLength(pose);

      // EMA
      const eS = which === "left" ? lElbowEMA.current.push(e) : rElbowEMA.current.push(e);
      const tS = which === "left" ? lTrunkEMA.current.push(t) : rTrunkEMA.current.push(t);
      const sS = which === "left" ? lStepEMA.current.push(s) : rStepEMA.current.push(s);

      if (which === "left") {
        if (eS != null) leftElbowSeq.current.push(eS);
        if (tS != null) leftTrunkSeq.current.push(tS);
        if (sS != null) leftStepSeq.current.push(sS);
      } else {
        if (eS != null) rightElbowSeq.current.push(eS);
        if (tS != null) rightTrunkSeq.current.push(tS);
        if (sS != null) rightStepSeq.current.push(sS);
      }

      // 動画が終盤になったらKPI更新（都度でもOK）
      if (video.currentTime > (video.duration * 0.85)) {
        if (which === "left") {
          setKpiLeft(computePitchKPI(leftElbowSeq.current, leftTrunkSeq.current, leftStepSeq.current));
        } else {
          setKpiRight(computePitchKPI(rightElbowSeq.current, rightTrunkSeq.current, rightStepSeq.current));
        }
      }

      // ★★★ ここから追加：毎フレームの骨格描画 ★★★
      const canvas = which === "left" ? leftCanvasRef.current : rightCanvasRef.current;
      if (!canvas) return;

      // 万一サイズ未設定ならここで動画に合わせる
      if (canvas.width === 0 || canvas.height === 0) {
        syncCanvasSize(video, canvas);
      }
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // ① デバッグ: Canvasが見えるか（左上に赤い四角）
      ctx.fillStyle = "rgba(255,0,0,0.6)";
      ctx.fillRect(8, 8, 10, 10);

      // ② デバッグ: 検出できているキーポイント数を表示
      const good = pose.keypoints.filter(k => (k.score ?? 0) > 0.2).length;
      ctx.fillStyle = "rgba(0,0,0,0.8)";
      ctx.font = "14px system-ui";
      ctx.fillText(`${good} kp`, 24, 18);

      // ③ スケルトン（しきい値ゆるめ）
      drawSkeleton(ctx, pose, { lineWidth: 3, pointSize: 3, minScore: 0.2 });

      // ④ 角度ラベル（EMAの最新値を表示）
      const emaElbowDeg = which === "left" ? lElbowEMA.current.value() : rElbowEMA.current.value();
      const emaTrunkDeg = which === "left" ? lTrunkEMA.current.value() : rTrunkEMA.current.value();
      drawAngles(ctx, pose, emaElbowDeg, emaTrunkDeg);
    }

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [detector, powerSave]);

  // 再生/停止
  const playPause = () => {
    const L = leftVideoRef.current, R = rightVideoRef.current;
    if (!L || !R) return;
    if (playing) { L.pause(); R.pause(); setPlaying(false); }
    else { L.playbackRate = speed; R.playbackRate = speed; L.play(); R.play(); setPlaying(true); }
  };

  const replay = () => {
    const L = leftVideoRef.current, R = rightVideoRef.current;
    if (!L || !R) return;
    // 既存の記録をクリア
    leftElbowSeq.current = []; leftTrunkSeq.current = []; leftStepSeq.current = [];
    rightElbowSeq.current = []; rightTrunkSeq.current = []; rightStepSeq.current = [];
    lElbowEMA.current = new EMA(0.35); lTrunkEMA.current = new EMA(0.35); lStepEMA.current = new EMA(0.35);
    rElbowEMA.current = new EMA(0.35); rTrunkEMA.current = new EMA(0.35); rStepEMA.current = new EMA(0.35);
    setKpiLeft(null); setKpiRight(null);

    L.currentTime = 0; R.currentTime = 0;
    if (playing) { L.play(); R.play(); }
  };

  const changeSpeed = (s: number) => {
    setSpeed(s);
    const L = leftVideoRef.current, R = rightVideoRef.current;
    if (L) L.playbackRate = s;
    if (R) R.playbackRate = s;
  };

const onLoadVideo = (e: React.ChangeEvent<HTMLInputElement>, side: "left" | "right") => {
  const file = e.target.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);

  const v = (side === "left" ? leftVideoRef : rightVideoRef).current!;
  const c = (side === "left" ? leftCanvasRef : rightCanvasRef).current!;

  // 先にハンドラを登録（メタデータ読み込みが速すぎて取りこぼさないため）
  const sync = () => { if (v && c) syncCanvasSize(v, c); };
  v.onloadedmetadata = sync;
  v.onloadeddata = sync; // 予備
  v.src = url;           // 最後にセット

  if (side === "left") setLeftFileName(file.name); else setRightFileName(file.name);
};

  const renderKPI = (k: any, label: string) => {
    if (!k) return <div style={{opacity:0.6}}>（動画の終盤で自動計算します）</div>;
    return (
      <div style={{fontSize:14, lineHeight:1.6}}>
        <div><b>{label}</b></div>
        <div>最大外旋の目安（肘角の最小値）：{fmt(k.maxExternalRotation, 1)}°</div>
        <div>リリース推定フレーム：{k.releaseFrame ?? "-"}</div>
        <div>リリース時の肘角：{fmt(k.releaseElbowAngle, 1)}°</div>
        <div>最大体幹傾き：{fmt(k.maxTrunkLean, 1)}°</div>
        <div>最大ステップ幅（相対）：{fmt(k.maxStepLen, 0)}</div>
        <ul style={{marginTop:6}}>
          {coachTips(k).map((t: string, i: number) => <li key={i}>{t}</li>)}
        </ul>
      </div>
    );
  };

  const compDelta = () => {
    if (!kpiLeft || !kpiRight) return null;
    const d = (a: number|null, b: number|null) => (a!=null && b!=null) ? (a-b) : null;
    const rows = [
      ["最大外旋（肘角min／小さいほど曲がる）", d(kpiLeft.maxExternalRotation, kpiRight.maxExternalRotation), "°"],
      ["リリース肘角（大きい=伸びている）", d(kpiLeft.releaseElbowAngle, kpiRight.releaseElbowAngle), "°"],
      ["最大体幹傾き（前傾）", d(kpiLeft.maxTrunkLean, kpiRight.maxTrunkLean), "°"],
      ["最大ステップ幅（相対）", d(kpiLeft.maxStepLen, kpiRight.maxStepLen), ""],
    ];
    return (
      <table style={{borderCollapse:"collapse", marginTop:8, fontSize:14}}>
        <thead>
          <tr>
            <th style={th}>指標</th>
            <th style={th}>左 − 右（差）</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r,i)=>(
            <tr key={i}>
              <td style={td}>{r[0]}</td>
              <td style={td}>{r[1]==null ? "-" : `${(r[1] as number).toFixed(1)}${r[2]}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div style={{padding:12}}>
      <h2 style={{marginTop:0}}>キャッチボール：投球フォーム比較</h2>

      {/* 動画読み込み */}
      <div style={{display:"flex", gap:12, flexWrap:"wrap", alignItems:"center"}}>
        <div>
          <label style={uploader}>
            左動画を選ぶ
            <input type="file" accept="video/*" onChange={(e)=>onLoadVideo(e,"left")} style={{display:"none"}} />
          </label>
          <div style={{fontSize:12, opacity:0.7}}>{leftFileName || "未選択"}</div>
        </div>
        <div>
          <label style={uploader}>
            右動画を選ぶ
            <input type="file" accept="video/*" onChange={(e)=>onLoadVideo(e,"right")} style={{display:"none"}} />
          </label>
          <div style={{fontSize:12, opacity:0.7}}>{rightFileName || "未選択"}</div>
        </div>
      </div>

      {/* 並列表示（動画+骨格オーバーレイ） */}
      <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginTop:12}}>
        <div style={stackWrap}>
          <video ref={leftVideoRef}  playsInline muted style={vid} />
          <canvas ref={leftCanvasRef} style={overlayCanvas}/>
        </div>
        <div style={stackWrap}>
          <video ref={rightVideoRef} playsInline muted style={vid} />
          <canvas ref={rightCanvasRef} style={overlayCanvas}/>
        </div>
      </div>

      {/* コントロール */}
      <div style={{ marginTop:10, display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
        <button onClick={playPause}>{playing ? "⏸ 一時停止" : "▶ 再生"}</button>
        <button onClick={replay}>⟲ リプレイ</button>
        <span>速度:</span>
        {[0.25, 0.5, 0.75, 1].map(s => (
          <button key={s} onClick={()=>changeSpeed(s)} disabled={speed===s}>{s}x</button>
        ))}
        <label style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={powerSave} onChange={e => setPowerSave(e.target.checked)} />
          省エネモード（発熱を抑える）
        </label>
      </div>

      {/* 結果パネル */}
      <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginTop:12}}>
        <div style={panel}>{renderKPI(kpiLeft, "左動画のKPI")}</div>
        <div style={panel}>{renderKPI(kpiRight, "右動画のKPI")}</div>
      </div>

      <div style={{marginTop:12}}>
        <h3 style={{margin:"12px 0 6px"}}>左右差（左 − 右）</h3>
        {compDelta()}
        <div style={{fontSize:12, opacity:0.7, marginTop:6}}>
          ※ 肘角：小さい＝曲げている／大きい＝伸びている。動画の解像度・撮影角度により相対比較が基本です。
        </div>
      </div>
    </div>
  );
}

const vid: React.CSSProperties = { width:"100%", background:"#000", borderRadius:8 };
const panel: React.CSSProperties = { padding:12, border:"1px solid #eee", borderRadius:8, background:"#fff" };
const uploader: React.CSSProperties = { display:"inline-block", padding:"8px 12px", border:"1px solid #ccc", borderRadius:6, cursor:"pointer", background:"#fafafa" };
const th: React.CSSProperties = { border:"1px solid #ddd", padding:"6px 8px", background:"#f6f6f6", textAlign:"left" };
const td: React.CSSProperties = { border:"1px solid #eee", padding:"6px 8px" };

const stackWrap: React.CSSProperties = { position:"relative", width:"100%" };
const overlayCanvas: React.CSSProperties = {
  position:"absolute", left:0, top:0, width:"100%", height:"100%", pointerEvents:"none", zIndex:1
};

function fmt(v?: number|null, d=1) { return v==null ? "-" : v.toFixed(d); }
