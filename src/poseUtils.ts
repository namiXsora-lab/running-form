// src/poseUtils.ts
import * as tf from "@tensorflow/tfjs";
import * as posedetection from "@tensorflow-models/pose-detection";
import { Pose } from "@tensorflow-models/pose-detection";

export type Detector = posedetection.PoseDetector;

// ====== 初期化 ======
export async function createDetector(): Promise<Detector> {
  const model = posedetection.SupportedModels.MoveNet;
  return await posedetection.createDetector(model, {
    modelType: "Thunder", // 精度優先（LightningでもOK）
    enableSmoothing: true,
  });
}



// ====== キーポイント取得（名前→index）======
const KP = posedetection.util.getKeypointIndexBySide(posedetection.SupportedModels.MoveNet);
// 主要部位のヘルパ
function get(pose: Pose, name: posedetection.Keypoint["name"]) {
  const k = pose.keypoints.find((k) => k.name === name);
  return k && k.score !== undefined && k.score > 0.3 ? k : undefined;
}

export type XY = { x: number; y: number };
function vec(a?: posedetection.Keypoint, b?: posedetection.Keypoint): XY | undefined {
  if (!a || !b) return;
  return { x: b.x - a.x, y: b.y - a.y };
}
function deg(rad: number) { return (rad * 180) / Math.PI; }

// 2ベクトルのなす角（0-180）
export function angleBetween(u?: XY, v?: XY): number | undefined {
  if (!u || !v) return;
  const dot = u.x * v.x + u.y * v.y;
  const nu = Math.hypot(u.x, u.y);
  const nv = Math.hypot(v.x, v.y);
  if (nu === 0 || nv === 0) return;
  const cos = Math.min(1, Math.max(-1, dot / (nu * nv)));
  return deg(Math.acos(cos));
}

// 距離
export function dist(a?: posedetection.Keypoint, b?: posedetection.Keypoint): number | undefined {
  if (!a || !b) return;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// 角度：肘（上腕と前腕の角度）
export function elbowAngle(pose: Pose, side: "left" | "right"): number | undefined {
  const shoulder = get(pose, `${side}_shoulder` as any);
  const elbow    = get(pose, `${side}_elbow` as any);
  const wrist    = get(pose, `${side}_wrist` as any);
  return angleBetween(vec(elbow, shoulder), vec(elbow, wrist));
}

// 体幹の傾き（肩-腰の線と水平の角度）
export function trunkLeanDeg(pose: Pose): number | undefined {
  const ls = get(pose, "left_shoulder" as any);
  const rs = get(pose, "right_shoulder" as any);
  const lh = get(pose, "left_hip" as any);
  const rh = get(pose, "right_hip" as any);
  if (!ls || !rs || !lh || !rh) return;
  const cx = (ls.x + rs.x) / 2, cy = (ls.y + rs.y) / 2;
  const hx = (lh.x + rh.x) / 2, hy = (lh.y + rh.y) / 2;
  const dx = cx - hx, dy = cy - hy;
  const a = Math.atan2(dy, dx); // -pi..pi
  return Math.abs(deg(a)); // 正の角度（前傾/後傾の大きさ）
}

// ステップ幅（前脚か後脚かは右投げ想定で right を軸にした簡易版）
export function stepLength(pose: Pose): number | undefined {
  const la = get(pose, "left_ankle" as any);
  const ra = get(pose, "right_ankle" as any);
  return dist(la, ra); // ピクセル距離（相対比較用途）
}

// ====== 簡易平滑化（EMA）======
export class EMA {
  private alpha: number;
  private v?: number;
  constructor(alpha = 0.3) { this.alpha = alpha; }
  push(x?: number): number | undefined {
    if (x == null) return this.v;
    this.v = this.v == null ? x : this.v + this.alpha * (x - this.v);
    return this.v;
  }
  value() { return this.v; }
}

// ====== ここから追加：スケルトン描画ヘルパ ======

// MoveNet 用の「つながっている関節の組み合わせ」を取得
const ADJ_PAIRS = posedetection.util.getAdjacentPairs(
  posedetection.SupportedModels.MoveNet
);

// スケルトン描画の見た目オプション（太さ・点サイズなど）
export type DrawOpts = {
  lineWidth?: number;
  pointSize?: number;
  minScore?: number;   // このスコア未満の点は描画しない（ノイズ除け）
};

// 骨格ライン（スケルトン）を描く
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  opts: DrawOpts = {}
) {
  const { lineWidth = 3, pointSize = 3, minScore = 0.35 } = opts;

  // 線（関節どうしを結ぶ）
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.beginPath();
  for (const [i, j] of ADJ_PAIRS) {
    const a = pose.keypoints[i];
    const b = pose.keypoints[j];
    if (!a || !b) continue;
    if ((a.score ?? 0) < minScore || (b.score ?? 0) < minScore) continue;
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();

  // 関節点（白い下地→黒い点の順で描くと視認性UP）
  for (const k of pose.keypoints) {
    if (!k || (k.score ?? 0) < minScore) continue;
    ctx.beginPath();
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.arc(k.x, k.y, pointSize + 1, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.fillStyle = "rgba(0,0,0,0.9)";
    ctx.arc(k.x, k.y, pointSize, 0, Math.PI * 2);
    ctx.fill();
  }
}

// 右肘角と体幹傾きのガイド線＋角度ラベル（任意で表示）
export function drawAngles(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  rightElbowDeg?: number,  // 表示する角度値（例：EMA後の値）
  trunkDeg?: number,       // 同上
  minScore = 0.35
) {
  const kp = (name: posedetection.Keypoint["name"]) =>
    pose.keypoints.find((k) => k.name === name);

  // 右肘のガイド（三角形＋角度ラベル）
  const rShoulder = kp("right_shoulder");
  const rElbow    = kp("right_elbow");
  const rWrist    = kp("right_wrist");
  if (
    rShoulder && rElbow && rWrist &&
    (rShoulder.score ?? 0) >= minScore &&
    (rElbow.score ?? 0) >= minScore &&
    (rWrist.score ?? 0) >= minScore
  ) {
    ctx.strokeStyle = "rgba(30,144,255,0.9)"; // 青
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rElbow.x, rElbow.y);
    ctx.lineTo(rShoulder.x, rShoulder.y);
    ctx.lineTo(rWrist.x, rWrist.y);
    ctx.stroke();

    if (rightElbowDeg != null) {
      ctx.fillStyle = "rgba(30,144,255,0.95)";
      ctx.font = "12px system-ui, -apple-system, sans-serif";
      ctx.fillText(`${rightElbowDeg.toFixed(0)}°`, rElbow.x + 6, rElbow.y - 6);
    }
  }

  // 体幹のガイド（腰中心→肩中心の線＋角度ラベル）
  const ls = kp("left_shoulder"), rs = kp("right_shoulder");
  const lh = kp("left_hip"),      rh = kp("right_hip");
  if (ls && rs && lh && rh) {
    const ok =
      (ls.score ?? 0) >= minScore &&
      (rs.score ?? 0) >= minScore &&
      (lh.score ?? 0) >= minScore &&
      (rh.score ?? 0) >= minScore;
    if (ok) {
      const cx = (ls.x + rs.x) / 2, cy = (ls.y + rs.y) / 2; // 肩の中心
      const hx = (lh.x + rh.x) / 2, hy = (lh.y + rh.y) / 2; // 腰の中心
      ctx.strokeStyle = "rgba(50,205,50,0.9)"; // 緑
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(cx, cy);
      ctx.stroke();

      if (trunkDeg != null) {
        ctx.fillStyle = "rgba(50,205,50,0.95)";
        ctx.font = "12px system-ui, -apple-system, sans-serif";
        ctx.fillText(`${trunkDeg.toFixed(0)}°`, cx + 6, cy + 6);
      }
    }
  }
}
// ====== 追加ここまで ======

// ====== 投球KPI抽出 ======
export type PitchKPI = {
  maxExternalRotation: number | null;   // 〈最大“外旋”＝肘屈曲が小さいほど外旋は小…〉簡易近似：肘角の最小値（~90°前後）
  releaseFrame: number | null;          // リリース推定フレーム
  releaseElbowAngle: number | null;     // リリース時の肘角（~160-175°が目安）
  maxTrunkLean: number | null;          // 最大体幹傾き
  maxStepLen: number | null;            // 最大ステップ幅（相対値）
};

export function computePitchKPI(elbowSeq: number[], trunkSeq: number[], stepSeq: number[]): PitchKPI {
  // 最大外旋 ≒ 肘角の最小値（小さい=曲がっている）
  const minElbow = elbowSeq.length ? Math.min(...elbowSeq) : null;

  // リリース ≒ 肘角が「屈曲→伸展」に転じて 165° を初めて超えたフレーム（直前10fの平均が <150°）
  let releaseIdx: number | null = null;
  for (let i = 12; i < elbowSeq.length; i++) {
    const prev = elbowSeq.slice(i - 10, i);
    const prevMean = prev.reduce((a, b) => a + b, 0) / prev.length;
    if (prev.length >= 5 && prevMean < 150 && elbowSeq[i] >= 165) {
      releaseIdx = i;
      break;
    }
  }

  const releaseElbow = releaseIdx != null ? elbowSeq[releaseIdx] : null;
  const maxTrunk = trunkSeq.length ? Math.max(...trunkSeq) : null;
  const maxStep  = stepSeq.length ? Math.max(...stepSeq) : null;

  return {
    maxExternalRotation: minElbow ?? null,
    releaseFrame: releaseIdx,
    releaseElbowAngle: releaseElbow,
    maxTrunkLean: maxTrunk ?? null,
    maxStepLen: maxStep ?? null,
  };
}

// シンプルな助言（小学生向け）
export function coachTips(k: PitchKPI) {
  const tips: string[] = [];
  if (k.maxExternalRotation != null && k.maxExternalRotation > 120) {
    tips.push("腕を引くタイミングを少し早めて、肘がもう少し曲がるとパワーが伝わるよ！");
  }
  if (k.releaseElbowAngle != null && k.releaseElbowAngle < 160) {
    tips.push("ボールをはなす時に、もう少し腕をしっかり伸ばしてみよう！");
  }
  if (k.maxTrunkLean != null && k.maxTrunkLean < 8) {
    tips.push("体を少し前に倒す（おへそをゴールに向ける）イメージで投げてみよう！");
  }
  if (k.maxStepLen != null && k.maxStepLen < 60) {
    tips.push("一歩をもう少し大きく踏み出すと、強いボールになりやすいよ！");
  }
  if (tips.length === 0) tips.push("とても良いフォーム！この調子で投げてみよう👌");
  return tips;
}
