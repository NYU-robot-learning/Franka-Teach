#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay a given list / array of absolute Franka actions,
but APPLY THEM AS DELTAS: for each new absolute GT, compute
delta vs previous GT and compose that delta on top of the last
commanded pose before sending.

Inputs supported:
- --actions '[...]'
- --file .npy/.npz/.pkl/.pickle/.txt/.csv
  * CSV auto-detects numeric vs FrankaTeach headers.

Safety:
- thresholds operate on the absolute GT sequence via interpolation;
  after splitting, each adjacent pair becomes a small delta that is
  integrated into the last command.

Gripper:
- kept ABSOLUTE from GT (safer & consistent with your dataset).
"""

import argparse, ast, pickle, sys, time, math, csv, re
from pathlib import Path
import numpy as np

from frankateach.network import create_request_socket
from frankateach.messages import FrankaAction
from frankateach.constants import HOST, CONTROL_PORT

# ---------- Axis-angle & quaternion helpers ----------
def parse_aa(s: str) -> np.ndarray:
    """Parse FrankaTeach-style axis-angle '[np.float64(...), ...]' -> (6,) float32."""
    if s is None:
        raise ValueError("cmd_pose_aa is missing in CSV row")
    cleaned = re.sub(r'np\.float64\(|\)', '', s).strip('[]')
    arr = np.fromstring(cleaned, sep=',', dtype=np.float32)
    if arr.size != 6:
        raise ValueError(f"cmd_pose_aa should have 6 numbers, got {arr.size}: {s[:80]}...")
    return arr

def aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """scaled axis-angle (ax,ay,az) -> quaternion (x,y,z,w)."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = (aa / angle).astype(np.float64)
    half = angle * 0.5
    s = math.sin(half)
    q = np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(half)], dtype=np.float64)
    return q.astype(np.float32)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([0,0,0,1], dtype=np.float64)
    return (q / n).astype(np.float64)

def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (x,y,z,w)."""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    q = np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)
    return quat_normalize(q).astype(np.float32)

def quat_relative(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Shortest-path relative rotation q_rel that maps q_from -> q_to."""
    q_from = quat_normalize(q_from)
    q_to   = quat_normalize(q_to)
    # Align sign to avoid 2π jumps.
    if float(np.dot(q_from, q_to)) < 0.0:
        q_to = -q_to
    q_rel = quat_mul(quat_conj(q_from), q_to)  # conj(from) * to
    return q_rel

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quats."""
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return quat_normalize(q).astype(np.float32)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    q = s0 * q0 + s1 * q1
    return quat_normalize(q).astype(np.float32)

def rot_angle_between(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle between unit quats (radians)."""
    dot = float(np.dot(quat_normalize(q0), quat_normalize(q1)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(abs(dot))

def to_action(pos: np.ndarray, quat: np.ndarray, grip: float) -> FrankaAction:
    return FrankaAction(pos=pos.astype(np.float32),
                        quat=quat.astype(np.float32),
                        gripper=float(grip),
                        reset=False,
                        timestamp=time.time())

# ---------- Loading ----------
def _load_numeric_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=',')
    return np.asarray(arr)

def _load_frankateach_csv(path: Path, grip_default: float) -> np.ndarray:
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header; expected FrankaTeach header for --csv_format frankateach")
        grip_key = 'cmd_gripper_state' if 'cmd_gripper_state' in reader.fieldnames else ('gripper_state' if 'gripper_state' in reader.fieldnames else None)
        for i, row in enumerate(reader, start=1):
            try:
                aa = parse_aa(row.get('cmd_pose_aa'))
                grip = float(row[grip_key]) if grip_key and row.get(grip_key) not in (None, '') else float(grip_default)
                rows.append(np.concatenate([aa, [grip]], axis=0).astype(np.float32))
            except Exception as e:
                raise ValueError(f"CSV row {i} parse error: {e}")
    if not rows:
        raise ValueError("Frankateach CSV parsed 0 rows")
    arr = np.stack(rows, axis=0)
    print(f"Detected FrankaTeach CSV: parsed {arr.shape[0]} frames from {path.name}")
    return arr

def _load_csv_auto(path: Path, grip_default: float) -> np.ndarray:
    with open(path, 'r', newline='') as f:
        header = f.readline().strip().lower()
    if ('cmd_pose_aa' in header) or ('created timestamp' in header) or ('cmd_gripper_state' in header):
        return _load_frankateach_csv(path, grip_default)
    try:
        return _load_numeric_csv(path)
    except Exception as e:
        raise ValueError(f"CSV auto-detect failed; try --csv_format frankateach or provide numeric CSV. Root cause: {e}")

def _load_from_file(path: Path, csv_format: str, grip_default: float) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == '.npy':
        arr = np.load(path)
    elif suf == '.npz':
        npz = np.load(path); arr = npz[npz.files[0]]
    elif suf in ('.pkl', '.pickle'):
        with open(path, 'rb') as f: arr = pickle.load(f)
    elif suf == '.txt':
        arr = np.loadtxt(path)
    elif suf == '.csv':
        arr = _load_frankateach_csv(path, grip_default) if csv_format == 'frankateach' \
              else _load_numeric_csv(path) if csv_format == 'numeric' \
              else _load_csv_auto(path, grip_default)
    else:
        raise ValueError(f"Unsupported file type: {suf}")
    return np.asarray(arr)

def _load_actions(args) -> np.ndarray:
    if args.file is None and args.actions is None:
        raise ValueError("必须通过 --file 或 --actions 提供动作数据")
    if args.file is not None and args.actions is not None:
        raise ValueError("请二选一：--file 或 --actions")
    if args.file is not None:
        arr = _load_from_file(Path(args.file), args.csv_format, args.grip_default)
    else:
        try:
            arr = np.asarray(ast.literal_eval(args.actions))
        except Exception as e:
            raise ValueError(f"--actions 解析失败：{e}")

    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.asarray(arr, dtype=np.float32)

    if arr.shape[1] == 6:
        grip_col = np.full((arr.shape[0], 1), float(args.grip_default), dtype=np.float32)
        arr = np.concatenate([arr, grip_col], axis=1)
    elif arr.shape[1] != 7:
        raise ValueError(f"动作维度应为 6 或 7，收到 shape={arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("输入包含非有限数值 (NaN/Inf)")

    return arr

# ---------- Frames & thresholds on ABSOLUTE GT ----------
def build_frames_abs(actions: np.ndarray):
    """Absolute GT -> [{'pos','quat','grip'}]."""
    frames = []
    for row in actions:
        pos  = row[:3].astype(np.float32)
        aa   = row[3:6].astype(np.float32)
        quat = aa_to_quat(aa)
        grip = float(row[6])
        frames.append({'pos': pos, 'quat': quat, 'grip': grip})
    return frames

def enforce_thresholds_abs(frames, trans_thresh, rot_thresh_rad, grip_thresh, on_violate='split'):
    """Operate on ABSOLUTE GT frames; splits via linear/SLERP so adjacent deltas are small."""
    if len(frames) <= 1:
        return frames, 0, 0
    out = [frames[0]]
    inserted = 0
    skipped = 0
    def need(val): return val is not None and val > 0.0

    for nxt in frames[1:]:
        prv = out[-1]
        d_trans = float(np.linalg.norm(nxt['pos'] - prv['pos']))
        d_rot   = rot_angle_between(prv['quat'], nxt['quat'])
        d_grip  = abs(nxt['grip'] - prv['grip'])

        exceed = (
            (need(trans_thresh) and d_trans > trans_thresh) or
            (need(rot_thresh_rad) and d_rot   > rot_thresh_rad) or
            (need(grip_thresh) and d_grip     > grip_thresh)
        )
        if not exceed:
            out.append(nxt); continue

        if on_violate == 'skip':
            skipped += 1; continue
        if on_violate == 'stop':
            raise ValueError(f"Step exceeds threshold(s): Δx={d_trans:.4f} m, Δθ={math.degrees(d_rot):.2f}°, Δgrip={d_grip:.3f}")

        # split (interpolate in absolute space)
        n_div = 1
        if need(trans_thresh) and d_trans > 0: n_div = max(n_div, int(math.ceil(d_trans / trans_thresh)))
        if need(rot_thresh_rad) and d_rot > 0: n_div = max(n_div, int(math.ceil(d_rot / rot_thresh_rad)))
        if need(grip_thresh) and d_grip > 0:   n_div = max(n_div, int(math.ceil(d_grip / grip_thresh)))

        for k in range(1, n_div):
            t = k / float(n_div)
            pos  = (1 - t) * prv['pos'] + t * nxt['pos']
            quat = quat_slerp(prv['quat'], nxt['quat'], t)
            grip = (1 - t) * prv['grip'] + t * nxt['grip']
            out.append({'pos': pos.astype(np.float32),
                        'quat': quat.astype(np.float32),
                        'grip': float(grip)})
            inserted += 1
        out.append(nxt)

    return out, inserted, skipped

# ---------- Replay (delta integrate) ----------
def replay_given(actions_abs: np.ndarray, hz: float, speed: float,
                 trans_thresh: float, rot_thresh_deg: float, grip_thresh: float,
                 on_violate: str):
    if actions_abs.shape[0] < 1:
        print("动作序列为空，退出"); return

    # ABSOLUTE GT -> threshold split -> small absolute steps
    frames_abs = build_frames_abs(actions_abs)
    rot_thresh_rad = math.radians(rot_thresh_deg) if rot_thresh_deg > 0 else 0.0
    frames_abs, inserted, skipped = enforce_thresholds_abs(
        frames_abs, trans_thresh, rot_thresh_rad, grip_thresh, on_violate=on_violate
    )
    print(f"Thresholds applied: inserted {inserted} micro-steps, skipped {skipped} steps, final frames={len(frames_abs)}")

    sock = create_request_socket(HOST, CONTROL_PORT)

    # Move to first absolute frame
    f0 = frames_abs[0]
    print("Step-0: move to first absolute frame …")
    import ipdb; ipdb.set_trace()  # Debug
    sock.send(pickle.dumps(to_action(f0['pos'], f0['quat'], f0['grip']), protocol=-1))
    _ = sock.recv()
    time.sleep(0.8)

    # Initialize "last commanded" pose with what we just sent
    pos_cmd  = f0['pos'].copy()
    quat_cmd = f0['quat'].copy()
    grip_cmd = float(f0['grip'])

    n_frames = len(frames_abs) - 1
    print(f"Start replay-as-delta, {n_frames} frames, hz={hz} Hz, speed×{speed}")

    dt = 0.0
    if speed > 0 and hz > 0:
        dt = (1.0 / float(hz)) / float(speed)
    t0 = time.perf_counter()

    for i in range(1, len(frames_abs)):
        prev_abs = frames_abs[i-1]
        curr_abs = frames_abs[i]

        # --- compute DELTAS from absolute GT ---
        dpos  = (curr_abs['pos'] - prev_abs['pos']).astype(np.float32)
        q_rel = quat_relative(prev_abs['quat'], curr_abs['quat'])  # robust relative rotation

        # --- compose deltas onto LAST COMMANDED pose ---
        pos_cmd  = (pos_cmd + dpos).astype(np.float32)
        quat_cmd = quat_mul(quat_cmd, q_rel)  # last_cmd ⊗ q_rel
        grip_cmd = float(curr_abs['grip'])    # keep gripper ABSOLUTE

        # pacing
        if dt > 0.0:
            target = t0 + i * dt
            while True:
                remain = target - time.perf_counter()
                if remain <= 0: break
                time.sleep(0.001 if remain > 0.01 else 0.0002)
        
        # import ipdb; ipdb.set_trace()  # Debugging point to inspect variables during replay

        # send composed absolute command
        sock.send(pickle.dumps(to_action(pos_cmd, quat_cmd, grip_cmd), protocol=-1))
        _ = sock.recv()

    print("Replay done")
    sock.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="动作文件路径（.npy/.npz/.pkl/.pickle/.txt/.csv）")
    ap.add_argument("--actions", help="Python 字面量列表/数组字符串，例如 '[[...],[...]]'")
    ap.add_argument("--csv_format", choices=["auto", "numeric", "frankateach"], default="auto")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="播放速度倍率；0 为尽快播放（无等待）")
    ap.add_argument("--grip_default", type=float, default=0.0,
                    help="当仅提供 6 维动作时，用此常数填充 gripper")

    # Thresholds (applied on ABSOLUTE GT, before delta composition)
    ap.add_argument("--trans_thresh", type=float, default=0.02, help="单步最大平移 (m)；<=0 关闭")
    ap.add_argument("--rot_thresh_deg", type=float, default=10.0, help="单步最大旋转角 (度)；<=0 关闭")
    ap.add_argument("--grip_thresh", type=float, default=0.0, help="单步最大夹爪变化；<=0 关闭")
    ap.add_argument("--on_violate", choices=["split", "skip", "stop"], default="split")

    args = ap.parse_args()

    try:
        actions_abs = _load_actions(args)  # ABSOLUTE GT
        replay_given(actions_abs,
                     hz=args.hz, speed=args.speed,
                     trans_thresh=args.trans_thresh,
                     rot_thresh_deg=args.rot_thresh_deg,
                     grip_thresh=args.grip_thresh,
                     on_violate=args.on_violate)
    except KeyboardInterrupt:
        sys.exit("\n用户终止")
    except Exception as e:
        sys.exit(f"错误：{e}")

if __name__ == "__main__":
    main()
