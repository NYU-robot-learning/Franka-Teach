#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay a given list / array of absolute Franka actions with per-step safety thresholds.

Supports inputs:
- Python literal via --actions '[...]'
- Files via --file:
  * .npy / .npz / .pkl / .pickle / .txt
  * .csv:
      - numeric CSV (6 or 7 numeric columns, no header)
      - FrankaTeach CSV with headers (e.g., created timestamp, cmd_pose_aa, cmd_gripper_state)

CSV format control:
  --csv_format auto|numeric|frankateach   (default: auto)

Threshold handling:
  --trans_thresh (m), --rot_thresh_deg (deg), --grip_thresh, --on_violate split|skip|stop
"""

import argparse, ast, pickle, sys, time, math, csv, re
from pathlib import Path
import numpy as np

from frankateach.network import create_request_socket
from frankateach.messages import FrankaAction
from frankateach.constants import HOST, CONTROL_PORT

# ---------- Helpers: parsing & math ----------
def parse_aa(s: str) -> np.ndarray:
    """
    Parse FrankaTeach-style stringified axis-angle pose '[np.float64(...), ...]' → ndarray (6,)
    """
    if s is None:
        raise ValueError("cmd_pose_aa is missing in CSV row")
    # Remove 'np.float64(' ... ')' and brackets, then split by comma
    cleaned = re.sub(r'np\.float64\(|\)', '', s).strip('[]')
    arr = np.fromstring(cleaned, sep=',', dtype=np.float32)
    if arr.size != 6:
        raise ValueError(f"cmd_pose_aa should have 6 numbers, got {arr.size}: {s[:80]}...")
    return arr

def axisangle2quat(vec: np.ndarray) -> np.ndarray:
    """scaled axis-angle (ax,ay,az) → quaternion (x,y,z,w)"""
    angle = float(np.linalg.norm(vec))
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = (vec / angle).astype(np.float64)
    half = angle * 0.5
    s = math.sin(half)
    q = np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(half)], dtype=np.float64)
    return q.astype(np.float32)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([0,0,0,1], dtype=np.float64)
    return (q / n).astype(np.float64)

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
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
    # Numeric CSV with 6 or 7 columns, no header
    arr = np.loadtxt(path, delimiter=',')
    return np.asarray(arr)

def _load_frankateach_csv(path: Path, grip_default: float) -> np.ndarray:
    # Parse named CSV with FrankaTeach columns (cmd_pose_aa, cmd_gripper_state)
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header; expected FrankaTeach header for --csv_format frankateach")
        need_pose = None
        # Support both 'cmd_gripper_state' and 'gripper_state' naming
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
    # Peek first line to decide parsing mode
    with open(path, 'r', newline='') as f:
        first = f.readline()
    header = first.strip().lower()
    if ('cmd_pose_aa' in header) or ('created timestamp' in header) or ('cmd_gripper_state' in header):
        return _load_frankateach_csv(path, grip_default)
    # fallback: try numeric
    try:
        return _load_numeric_csv(path)
    except Exception as e:
        raise ValueError(f"CSV auto-detect failed; try --csv_format frankateach or provide numeric CSV. Root cause: {e}")

def _load_from_file(path: Path, csv_format: str, grip_default: float) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == '.npy':
        arr = np.load(path)
    elif suf == '.npz':
        npz = np.load(path)
        key = npz.files[0]
        arr = npz[key]
    elif suf in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        arr = np.asarray(obj)
    elif suf == '.txt':
        arr = np.loadtxt(path)
    elif suf == '.csv':
        if csv_format == 'frankateach':
            arr = _load_frankateach_csv(path, grip_default)
        elif csv_format == 'numeric':
            arr = _load_numeric_csv(path)
        else:
            arr = _load_csv_auto(path, grip_default)
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
            lit = ast.literal_eval(args.actions)
        except Exception as e:
            raise ValueError(f"--actions 解析失败：{e}")
        arr = np.asarray(lit)

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

# ---------- Frame prep & thresholds ----------
def build_frames(actions: np.ndarray):
    frames = []
    for row in actions:
        pos = row[:3].astype(np.float32)
        aa  = row[3:6].astype(np.float64)
        quat = axisangle2quat(aa)
        frames.append({'pos': pos, 'quat': quat, 'grip': float(row[6])})
    return frames

def enforce_thresholds(frames, trans_thresh, rot_thresh_rad, grip_thresh,
                       on_violate='split'):
    if len(frames) <= 1:
        return frames, 0, 0

    out = [frames[0]]
    inserted = 0
    skipped = 0
    def need_check(val): return val is not None and val > 0.0

    for nxt in frames[1:]:
        prv = out[-1]
        d_trans = float(np.linalg.norm(nxt['pos'] - prv['pos']))
        d_rot   = rot_angle_between(prv['quat'], nxt['quat'])
        d_grip  = abs(nxt['grip'] - prv['grip'])

        exceed_trans = need_check(trans_thresh) and (d_trans > trans_thresh)
        exceed_rot   = need_check(rot_thresh_rad) and (d_rot   > rot_thresh_rad)
        exceed_grip  = need_check(grip_thresh) and (d_grip     > grip_thresh)
        if not (exceed_trans or exceed_rot or exceed_grip):
            out.append(nxt)
            continue

        if on_violate == 'skip':
            skipped += 1
            continue
        if on_violate == 'stop':
            raise ValueError(
                f"Step exceeds threshold(s): Δx={d_trans:.4f} m, Δθ={math.degrees(d_rot):.2f}°, Δgrip={d_grip:.4f}"
            )

        # split
        n_div = 1
        if need_check(trans_thresh) and d_trans > 0:
            n_div = max(n_div, int(math.ceil(d_trans / trans_thresh)))
        if need_check(rot_thresh_rad) and d_rot > 0:
            n_div = max(n_div, int(math.ceil(d_rot / rot_thresh_rad)))
        if need_check(grip_thresh) and d_grip > 0:
            n_div = max(n_div, int(math.ceil(d_grip / grip_thresh)))

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

# ---------- Replay ----------
def replay_given(actions: np.ndarray, hz: float, speed: float,
                 trans_thresh: float, rot_thresh_deg: float, grip_thresh: float,
                 on_violate: str):
    if actions.shape[0] < 1:
        print("动作序列为空，退出"); return

    frames = build_frames(actions)
    rot_thresh_rad = math.radians(rot_thresh_deg) if rot_thresh_deg > 0 else 0.0
    frames, inserted, skipped = enforce_thresholds(
        frames, trans_thresh, rot_thresh_rad, grip_thresh, on_violate=on_violate
    )
    print(f"Thresholds applied: inserted {inserted} micro-steps, skipped {skipped} steps, final frames={len(frames)}")

    sock = create_request_socket(HOST, CONTROL_PORT)

    # Move to first frame
    first = frames[0]
    print("Step-0: move to first frame pose …")
    # import ipdb; ipdb.set_trace()  # Debugging point to inspect variables during replay
    sock.send(pickle.dumps(to_action(first['pos'], first['quat'], first['grip']), protocol=-1))
    _ = sock.recv()
    time.sleep(1.0)

    n_frames = len(frames) - 1
    print(f"Start replay, {n_frames} frames, hz={hz} Hz, speed×{speed}")

    dt = 0.0
    if speed > 0 and hz > 0:
        dt = (1.0 / float(hz)) / float(speed)

    t0 = time.perf_counter()

    for i, fr in enumerate(frames[1:], start=1):
        if dt > 0.0:
            target = t0 + i * dt
            while True:
                remain = target - time.perf_counter()
                if remain <= 0:
                    break
                time.sleep(0.001 if remain > 0.01 else 0.0002)

        sock.send(pickle.dumps(to_action(fr['pos'], fr['quat'], fr['grip']), protocol=-1))
        _ = sock.recv()

    print("Replay done")
    sock.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="动作文件路径（.npy/.npz/.pkl/.pickle/.txt/.csv）")
    ap.add_argument("--actions", help="Python 字面量列表/数组字符串，例如 '[[...],[...]]'")
    ap.add_argument("--csv_format", choices=["auto", "numeric", "frankateach"], default="auto",
                    help="CSV 解析方式：auto(自动识别)、numeric(纯数值无表头)、frankateach(带表头)")
    ap.add_argument("--hz", type=float, default=30.0, help="基准频率（Hz），用于定时播放")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="播放速度倍率；0 为尽快播放（无等待）")
    ap.add_argument("--grip_default", type=float, default=0.0,
                    help="当仅提供 6 维动作时，用此常数填充 gripper")

    # Thresholds
    ap.add_argument("--trans_thresh", type=float, default=0.02,
                    help="单步最大平移 (m)；<=0 关闭检查")
    ap.add_argument("--rot_thresh_deg", type=float, default=10.0,
                    help="单步最大旋转角 (度)；<=0 关闭检查")
    ap.add_argument("--grip_thresh", type=float, default=0.0,
                    help="单步最大夹爪变化；<=0 关闭检查")
    ap.add_argument("--on_violate", choices=["split", "skip", "stop"], default="split",
                    help="超过阈值时的处理：split=细分插值(默认), skip=跳过该步, stop=报错退出")

    args = ap.parse_args()

    try:
        actions = _load_actions(args)
        replay_given(actions,
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
