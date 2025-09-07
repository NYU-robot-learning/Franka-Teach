#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, math, pickle, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from frankateach.network import create_request_socket
from frankateach.messages import FrankaAction
from frankateach.constants import HOST, CONTROL_PORT


# ---------- math / quat ----------
def axisangle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, np.float64)
    ang = float(np.linalg.norm(aa))
    if ang < 1e-12:
        return np.array([0, 0, 0, 1], np.float32)
    axis = aa / ang
    s = math.sin(ang / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(ang/2.0)], np.float32)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([0,0,0,1], np.float64)
    return (q / n).astype(np.float64)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    q = np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], np.float64)
    return quat_normalize(q).astype(np.float32)


# ---------- robot I/O ----------
def to_action(pos: np.ndarray, quat: np.ndarray, grip: float) -> FrankaAction:
    return FrankaAction(
        pos=np.asarray(pos, np.float32),
        quat=np.asarray(quat, np.float32),
        gripper=float(grip),
        reset=False,
        timestamp=time.time(),
    )

def send_and_recv(sock, pos, quat, grip, reset=False):
    cmd = FrankaAction(
        pos=np.asarray(pos, np.float32),
        quat=np.asarray(quat, np.float32),
        gripper=float(grip),
        reset=reset,
        timestamp=time.time(),
    )
    sock.send(pickle.dumps(cmd, protocol=-1))
    reply = sock.recv()  # 这里可以扩展解析返回状态
    return reply


# ---------- CSV loading ----------
Row = Dict[str, float]

def _float(row: dict, key: str) -> float:
    v = row.get(key, None)
    if v is None or v == "":
        raise ValueError(f"CSV missing value for column '{key}'")
    return float(v)

def load_delta_csv(path: Path,
                   episodes_filter: List[int] | None
                   ) -> Dict[int, List[Row]]:
    """
    返回 {episode_index: [rows...]}，每个 row 含 st_* 与 act_*，按 frame_index 排序
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        need = {"episode_index","frame_index",
                "st_x","st_y","st_z","st_aax","st_aay","st_aaz","st_grip",
                "act_dx","act_dy","act_dz","act_daax","act_daay","act_daaz","act_grip"}
        if not need.issubset(set(h.lower() for h in reader.fieldnames or [])):
            # 容忍大小写；构造一个小写映射
            # 重新创建 reader 并把键统一为小写
            f.seek(0); reader = csv.DictReader(f)
        groups: Dict[int, List[Row]] = defaultdict(list)
        for raw in reader:
            # normalize keys to lower
            row = {k.lower(): v for k, v in raw.items()}
            ep = int(_float(row, "episode_index"))
            if (episodes_filter is not None) and (ep not in episodes_filter):
                continue
            parsed: Row = {
                "episode_index": ep,
                "frame_index": int(_float(row, "frame_index")),
                # seed absolute state
                "st_x": _float(row, "st_x"),
                "st_y": _float(row, "st_y"),
                "st_z": _float(row, "st_z"),
                "st_aax": _float(row, "st_aax"),
                "st_aay": _float(row, "st_aay"),
                "st_aaz": _float(row, "st_aaz"),
                "st_grip": _float(row, "st_grip"),
                # delta action
                "act_dx": _float(row, "act_dx"),
                "act_dy": _float(row, "act_dy"),
                "act_dz": _float(row, "act_dz"),
                "act_daax": _float(row, "act_daax"),
                "act_daay": _float(row, "act_daay"),
                "act_daaz": _float(row, "act_daaz"),
                "act_grip": _float(row, "act_grip"),
            }
            groups[ep].append(parsed)
        # sort by frame_index
        for ep, lst in groups.items():
            lst.sort(key=lambda r: r["frame_index"])
        return dict(sorted(groups.items(), key=lambda kv: kv[0]))


# ---------- delta thresholding / splitting ----------
def split_delta(dpos: np.ndarray, daa: np.ndarray,
                trans_thresh: float,
                rot_thresh_rad: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    把单步 delta 拆分为多小步（线性 + 轴角等分）以满足阈值；返回 [(dpos_i, daa_i), ...]
    """
    dpos = np.asarray(dpos, np.float32)
    daa  = np.asarray(daa,  np.float32)
    n_div = 1
    if trans_thresh and trans_thresh > 0:
        n_div = max(n_div, int(math.ceil(np.linalg.norm(dpos) / trans_thresh))) if np.linalg.norm(dpos) > 0 else n_div
    if rot_thresh_rad and rot_thresh_rad > 0:
        ang = float(np.linalg.norm(daa))
        n_div = max(n_div, int(math.ceil(ang / rot_thresh_rad))) if ang > 0 else n_div
    if n_div <= 1:
        return [(dpos, daa)]
    frac = 1.0 / float(n_div)
    return [(dpos * frac, daa * frac) for _ in range(n_div)]


# ---------- main replay ----------
def main():
    ap = argparse.ArgumentParser(description="Replay delta-action CSV (compose deltas onto last commanded pose).")
    ap.add_argument("--csv", required=True, help="path to CSV with st_* and act_* columns")
    ap.add_argument("--episodes", type=str, default="",
                    help="comma/range list, e.g. '12' or '1,3,5-7'; default: all episodes")
    ap.add_argument("--hz", type=float, default=30.0, help="command rate (Hz)")
    ap.add_argument("--speed", type=float, default=1.0, help="time scaling; 0 = as fast as possible")
    ap.add_argument("--grip_mode", choices=["abs","delta","freeze"], default="abs",
                    help="interpretation of act_grip column")
    # per-step delta thresholds (before发送)
    ap.add_argument("--delta_thresh_trans", type=float, default=0.02,
                    help="max ‖Δpos‖ per step (meters); <=0 disables")
    ap.add_argument("--delta_thresh_rot_deg", type=float, default=10.0,
                    help="max |Δθ| per step (degrees); <=0 disables")
    ap.add_argument("--on_violate", choices=["split","skip","stop"], default="split",
                    help="when a delta exceeds threshold")
    args = ap.parse_args()

    # parse episodes filter
    episodes_filter: List[int] | None = None
    if args.episodes.strip():
        s = args.episodes.replace(" ", "")
        eps: set[int] = set()
        for token in s.split(","):
            if "-" in token:
                a, b = token.split("-")
                a, b = int(a), int(b)
                for e in range(min(a,b), max(a,b)+1): eps.add(e)
            else:
                eps.add(int(token))
        episodes_filter = sorted(list(eps))

    groups = load_delta_csv(Path(args.csv), episodes_filter)
    if not groups:
        print("CSV 没有匹配的 episode，退出"); return

    sock = create_request_socket(HOST, CONTROL_PORT)

    # pacing
    base_dt = 0.0 if args.speed == 0 or args.hz <= 0 else (1.0 / args.hz) / max(1e-9, args.speed)
    step_counter = 0
    t0 = time.perf_counter()

    rot_thresh_rad = math.radians(args.delta_thresh_rot_deg) if args.delta_thresh_rot_deg > 0 else 0.0
    print(f"[INFO] Episodes to replay: {list(groups.keys())}")
    print(f"[INFO] hz={args.hz}, speed×{args.speed}, base_dt={base_dt:.4f}s, Δpos_thresh={args.delta_thresh_trans}m, Δrot_thresh={args.delta_thresh_rot_deg}° ({rot_thresh_rad:.4f} rad)")

    try:
        for ep, rows in groups.items():
            if not rows:
                continue
            # seed pose from the first row's st_*
            seed = rows[0]
            pos_cmd  = np.array([seed["st_x"], seed["st_y"], seed["st_z"]], np.float32)
            quat_cmd = axisangle_to_quat(np.array([seed["st_aax"], seed["st_aay"], seed["st_aaz"]], np.float32))
            grip_cmd = float(seed["st_grip"])

            print(f"\n[EP {ep}] move to first state: pos={np.round(pos_cmd,4)} aa≈{np.round([seed['st_aax'],seed['st_aay'],seed['st_aaz']],4)} grip={grip_cmd:.3f}")
            send_and_recv(sock, pos_cmd, quat_cmd, grip_cmd, reset=False)
            time.sleep(0.8)

            skipped = 0
            inserted = 0

            for row in rows:
                dpos = np.array([row["act_dx"], row["act_dy"], row["act_dz"]], np.float32)
                daa  = np.array([row["act_daax"], row["act_daay"], row["act_daaz"]], np.float32)

                # threshold handling
                need_split = False
                if args.delta_thresh_trans > 0 and np.linalg.norm(dpos) > args.delta_thresh_trans:
                    need_split = True
                if rot_thresh_rad > 0 and np.linalg.norm(daa) > rot_thresh_rad:
                    need_split = True

                if need_split and args.on_violate == "skip":
                    skipped += 1
                    continue
                if need_split and args.on_violate == "stop":
                    raise ValueError(f"Δ exceeds threshold: ‖Δx‖={np.linalg.norm(dpos):.4f}m, |Δθ|={np.linalg.norm(daa):.4f}rad")

                parts = split_delta(dpos, daa,
                                    trans_thresh=args.delta_thresh_trans if args.on_violate=="split" else 0.0,
                                    rot_thresh_rad=rot_thresh_rad if args.on_violate=="split" else 0.0)

                if len(parts) > 1:
                    inserted += (len(parts) - 1)

                for dpos_i, daa_i in parts:
                    # integrate
                    pos_cmd = (pos_cmd + dpos_i).astype(np.float32)
                    dquat   = axisangle_to_quat(daa_i)
                    quat_cmd = quat_mul(quat_cmd, dquat)

                    # gripper
                    if args.grip_mode == "abs":
                        grip_cmd = float(row["act_grip"])
                    elif args.grip_mode == "delta":
                        grip_cmd = float(grip_cmd + row["act_grip"])
                    elif args.grip_mode == "freeze":
                        grip_cmd = float(grip_cmd)
                    else:
                        raise ValueError("bad grip_mode")

                    # pacing
                    if base_dt > 0.0:
                        step_counter += 1
                        target = t0 + step_counter * base_dt
                        while True:
                            remain = target - time.perf_counter()
                            if remain <= 0: break
                            time.sleep(0.001 if remain > 0.01 else 0.0002)

                    # send
                    send_and_recv(sock, pos_cmd, quat_cmd, grip_cmd, reset=False)

            print(f"[EP {ep}] done. inserted micro-steps={inserted}, skipped steps={skipped}")

        print("\n[INFO] Replay finished.")
    except KeyboardInterrupt:
        print("\n[INFO] 用户终止")
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
