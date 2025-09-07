#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay FrankaTeach state.csv ，先定位到首帧，再按时间戳播放
"""

import argparse, ast, csv, pickle, sys, time
import numpy as np
import re
import math
from pathlib import Path
from frankateach.network import create_request_socket
from frankateach.messages import FrankaAction
from frankateach.constants import HOST, CONTROL_PORT

# ---------- 辅助函数 ----------
def parse_aa(s: str) -> np.ndarray:
    """
    '"[np.float64(0.5), np.float64(-0.1), ...]"'  →  ndarray shape (6,)
    """
    # 1) 去掉 np.float64( … ) 以及方括号
    cleaned = re.sub(r'np\.float64\(|\)', '', s).strip('[]')
    # 2) 直接按逗号分割成浮点数
    return np.fromstring(cleaned, sep=',', dtype=np.float32)

def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q

def aa_to_action(pos_aa, grip):
    pos  = pos_aa[:3]
    axis = pos_aa[3:]
    quat = (
        np.array([1, 0, 0, 0], dtype=np.float32)
        if np.linalg.norm(axis) < 1e-6
        else axisangle2quat(axis.astype(np.float64)).astype(np.float32)
    )
    return FrankaAction(pos=pos, quat=quat, gripper=float(grip),
                        reset=False, timestamp=time.time())

# ---------- 主函数 ----------
def replay(csv_path, speed):
    sock = create_request_socket(HOST, CONTROL_PORT)

    # 读取 CSV
    traj = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            ts   = float(row['created timestamp'])
            aa   = parse_aa(row['cmd_pose_aa'])
            grip = float(row['cmd_gripper_state'])
            traj.append((ts, aa, grip))
    if not traj:
        print("CSV 为空，退出"); return

    # ---------- 先把机器人送到首帧 ----------
    first_ts, first_aa, first_g = traj[0]
    print("Step‑0: move to first frame pose …")
    sock.send(pickle.dumps(aa_to_action(first_aa, first_g), protocol=-1))
    _ = sock.recv()           # 读取 FrankaState，可扩展检查是否到位
    time.sleep(1.0)           # 简易等待；换成轮询到位更严谨

    # ---------- 再开始正式回放 ----------
    print(f"Start replay, {len(traj)-1} frames, speed×{speed}")
    t0_real = time.time()
    t0_csv  = first_ts

    for ts_csv, aa, g in traj[1:]:          # 从第 2 帧开始
        if speed > 0:
            dt = (ts_csv - t0_csv) / speed
            while time.time() - t0_real < dt:
                time.sleep(0.2)

        sock.send(pickle.dumps(aa_to_action(aa, g), protocol=-1))
        _ = sock.recv()                     # 可存包或检查误差

    print("Replay done")
    sock.close()

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="state.csv 路径")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="播放速度倍率，0 为尽快播放")
    args = ap.parse_args()
    try:
        replay(args.csv, args.speed)
    except KeyboardInterrupt:
        sys.exit("\n用户终止")
