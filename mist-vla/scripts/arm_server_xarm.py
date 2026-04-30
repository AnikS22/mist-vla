#!/usr/bin/env python3
"""
Arm Server for UFactory xArm 6
===============================
Exposes the same HTTP API that run_model_yahboom_loop.py expects:
  GET  /status   → {ok, coords: [x,y,z,rx,ry,rz], angles: [...]}
  POST /action   → handles move_to, set_gripper, home, status
  GET  /snapshot  → JPEG bytes from camera

Install:
  pip install flask xArm-Python-SDK opencv-python

Run:
  python arm_server_xarm.py --ip 192.168.1.xxx --camera 0 --port 5000
"""

import argparse
import io
import json
import time
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# Globals set in main()
arm = None
cap = None
cap_lock = threading.Lock()


# ─── Camera ───────────────────────────────────────────────────────

def grab_frame():
    with cap_lock:
        ret, frame = cap.read()
    if not ret:
        return None
    return frame


@app.route("/snapshot", methods=["GET"])
def snapshot():
    frame = grab_frame()
    if frame is None:
        return "camera error", 500
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


# ─── Status ───────────────────────────────────────────────────────

@app.route("/status", methods=["GET", "POST"])
def status():
    try:
        code, pose = arm.get_position()
        if code != 0:
            return jsonify({"ok": False, "error": f"get_position code {code}"})
        # pose = [x, y, z, roll, pitch, yaw] in mm and degrees
        code2, angles = arm.get_servo_angle()
        return jsonify({
            "ok": True,
            "coords": [float(v) for v in pose],
            "angles": [float(v) for v in angles] if code2 == 0 else [],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ─── Actions ──────────────────────────────────────────────────────

@app.route("/action", methods=["POST"])
def action():
    data = request.json
    if not data:
        return jsonify({"ok": False, "error": "no JSON body"})

    act = data.get("action", "")
    t0 = time.time()

    try:
        if act == "status":
            code, pose = arm.get_position()
            return jsonify({
                "ok": code == 0,
                "coords": [float(v) for v in pose] if code == 0 else [],
            })

        elif act == "home":
            # Move to a safe home position
            code = arm.set_position(
                x=200, y=0, z=300, roll=180, pitch=0, yaw=0,
                speed=50, wait=True
            )
            return jsonify({"ok": code == 0, "elapsed_ms": int((time.time() - t0) * 1000)})

        elif act == "move_to":
            coords = data.get("coords", [])
            if len(coords) != 6:
                return jsonify({"ok": False, "error": f"coords must be 6 values, got {len(coords)}"})
            speed = int(data.get("speed", 100))
            wait = data.get("wait", True)
            if isinstance(wait, (int, float)):
                wait_val = wait > 0
            else:
                wait_val = bool(wait)

            x, y, z, rx, ry, rz = [float(v) for v in coords]
            code = arm.set_position(
                x=x, y=y, z=z, roll=rx, pitch=ry, yaw=rz,
                speed=speed, wait=wait_val
            )
            # Read final position
            _, final = arm.get_position()
            elapsed = int((time.time() - t0) * 1000)
            return jsonify({
                "ok": code == 0,
                "final_coords": [float(v) for v in final] if final else [],
                "elapsed_ms": elapsed,
            })

        elif act == "set_gripper":
            value = int(data.get("value", 50))
            speed = int(data.get("speed", 2000))
            # xArm gripper: 0 = fully closed, 850 = fully open
            # Map 0-100 to 0-850
            pos = int(value * 8.5)
            code = arm.set_gripper_position(pos, speed=speed, wait=True)
            return jsonify({
                "ok": code == 0,
                "elapsed_ms": int((time.time() - t0) * 1000),
            })

        elif act == "open_gripper":
            code = arm.set_gripper_position(850, speed=2000, wait=True)
            return jsonify({"ok": code == 0})

        elif act == "close_gripper":
            code = arm.set_gripper_position(0, speed=2000, wait=True)
            return jsonify({"ok": code == 0})

        else:
            return jsonify({"ok": False, "error": f"unknown action: {act}"})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": int((time.time() - t0) * 1000)})


# ─── Main ─────────────────────────────────────────────────────────

def main():
    global arm, cap

    parser = argparse.ArgumentParser(description="xArm 6 HTTP API Server")
    parser.add_argument("--ip", required=True, help="xArm IP address (e.g. 192.168.1.xxx)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    args = parser.parse_args()

    # Connect to xArm
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)  # Position control mode
    arm.set_state(state=0)  # Sport state
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    print(f"Connected to xArm at {args.ip}")

    # Get initial position
    code, pose = arm.get_position()
    if code == 0:
        print(f"Current position: {[f'{v:.1f}' for v in pose]}")
    else:
        print(f"WARNING: get_position returned code {code}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"WARNING: Camera {args.camera} not opened")
    else:
        print(f"Camera {args.camera} opened: {int(cap.get(3))}x{int(cap.get(4))}")

    print(f"Starting server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
