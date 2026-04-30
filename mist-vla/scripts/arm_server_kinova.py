#!/usr/bin/env python3
"""
Arm Server for Kinova Gen3 Lite
================================
Same HTTP API as arm_server_xarm.py.

Install:
  pip install flask opencv-python
  # Kinova Kortex API: download from Kinova's GitHub or pip install kortex-api

Run:
  python arm_server_kinova.py --ip 192.168.1.xxx --camera 0 --port 5000
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

arm = None
cap = None
cap_lock = threading.Lock()


def grab_frame():
    with cap_lock:
        ret, frame = cap.read()
    return frame if ret else None


@app.route("/snapshot", methods=["GET"])
def snapshot():
    frame = grab_frame()
    if frame is None:
        return "camera error", 500
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/status", methods=["GET", "POST"])
def status():
    try:
        from kortex_api.autogen.messages import Base_pb2
        feedback = arm.GetMeasuredCartesianPose()
        pose = [feedback.x, feedback.y, feedback.z,
                feedback.theta_x, feedback.theta_y, feedback.theta_z]
        angles_feedback = arm.GetMeasuredJointAngles()
        angles = [jt.value for jt in angles_feedback.joint_angles]
        return jsonify({
            "ok": True,
            "coords": [float(v) for v in pose],
            "angles": [float(v) for v in angles],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/action", methods=["POST"])
def action():
    data = request.json
    if not data:
        return jsonify({"ok": False, "error": "no JSON body"})

    act = data.get("action", "")
    t0 = time.time()

    try:
        from kortex_api.autogen.messages import Base_pb2

        if act == "status":
            feedback = arm.GetMeasuredCartesianPose()
            pose = [feedback.x, feedback.y, feedback.z,
                    feedback.theta_x, feedback.theta_y, feedback.theta_z]
            return jsonify({"ok": True, "coords": [float(v) for v in pose]})

        elif act == "home":
            # Move to home position using angular action
            action_msg = Base_pb2.Action()
            action_msg.name = "Home"
            action_msg.application_data = ""
            # Use Kinova's built-in home
            arm.ReadAllSequences()  # dummy call to verify connection
            # Simplified: move to a known safe position
            return _move_cartesian(200, 0, 300, 180, 0, 0, speed=50, t0=t0)

        elif act == "move_to":
            coords = data.get("coords", [])
            if len(coords) != 6:
                return jsonify({"ok": False, "error": f"need 6 coords"})
            speed = int(data.get("speed", 100))
            x, y, z, rx, ry, rz = [float(v) for v in coords]
            return _move_cartesian(x, y, z, rx, ry, rz, speed, t0)

        elif act == "set_gripper":
            value = int(data.get("value", 50))
            speed = float(data.get("speed", 1.0))
            gripper_command = Base_pb2.GripperCommand()
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            finger = gripper_command.gripper.finger.add()
            finger.finger_identifier = 0
            finger.value = float(value) / 100.0  # 0.0 to 1.0
            arm.SendGripperCommand(gripper_command)
            time.sleep(1.0)  # Wait for gripper
            return jsonify({"ok": True, "elapsed_ms": int((time.time() - t0) * 1000)})

        else:
            return jsonify({"ok": False, "error": f"unknown action: {act}"})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


def _move_cartesian(x, y, z, rx, ry, rz, speed, t0):
    from kortex_api.autogen.messages import Base_pb2

    action_msg = Base_pb2.Action()
    action_msg.name = "MoveTo"
    action_msg.application_data = ""

    cartesian = action_msg.reach_pose.target_pose
    cartesian.x = float(x) / 1000.0  # Kinova uses meters
    cartesian.y = float(y) / 1000.0
    cartesian.z = float(z) / 1000.0
    cartesian.theta_x = float(rx)
    cartesian.theta_y = float(ry)
    cartesian.theta_z = float(rz)

    arm.ExecuteAction(action_msg)
    time.sleep(0.5)  # Wait for motion

    # Read final
    feedback = arm.GetMeasuredCartesianPose()
    final = [feedback.x * 1000, feedback.y * 1000, feedback.z * 1000,
             feedback.theta_x, feedback.theta_y, feedback.theta_z]
    return jsonify({
        "ok": True,
        "final_coords": [float(v) for v in final],
        "elapsed_ms": int((time.time() - t0) * 1000),
    })


def main():
    global arm, cap

    parser = argparse.ArgumentParser(description="Kinova Gen3 Lite HTTP API Server")
    parser.add_argument("--ip", required=True, help="Kinova arm IP")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # Connect to Kinova
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.messages import Session_pb2
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

    transport = TCPTransport()
    transport.connect(args.ip, 10000)
    router = RouterClient(transport, lambda e: print(f"Error: {e}"))
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = "admin"
    session_info.password = "admin"
    session_info.session_inactivity_timeout = 60000
    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)
    arm = BaseClient(router)
    print(f"Connected to Kinova at {args.ip}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera: {int(cap.get(3))}x{int(cap.get(4))}")

    print(f"Server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
