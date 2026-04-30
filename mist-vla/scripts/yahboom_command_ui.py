#!/usr/bin/env python3
"""Small local web UI to send prompt commands to the Yahboom control loop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_model_yahboom_loop.py"
DEFAULT_HOME_COORDS = [130.0, -25.0, 350.0, -92.0, -45.0, -87.0]


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Yahboom Prompt UI</title>
  <style>
    body { font-family: sans-serif; margin: 18px; background: #0f1115; color: #f0f3f8; }
    .row { display: flex; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
    label { display: flex; flex-direction: column; font-size: 13px; gap: 4px; }
    input, select, textarea, button {
      background: #171a21; color: #f0f3f8; border: 1px solid #2d3340; border-radius: 6px; padding: 8px;
    }
    textarea { width: 100%; min-height: 62px; }
    button { cursor: pointer; }
    #logs { background: #0b0d12; border: 1px solid #2d3340; border-radius: 6px; padding: 10px; height: 360px; overflow: auto; white-space: pre-wrap; }
    .ok { color: #67d17a; }
    .bad { color: #ff7b72; }
  </style>
</head>
<body>
  <h2>Yahboom Prompt Control UI</h2>
  <p>Send natural-language commands to OpenVLA and execute on robot.</p>

  <div class="row">
    <label style="flex: 1 1 520px;">
      Prompt
      <textarea id="instruction">pick up yellow cube</textarea>
    </label>
  </div>

  <div class="row">
    <label>Policy
      <select id="policy">
        <option value="openvla">openvla</option>
        <option value="openvla_oft">openvla_oft</option>
        <option value="smolvla">smolvla</option>
      </select>
    </label>
    <label>Model Name <input id="model_name" value="openvla/openvla-7b" /></label>
    <label>Jetson Host <input id="jetson_host" value="192.168.55.1" /></label>
    <label>Jetson Port <input id="jetson_port" type="number" value="5000" /></label>
  </div>

  <div class="row">
    <label>Steps (0 = run until Stop) <input id="steps" type="number" value="0" /></label>
    <label>dt <input id="dt" type="number" step="0.01" value="0.03" /></label>
    <label>Speed <input id="speed" type="number" value="40" /></label>
    <label>xyz gain <input id="xyz_gain" type="number" step="0.01" value="0.2" /></label>
    <label>rot gain <input id="rot_gain" type="number" step="0.01" value="0.3" /></label>
  </div>

  <div class="row">
    <label><input id="execute" type="checkbox" checked /> Execute on robot</label>
    <label><input id="unsafe" type="checkbox" /> Unsafe mode (no constraints)</label>
    <label><input id="disable_steering" type="checkbox" checked /> Disable steering</label>
    <label><input id="wait_for_motion" type="checkbox" /> Wait for each motion (slower)</label>
    <label>Device <input id="device" value="cuda" /></label>
    <label>Device map <input id="device_map" value="auto" /></label>
  </div>

  <div class="row">
    <button onclick="startRun()">Start Run</button>
    <button onclick="stopRun()">Stop Run</button>
    <button onclick="homeRobot()">Home</button>
    <button onclick="clearLogs()">Clear Logs</button>
    <span id="status" style="margin-left: 8px;"></span>
  </div>

  <h3>Live Output</h3>
  <div id="logs"></div>

  <script>
    let cursor = 0;
    async function api(path, method="GET", body=null) {
      const res = await fetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : null
      });
      return await res.json();
    }
    function readValue(id, parseFloatLike=false) {
      const v = document.getElementById(id).value;
      return parseFloatLike ? Number(v) : v;
    }
    function isChecked(id) { return document.getElementById(id).checked; }

    async function startRun() {
      const payload = {
        instruction: readValue("instruction"),
        policy: readValue("policy"),
        model_name: readValue("model_name"),
        jetson_host: readValue("jetson_host"),
        jetson_port: Number(readValue("jetson_port")),
        steps: Number(readValue("steps")),
        dt: Number(readValue("dt")),
        speed: Number(readValue("speed")),
        xyz_gain: Number(readValue("xyz_gain")),
        rot_gain: Number(readValue("rot_gain")),
        execute: isChecked("execute"),
        unsafe_no_constraints: isChecked("unsafe"),
        disable_steering: isChecked("disable_steering"),
        wait_for_motion: isChecked("wait_for_motion"),
        device: readValue("device"),
        device_map: readValue("device_map")
      };
      const out = await api("/api/start", "POST", payload);
      if (!out.ok) {
        alert("Start failed: " + out.error);
      }
    }
    async function stopRun() {
      await api("/api/stop", "POST", {});
    }
    async function homeRobot() {
      const payload = {
        jetson_host: readValue("jetson_host"),
        jetson_port: Number(readValue("jetson_port")),
        speed: Number(readValue("speed"))
      };
      const out = await api("/api/home", "POST", payload);
      if (!out.ok) {
        alert("Home failed: " + out.error);
      }
    }
    async function poll() {
      const st = await api("/api/state");
      const status = document.getElementById("status");
      status.className = st.running ? "ok" : "bad";
      status.textContent = st.running ? ("Running (pid " + st.pid + ")") : ("Idle" + (st.exit_code === null ? "" : ", exit=" + st.exit_code));
      const lg = await api("/api/logs?since=" + cursor);
      if (lg.ok && lg.lines.length) {
        cursor = lg.next_cursor;
        const box = document.getElementById("logs");
        for (const ln of lg.lines) {
          box.textContent += ln + "\\n";
        }
        box.scrollTop = box.scrollHeight;
      }
    }
    function clearLogs() {
      document.getElementById("logs").textContent = "";
      cursor = 0;
      api("/api/clear_logs", "POST", {});
    }
    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>
"""


@dataclass
class RunState:
    proc: Optional[subprocess.Popen] = None
    running: bool = False
    exit_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line.rstrip("\n"))
            # Cap to avoid unbounded memory growth.
            if len(self.logs) > 5000:
                self.logs = self.logs[-3000:]

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()


STATE = RunState()


def robot_action(base: str, payload: Dict[str, Any], timeout_s: float = 25.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}/action",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def robot_status(base: str, timeout_s: float = 8.0) -> Dict[str, Any]:
    with request.urlopen(f"{base}/status", timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_command(payload: Dict[str, Any]) -> List[str]:
    raw_steps = int(payload.get("steps", 0))
    steps = 1000000 if raw_steps <= 0 else raw_steps
    cmd = [
        sys.executable,
        str(RUN_LOOP_SCRIPT),
        "--policy",
        str(payload.get("policy", "openvla")),
        "--model-name",
        str(payload.get("model_name", "openvla/openvla-7b")),
        "--device",
        str(payload.get("device", "cuda")),
        "--device-map",
        str(payload.get("device_map", "auto")),
        "--instruction",
        str(payload.get("instruction", "pick up yellow cube")),
        "--steps",
        str(steps),
        "--dt",
        str(float(payload.get("dt", 0.7))),
        "--jetson-host",
        str(payload.get("jetson_host", "192.168.55.1")),
        "--jetson-port",
        str(int(payload.get("jetson_port", 5000))),
        "--action-space",
        "meters",
        "--xyz-gain",
        str(float(payload.get("xyz_gain", 0.2))),
        "--rot-gain",
        str(float(payload.get("rot_gain", 0.3))),
        "--speed",
        str(int(payload.get("speed", 10))),
        "--skip-redundant-gripper",
    ]
    if bool(payload.get("wait_for_motion", False)):
        cmd.append("--wait-for-motion")
    if bool(payload.get("disable_steering", True)):
        cmd.append("--disable-steering")
    if bool(payload.get("unsafe_no_constraints", False)):
        cmd.append("--unsafe-no-constraints")
    if bool(payload.get("execute", True)):
        cmd.append("--execute")
    return cmd


def start_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    with STATE.lock:
        if STATE.running and STATE.proc is not None and STATE.proc.poll() is None:
            return {"ok": False, "error": "Run already in progress"}
        STATE.logs.clear()
        STATE.exit_code = None
        cmd = build_command(payload)
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        STATE.proc = proc
        STATE.running = True
    STATE.append_log("$ " + " ".join(cmd))

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            STATE.append_log(line.rstrip("\n"))
        proc.stdout.close()

    def _waiter() -> None:
        rc = proc.wait()
        with STATE.lock:
            STATE.running = False
            STATE.exit_code = int(rc)
            STATE.proc = None
        STATE.append_log(f"[ui] process exited rc={rc}")

    threading.Thread(target=_reader, daemon=True).start()
    threading.Thread(target=_waiter, daemon=True).start()
    return {"ok": True, "pid": proc.pid, "cmd": cmd}


def stop_run() -> Dict[str, Any]:
    with STATE.lock:
        proc = STATE.proc
    if proc is None or proc.poll() is not None:
        return {"ok": True, "stopped": False}
    proc.terminate()
    try:
        proc.wait(timeout=4.0)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"ok": True, "stopped": True}


def home_robot(payload: Dict[str, Any]) -> Dict[str, Any]:
    with STATE.lock:
        if STATE.running and STATE.proc is not None and STATE.proc.poll() is None:
            return {"ok": False, "error": "Cannot home while a run is in progress. Stop run first."}
    host = str(payload.get("jetson_host", "192.168.55.1"))
    port = int(payload.get("jetson_port", 5000))
    speed = int(payload.get("speed", 10))
    base = f"http://{host}:{port}"
    try:
        rm = robot_action(
            base,
            {"action": "move_to", "coords": DEFAULT_HOME_COORDS, "speed": speed, "wait": True},
            timeout_s=40.0,
        )
        rs = robot_status(base)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    STATE.append_log(f"[ui] home move: ok={rm.get('ok')} err={rm.get('error')} final={rm.get('final_coords')}")
    return {"ok": bool(rm.get("ok")), "move_to": rm, "status": rs}


class Handler(BaseHTTPRequestHandler):
    server_version = "YahboomPromptUI/1.0"

    def _json(self, obj: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text: str, status: int = 200, ctype: str = "text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        raw = self.rfile.read(n).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._text(HTML)
            return
        if parsed.path == "/api/state":
            with STATE.lock:
                pid = None if STATE.proc is None else STATE.proc.pid
                out = {
                    "ok": True,
                    "running": bool(STATE.running),
                    "pid": pid,
                    "exit_code": STATE.exit_code,
                    "log_count": len(STATE.logs),
                }
            self._json(out)
            return
        if parsed.path == "/api/logs":
            qs = parse_qs(parsed.query)
            try:
                since = int(qs.get("since", ["0"])[0])
            except Exception:
                since = 0
            with STATE.lock:
                since = max(0, min(since, len(STATE.logs)))
                lines = STATE.logs[since:]
                next_cursor = len(STATE.logs)
            self._json({"ok": True, "lines": lines, "next_cursor": next_cursor})
            return
        self._json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except Exception as e:
            self._json({"ok": False, "error": f"invalid json: {e}"}, status=400)
            return
        if parsed.path == "/api/start":
            try:
                self._json(start_run(payload), status=200)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, status=500)
            return
        if parsed.path == "/api/stop":
            self._json(stop_run(), status=200)
            return
        if parsed.path == "/api/home":
            self._json(home_robot(payload), status=200)
            return
        if parsed.path == "/api/clear_logs":
            STATE.clear_logs()
            self._json({"ok": True}, status=200)
            return
        self._json({"ok": False, "error": "not found"}, status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep stdout clean for run logs.
        return


def main() -> int:
    ap = argparse.ArgumentParser(description="Local web UI for Yahboom prompt control")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not RUN_LOOP_SCRIPT.exists():
        print(f"Missing script: {RUN_LOOP_SCRIPT}", file=sys.stderr)
        return 2

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[ui] serving http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        _ = stop_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

