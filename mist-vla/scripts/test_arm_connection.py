#!/usr/bin/env python3
"""
Quick test: verify arm server is reachable and working.

Usage:
  python test_arm_connection.py --host http://192.168.1.xxx:5000
"""

import argparse
import json
import sys
import time
from urllib import request
from PIL import Image
import io


def api_get(base, path, timeout=5):
    with request.urlopen(f"{base}{path}", timeout=timeout) as resp:
        return resp.read()


def api_post(base, payload, timeout=10):
    data = json.dumps(payload).encode()
    req = request.Request(
        f"{base}/action", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="e.g. http://192.168.1.100:5000")
    args = parser.parse_args()
    base = args.host.rstrip("/")

    print(f"Testing {base}...")
    passed = 0
    failed = 0

    # 1. Status
    print("\n[1] GET /status")
    try:
        raw = api_get(base, "/status")
        st = json.loads(raw.decode())
        print(f"  ok={st['ok']}, coords={st.get('coords')}")
        if st["ok"] and len(st.get("coords", [])) == 6:
            print("  PASS")
            passed += 1
        else:
            print("  FAIL: bad status response")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 2. Snapshot
    print("\n[2] GET /snapshot")
    try:
        img_bytes = api_get(base, "/snapshot")
        img = Image.open(io.BytesIO(img_bytes))
        print(f"  Image: {img.size[0]}x{img.size[1]}, {len(img_bytes)} bytes")
        if img.size[0] > 0:
            print("  PASS")
            passed += 1
        else:
            print("  FAIL: empty image")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 3. Status via POST
    print("\n[3] POST /action {status}")
    try:
        r = api_post(base, {"action": "status"})
        print(f"  ok={r['ok']}, coords={r.get('coords')}")
        if r["ok"]:
            print("  PASS")
            passed += 1
        else:
            print(f"  FAIL: {r.get('error')}")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 4. Small move test (optional)
    print("\n[4] Small move test (will move arm 5mm up and back)")
    try:
        st = json.loads(api_get(base, "/status").decode())
        coords = st["coords"]
        print(f"  Current: {[f'{v:.1f}' for v in coords]}")

        # Move 5mm up
        target = list(coords)
        target[2] += 5.0
        r = api_post(base, {"action": "move_to", "coords": target, "speed": 30, "wait": True})
        print(f"  Move up: ok={r['ok']}, elapsed={r.get('elapsed_ms')}ms")

        time.sleep(0.5)

        # Move back
        r = api_post(base, {"action": "move_to", "coords": coords, "speed": 30, "wait": True})
        print(f"  Move back: ok={r['ok']}, elapsed={r.get('elapsed_ms')}ms")

        if r["ok"]:
            print("  PASS")
            passed += 1
        else:
            print(f"  FAIL: {r.get('error')}")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed. Ready for PULSE deployment.")
    else:
        print("Fix failures before running experiments.")


if __name__ == "__main__":
    main()
