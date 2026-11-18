"""
Capture 3 high-resolution live demo screenshots from Streamlit:
1. docs/demo_screenshot_folium_route_map.png (Tab 1: Folium Map with live prediction)
2. docs/demo_screenshot_live_trip_metrics.png (Tab 2: Live trip metrics from Omen)
3. docs/demo_screenshot_final_97pct_results.png (Tab 3: Final 97% results & progression)
"""

import os
import sys
import time
import json
import base64
import subprocess
import urllib.request
import websocket

def run_capture():
    print("Launching Chrome headless with remote debugging...")
    proc = subprocess.Popen([
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        "--window-size=1440,1050",
        "http://localhost:8501"
    ])
    
    time.sleep(3)
    ws = None
    try:
        targets_res = urllib.request.urlopen("http://localhost:9222/json")
        targets = json.load(targets_res)
        streamlit_target = None
        for t in targets:
            if "8501" in t.get("url", ""):
                streamlit_target = t
                break
        
        if not streamlit_target:
            print("ERROR: Streamlit target not found!")
            return False

        ws_url = streamlit_target["webSocketDebuggerUrl"]
        print(f"Connecting to CDP WebSocket: {ws_url}")
        ws = websocket.create_connection(ws_url, timeout=10)

        msg_id = 0
        def cdp_send(method, params=None):
            nonlocal msg_id
            msg_id += 1
            payload = {"id": msg_id, "method": method}
            if params:
                payload["params"] = params
            ws.send(json.dumps(payload))
            while True:
                res = json.loads(ws.recv())
                if res.get("id") == msg_id:
                    return res

        # Enable Page and Runtime
        cdp_send("Page.enable")
        cdp_send("Runtime.enable")

        print("Waiting for Streamlit page to load...")
        time.sleep(4)

        # Click the "Run Map-Matching Evaluation" button
        click_script = """
        (function() {
            var buttons = Array.from(document.querySelectorAll('button'));
            var btn = buttons.find(b => b.innerText.includes('Run Map-Matching Evaluation'));
            if (btn) {
                btn.click();
                return 'CLICKED';
            }
            return 'BUTTON_NOT_FOUND';
        })()
        """
        res = cdp_send("Runtime.evaluate", {"expression": click_script, "returnByValue": True})
        print("Click result:", res.get("result", {}).get("result", {}).get("value"))

        # Poll until the prediction is evaluated and rendered
        print("Waiting for live Omen inference to complete...")
        for attempt in range(25):
            time.sleep(1)
            check_script = """
            (function() {
                var text = document.body.innerText;
                if (text.includes('Evaluated trip') || text.includes('Performance Metrics Summary')) {
                    return 'DONE';
                }
                if (text.includes('Prediction failed')) {
                    return 'FAILED';
                }
                return 'WAITING';
            })()
            """
            check_res = cdp_send("Runtime.evaluate", {"expression": check_script, "returnByValue": True})
            status = check_res.get("result", {}).get("result", {}).get("value")
            if status == "DONE":
                print(f"Prediction arrived at attempt {attempt+1}!")
                break
            elif status == "FAILED":
                print("Prediction returned failure banner!")
                break
            print(f"Waiting... ({attempt+1})")

        time.sleep(3) # Wait for Folium tiles and Leaflet markers to render

        def take_screenshot(filename):
            print(f"Capturing screenshot: {filename}...")
            shot_res = cdp_send("Page.captureScreenshot", {"format": "png"})
            b64_data = shot_res.get("result", {}).get("data", "")
            if b64_data:
                filepath = os.path.join("docs", filename)
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                size_kb = os.path.getsize(filepath) / 1024
                print(f"Saved {filepath} ({size_kb:.1f} KB)")
                return True
            print("Failed to capture screenshot data")
            return False

        # Tab 1 is currently active: Folium map
        take_screenshot("demo_screenshot_folium_route_map.png")

        # Switch to Tab 2: Trip Evaluation Metrics
        print("Switching to Tab 2: Trip Evaluation Metrics...")
        tab2_script = """
        (function() {
            var tabs = Array.from(document.querySelectorAll('button[data-baseweb="tab"]'));
            var t2 = tabs.find(t => t.innerText.includes('Trip Evaluation Metrics'));
            if (t2) { t2.click(); return 'TAB2_CLICKED'; }
            return 'TAB2_NOT_FOUND';
        })()
        """
        cdp_send("Runtime.evaluate", {"expression": tab2_script, "returnByValue": True})
        time.sleep(2)
        take_screenshot("demo_screenshot_live_trip_metrics.png")

        # Switch to Tab 3: Final Model & Results
        print("Switching to Tab 3: Final Model & Results...")
        tab3_script = """
        (function() {
            var tabs = Array.from(document.querySelectorAll('button[data-baseweb="tab"]'));
            var t3 = tabs.find(t => t.innerText.includes('Final Model & Results'));
            if (t3) { t3.click(); return 'TAB3_CLICKED'; }
            return 'TAB3_NOT_FOUND';
        })()
        """
        cdp_send("Runtime.evaluate", {"expression": tab3_script, "returnByValue": True})
        time.sleep(2)
        take_screenshot("demo_screenshot_final_97pct_results.png")

        print("All screenshots successfully captured!")
        return True

    finally:
        if ws:
            ws.close()
        proc.terminate()

if __name__ == "__main__":
    run_capture()
