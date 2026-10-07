"""
EduPredict AI - Auto Tunnel & Launcher
Starts Streamlit, establishes Cloudflare Tunnel, extracts the exact live URL,
and writes it to CURRENT_LIVE_LINK.txt on your Desktop.
"""

import os
import sys

# Ensure UTF-8 output on Windows terminal
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import subprocess
import webbrowser
import re

APP_DIR = r"C:\Users\vivek\OneDrive\Desktop\EduPredict_AI"
DESKTOP_DIR = r"C:\Users\vivek\OneDrive\Desktop"
PYTHON_EXE = os.path.join(APP_DIR, ".venv", "Scripts", "python.exe")
CLOUDFLARED_EXE = os.path.join(APP_DIR, "cloudflared.exe")
LINK_FILE = os.path.join(DESKTOP_DIR, "CURRENT_LIVE_LINK.txt")

def kill_old_processes():
    print("[1/3] Cleaning up old processes on port 8501...")
    try:
        subprocess.run(
            'powershell -Command "Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force"',
            shell=True, capture_output=True
        )
        subprocess.run(
            'powershell -Command "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"',
            shell=True, capture_output=True
        )
    except Exception as e:
        print("Cleanup note:", e)
    time.sleep(2)

def start_streamlit():
    print("[2/3] Starting Streamlit Application on Port 8501...")
    cmd = [PYTHON_EXE, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
    st_proc = subprocess.Popen(cmd, cwd=APP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    return st_proc

def start_tunnel():
    print("[3/3] Connecting to Cloudflare Tunnel (Mobile & Web)...")
    cmd = [CLOUDFLARED_EXE, "tunnel", "--url", "http://localhost:8501"]
    tunnel_proc = subprocess.Popen(cmd, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    
    live_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    
    print("Obtaining secure public HTTPS link...")
    start_time = time.time()
    while time.time() - start_time < 30:
        line = tunnel_proc.stdout.readline()
        if line:
            match = url_pattern.search(line)
            if match:
                live_url = match.group(0)
                break
        time.sleep(0.1)
        
    if live_url:
        print("\n" + "="*60)
        print("SUCCESS! YOUR LIVE PUBLIC LINK IS READY:")
        print(f"--> {live_url}")
        print("="*60 + "\n")
        
        # Save to Desktop text file
        with open(LINK_FILE, "w", encoding="utf-8") as f:
            f.write(f"EduPredict AI Live Mobile & Web Link:\n\n{live_url}\n\n(This link works on all mobile phones, laptops, and networks while this launcher window is open)\n")
        
        print(f"Saved link to Desktop file: CURRENT_LIVE_LINK.txt")
        print("Opening link in your browser...")
        try:
            webbrowser.open(live_url)
        except Exception:
            pass
    else:
        print("Could not capture URL automatically, but tunnel is active.")

    print("\n[Server is LIVE and RUNNING. Keep this window open. Press Ctrl+C to stop]\n")
    try:
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("Stopping launcher...")

if __name__ == "__main__":
    kill_old_processes()
    start_streamlit()
    start_tunnel()
