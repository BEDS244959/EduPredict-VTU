import sys
import time
from pycloudflared import try_cloudflare

def main():
    print("Starting Cloudflare Tunnel to port 8501...")
    tunnel = try_cloudflare(port=8501)
    print(f"\n==========================================")
    print(f"🌟 DIRECT CLOUDFLARE PUBLIC URL:")
    print(f"{tunnel.tunnel}")
    print(f"==========================================\n")
    sys.stdout.flush()
    
    # Keep process alive
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping tunnel...")

if __name__ == "__main__":
    main()
