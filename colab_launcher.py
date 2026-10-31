# Run this in ONE Colab cell after cloning your repo, e.g.:
#   !git clone https://github.com/<your-username>/tokens-embeddings-explorer.git
#   %cd tokens-embeddings-explorer
#   %run colab_launcher.py
import subprocess, time, re, sys, os

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
if not os.path.exists("cloudflared"):
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared", shell=True)

subprocess.run("pkill -f streamlit; pkill -f cloudflared", shell=True)
time.sleep(2)

subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(6)

tunnel = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in tunnel.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        print("Open your app here:", m.group(0))
        break
