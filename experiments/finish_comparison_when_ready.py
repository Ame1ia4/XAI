"""Finish the dated report automatically after the active notebook run ends."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/2026-10-09"
status_path = OUT / "report_status.json"
status_path.write_text(json.dumps({"status":"waiting_for_notebook_completion",
    "automatic_report_update":True,"result_date":"2026-10-09"},indent=2))
print("Waiting for the four notebook attempts to finish.",flush=True)
while True:
    try:
        manifest=json.loads((OUT / "execution_manifest.json").read_text())
    except json.JSONDecodeError:
        time.sleep(1)
        continue
    if len(manifest["runs"])==4 and all(r["status"] != "running" for r in manifest["runs"]):
        break
    time.sleep(30)
os.environ["MPLCONFIGDIR"] = str(ROOT / ".venv/report_matplotlib_cache")
result=subprocess.run([sys.executable,str(ROOT / "experiments/summarize_comparison.py")])
status_path.write_text(json.dumps({"status":"report_updated" if result.returncode==0 else "report_update_failed",
    "automatic_report_update":True,"result_date":"2026-10-09",
    "completed_models":[r["model"] for r in manifest["runs"] if r["status"]=="completed"],
    "returncode":result.returncode},indent=2))
raise SystemExit(result.returncode)
