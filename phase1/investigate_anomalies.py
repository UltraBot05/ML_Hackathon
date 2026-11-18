"""
PathSnap Hackathon - Legacy Wrapper for Anomaly Investigation
Delegates directly to phase1/scripts/investigate_anomalies.py
"""
import sys
from pathlib import Path

# Add scripts directory to path and execute main
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from investigate_anomalies import run_investigation

if __name__ == "__main__":
    run_investigation()
