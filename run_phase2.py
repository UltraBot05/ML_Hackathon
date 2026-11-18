"""
PathSnap Hackathon — Phase 2 Root Entry Point
Executes Phase 2 pipeline from project root.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SCRIPT_PATH = PROJECT_ROOT / "phase2" / "scripts" / "run_phase2_pipeline.py"


def main():
    args = [sys.executable, str(SCRIPT_PATH)] + sys.argv[1:]
    res = subprocess.run(args, cwd=str(PROJECT_ROOT))
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()

