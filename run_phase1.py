"""
PathSnap Hackathon - Top-level Phase 1 entry point
Usage from project root: python run_phase1.py
"""
import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent / "phase1" / "scripts"
sys.path.insert(0, str(scripts_dir))

from run_phase1_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()

