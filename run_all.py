"""Default: reproduce the six-rate project. --legacy reproduces the original boundary study."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--legacy',action='store_true',help='Rebuild the original archived boundary-rate project')
args=parser.parse_args()
steps=['pipeline.py','research.py','context_analysis.py','watchlists.py','visualise.py','robustness.py','build_report.py','make_walkthrough.py','make_results_notebook.py','verify_results.py'] if args.legacy else ['run_sixes.py']
readme=(ROOT/'README.md').read_bytes() if args.legacy else None
try:
    for step in steps:
        subprocess.run([sys.executable,str(ROOT/step)],cwd=ROOT,check=True)
finally:
    if readme is not None:
        (ROOT/'README.md').write_bytes(readme)
