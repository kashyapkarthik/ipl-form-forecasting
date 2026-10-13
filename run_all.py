"""One command to reproduce the complete project using the pinned data snapshot."""
from pathlib import Path
import subprocess, sys, time, json
ROOT=Path(__file__).resolve().parent
steps=['pipeline.py','research.py','context_analysis.py','watchlists.py','visualise.py','robustness.py','build_report.py','make_walkthrough.py','make_results_notebook.py','verify_results.py']
started=time.monotonic()
for step in steps:
    print(f'\nRunning {step}',flush=True)
    subprocess.run([sys.executable,str(ROOT/step)],cwd=ROOT,check=True)
print(f'Completed all stages in {(time.monotonic()-started)/60:.1f} minutes')
