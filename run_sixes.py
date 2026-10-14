"""Reproduce the six-rate project, including its executed notebook and checks."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
if not (ROOT/'data/processed/deliveries.parquet').exists():
    subprocess.run([sys.executable,str(ROOT/'pipeline.py')],cwd=ROOT,check=True)
start=time.monotonic()
for module in ['study','context','cases','forecasts','verify','report']:
    print(f'Running sixes.{module}',flush=True)
    subprocess.run([sys.executable,'-m',f'sixes.{module}'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
print(f'Six-rate project completed in {(time.monotonic()-start)/60:.1f} minutes',flush=True)
