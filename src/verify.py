"""Run tests and the unchanged supplied scorer; write verifiable local evidence."""
import hashlib
import json
import subprocess
import sys
import zipfile
from . import config as C

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=C.ROOT,text=True,capture_output=True)
    print(tests.stdout+tests.stderr)
    if tests.returncode: raise SystemExit(tests.returncode)
    cmd=[sys.executable,'score.py','--predictions','outputs/validation_predictions.csv','--december-predictions','outputs/december_chart_inputs.csv','--output-dir','outputs/scorer_results']
    scorer=subprocess.run(cmd,cwd=C.ROOT,text=True,capture_output=True)
    print(scorer.stdout+scorer.stderr)
    if scorer.returncode: raise SystemExit(scorer.returncode)
    result={'tests_passed':True,'test_log':tests.stdout+tests.stderr,'scorer_passed':True,'scorer_log':scorer.stdout+scorer.stderr,
      'scorer_sha256':sha(C.ROOT/'score.py'),'predictions_sha256':sha(C.OUT/'validation_predictions.csv'),
      'december_sha256':sha(C.OUT/'december_chart_inputs.csv')}
    repeat=C.OUT/'repeatability.json'
    if repeat.exists(): result['repeatability']=json.loads(repeat.read_text())
    (C.OUT/'verification.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
