from pathlib import Path
import argparse,subprocess,sys,json,traceback,shutil
from pathlib import Path
from src.common import ROOT,now,save
from src.audit import audit
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/raw/accepted_2007_to_2018Q4.csv.gz');p.add_argument('--stage',choices=['audit','all'],default='all');args=p.parse_args()
run_id=now().replace(':','').replace('-','').split('.')[0]
out=ROOT/'reports'/run_id;out.mkdir(parents=True,exist_ok=False)
save(ROOT/'latest_attempt.json',{'directory':str(out),'stage':args.stage,'status':'running'})
try:
 if args.stage=='all':
  result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
  (out/'test_results.txt').write_text(result.stdout+result.stderr)
  if result.returncode: raise RuntimeError('Core tests failed; see test_results.txt')
 d=audit(out,args.data.resolve())
 print('AUDIT_OUTPUT',out,flush=True)
 if args.stage=='all':
  from src.pipeline import run
  run(out,d)
 else:save(ROOT/'latest_audit.json',{'directory':str(out),'stage':'audit_complete'})
 save(ROOT/'latest_attempt.json',{'directory':str(out),'stage':args.stage,'status':'passed','finished_at':now()})
except Exception:
 save(ROOT/'latest_attempt.json',{'directory':str(out),'stage':args.stage,'status':'failed','finished_at':now(),'traceback':traceback.format_exc()})
 raise
