#!/usr/bin/env python3
"""Serial, version-bound standardized neural assay queue. Never retries automatically."""
from pathlib import Path
import datetime, hashlib, json, subprocess, fcntl
BASE=Path('/data/artifacts/by-project/malecns-gpu/engineering/standardized-repair-20261007-v1')
RUNNER=Path('/home/xvhang/Projects/malecns-gpu/scripts/run_flygym_standardized_case_v1.py')
PYTHON='/data/environments/projects/flygym-reference/bin/python'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(value):
    p=BASE/'queue-status.json'; t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def main():
    lock=(BASE/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=BASE/'input-manifest.json'; review=BASE/'root-execution-review.json'
    plan=json.loads(manifest.read_text()); auth=json.loads(review.read_text())
    if digest(RUNNER)!=auth['harness_source_sha256']: raise RuntimeError('case runner hash changed')
    if digest(Path(__file__))!=auth['queue_source_sha256']: raise RuntimeError('queue source hash changed')
    if digest(BASE/'repair-source-freeze.json')!=auth['repair_source_freeze_sha256']: raise RuntimeError('repair freeze changed')
    if digest(manifest)!=auth['input_manifest_sha256']: raise RuntimeError('manifest changed')
    record={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':[],'current':None}
    save(record)
    try:
        for number,case in enumerate(plan['cases']):
            if digest(RUNNER)!=auth['harness_source_sha256']: raise RuntimeError('case runner hash changed')
            cid=case['case_id']
            if cid not in auth['authorized_case_ids']: raise RuntimeError('case unauthorized')
            unit=f'flygm-standardized-repair-{number:02d}-20261007-v1'
            record['current']={'case_id':cid,'assay':case['assay'],'unit':unit+'.scope'};save(record)
            log=BASE/'execution-logs'/f'{number:02d}-{cid}.log';log.parent.mkdir(parents=True,exist_ok=True)
            cmd=['systemd-run','--user','--scope','--unit='+unit,'-p','MemoryMax=5G','-p','CPUQuota=200%','-p','RuntimeMaxSec=900','env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',PYTHON,str(RUNNER),'--manifest',str(manifest),'--case-id',cid,'--root-review',str(review),'--execute']
            with log.open('x') as output: subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,check=True)
            record['completed'].append(record['current']);save(record)
        record.update(status='complete',current=None);save(record)
    except Exception as error:
        record.update(status='failed_stopped',error=repr(error));save(record);raise
if __name__=='__main__':main()
