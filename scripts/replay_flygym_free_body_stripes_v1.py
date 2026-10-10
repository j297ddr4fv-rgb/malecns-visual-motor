#!/usr/bin/env python3
"""CPU free-body replay of already decoded neural stripe commands."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/data/cache/flygym/matplotlib')
import numpy as np
import run_flygym_ap_body_yaw_calibration_v1 as m

BASE=m.ARTIFACT_ROOT.parent.parent/'feedback-diagnosis-20261007-v1/free-body-stripes'
INPUT=m.ARTIFACT_ROOT.parent/'scoped-stripe-replay-v1/runs'
CASES=('stripes_static','stripes_positive60','stripes_negative60')


def verify(protocol):
    for path,record in protocol['files'].items():
        if m.sha256(Path(path))!=record['sha256']:raise RuntimeError(f'Source changed: {path}')


def prepare():
    BASE.mkdir(parents=True,exist_ok=False)
    sources=m._source_records()
    sources[str(Path(__file__).resolve())]={'sha256':m.sha256(Path(__file__))}
    schedules=[]
    for cid in CASES:
        p=INPUT/cid/'decoded_frames.jsonl'
        rows=[json.loads(l) for l in p.open()]
        if len(rows)!=800:raise RuntimeError('Incomplete neural command schedule')
        for i,r in enumerate(rows):
            if r['seq']!=i or not math.isclose(r['frame_time_s'],i*.005,abs_tol=1e-9,rel_tol=0):raise RuntimeError('Schedule clock')
            if not 0<=r['control_power_P']<=1 or not -1<=r['signed_d']<=1:raise RuntimeError('Invalid command')
            if r['control_power_P']<=1e-6 and r['signed_d']!=0:raise RuntimeError('Static contract')
        sources[str(p)]={'sha256':m.sha256(p)}
        schedules.append(dict(case_id=cid,input=str(p),input_sha256=m.sha256(p)))
    m.write_json(BASE/'protocol.json',dict(status='prepared_not_executed',files=sources,cases=schedules,
        body='free-ground V4.1/AP, position servos; no external wrench or root teleport',
        input='same saved neural P/d schedules, calibrated-static reference',seed=1729,
        duration_s=4,control_ticks=4000,physics_steps=40000,native_samples=40001,
        windows_s=[[0,1],[1,1.5],[2,4],[1,4]],closed_loop=False,new_neural_frames=0,gpu=False))
    print(json.dumps(dict(status='prepared',cases=3)))


def run_case(case,protocol):
    from run_flygym_reference_walk import FlyGymReferenceSession,WalkConfig
    from flygym_phase_speed_gait_repaired_v1 import PhaseSpeedGaitController,controller_config_for_differential_ap
    from flygym_ap_endpoint_gesture_prototype import APEndpointGestureConfig
    verify(protocol)
    path=BASE/'runs'/case['case_id'];path.mkdir(parents=True,exist_ok=False)
    record=dict(status='running',case_id=case['case_id'],input_schedule=case,
                closed_loop=False,new_neural_frames=0,gpu=False)
    m.write_json(path/'run-record.json',record)
    try:
        rows=[json.loads(l) for l in Path(case['input']).open()]
        values=dict(m.read_json(m.WALK_EFFECTIVE_CONFIG)['walk_config']);values.update(duration_s=4.,seed=1729)
        for key in ('spawn_position_mm','spawn_quaternion_wxyz','position_actuator_forcerange_native','camera_resolution_hw'):
            if key in values:values[key]=tuple(values[key])
        config=WalkConfig(**values);gesture=APEndpointGestureConfig(**m.read_json(m.CONTROLLER_CONFIG)['config'])
    except BaseException:
        record.update(status='failed_before_session',error=traceback.format_exc());m.write_json(path/'run-record.json',record);raise
    session=None
    try:
        session=FlyGymReferenceSession(duration_s=4.,seed=1729,render_video=False,config=config,run_id=case['case_id'])
        record['initial_qpos_sha256']=m.array_sha256(session.data.qpos.copy())
        record['initial_qvel_sha256']=m.array_sha256(session.data.qvel.copy())
        controller=PhaseSpeedGaitController(session,gesture);session.controller=controller
        with (path/'control_ticks.jsonl').open('x') as log:
            for r in rows:
                controller.gesture_config=controller_config_for_differential_ap(r['signed_d'],base_config=gesture)
                for sub in range(5):
                    tick=r['seq']*5+sub
                    if not math.isclose(session.time_s,tick*.001,abs_tol=1e-8,rel_tol=0):raise RuntimeError('Body clock')
                    session.step_control_tick((r['control_power_P'],)*2)
                    log.write(json.dumps(dict(tick=tick,source_seq=r['seq'],time_s=tick*.001,
                        P_control=r['control_power_P'],signed_d=r['signed_d']))+'\n')
                if r['seq']%200==199:print(json.dumps(dict(case=case['case_id'],frames=r['seq']+1)),flush=True)
        if session._control_samples!=4000 or session._native_samples!=40001:raise RuntimeError('Physics counts')
        # Raw capture precedes schedule-specific analyses.
        metadata=controller.controller_metadata();metadata.update(input_source=case,closed_loop=False,new_neural_frames=0,
            configuration_scope='Constructor metadata only; zero AP fields are initial defaults, not effective commands.',
            effective_command_schedule=case,control_ticks_path=str(path/'control_ticks.jsonl'),
            neural_source_boundary='P and d from decoded DNp15 record; DNa02-like/DNg13-like gesture fields are engineered routing, not recordings from those named cells.')
        session.save(path,constant_drive_values_left_right=None,controller_metadata=metadata)
        a=session.all_arrays;n=session._native_samples;t=np.asarray(a['time_s'][:n]);yaw=np.unwrap(a['thorax_heading_yaw_rad'][:n])
        if not np.isfinite(yaw).all():raise RuntimeError('Nonfinite body yaw')
        windows=[]
        for start,end in protocol['windows_s']:
            delta=float(np.interp(end,t,yaw)-np.interp(start,t,yaw))
            windows.append(dict(window_s=[start,end],body_yaw_delta_deg=math.degrees(delta),mean_body_yaw_rate_rad_s=delta/(end-start)))
        verify(protocol)
        record.update(status='complete',native_samples=n,control_ticks=4000,physics_steps=40000,windows=windows)
        m.write_json(path/'run-record.json',record)
        m.write_json(path/'output-integrity.json',{str(p):m.sha256(p) for p in sorted(path.rglob('*')) if p.is_file()})
    except BaseException:
        record.update(status='failed_stopped',error=traceback.format_exc());m.write_json(path/'run-record.json',record);raise
    finally:
        if session is not None:session.close()


def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');ap.add_argument('--case');args=ap.parse_args()
    if args.prepare:prepare();return
    protocol=m.read_json(BASE/'protocol.json');verify(protocol);auth=m.read_json(BASE/'root-execution-review.json')
    if auth.get('status')!='approved' or auth.get('protocol_sha256')!=m.sha256(BASE/'protocol.json') or auth.get('runner_sha256')!=m.sha256(Path(__file__)):
        raise RuntimeError('Execution review changed')
    with (BASE/('case.lock' if args.case else 'queue.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.case:
            run_case(next(c for c in protocol['cases'] if c['case_id']==args.case),protocol);return
        status=dict(status='running',completed=[],current=None)
        try:
            for c in protocol['cases']:
                status['current']=c['case_id'];m.write_json(BASE/'queue-status.json',status)
                subprocess.run(['systemd-run','--user','--scope','--quiet','--property=MemoryMax=5G','--property=CPUQuota=200%','--property=RuntimeMaxSec=900',
                    '/data/environments/projects/flygym-reference/bin/python',str(Path(__file__).resolve()),'--execute','--case',c['case_id']],check=True)
                status['completed'].append(c['case_id'])
            records=[m.read_json(BASE/'runs'/c/'run-record.json') for c in CASES]
            if len({r['initial_qpos_sha256'] for r in records})!=1 or len({r['initial_qvel_sha256'] for r in records})!=1:raise RuntimeError('Initial states differ')
            m.write_json(BASE/'summary.json',dict(status='complete',cases=records,body_yaw_not_ball=True,closed_loop=False))
            status.update(status='complete',current=None)
        except BaseException as e:status.update(status='failed_stopped',error=repr(e));raise
        finally:m.write_json(BASE/'queue-status.json',status)


if __name__=='__main__':main()
