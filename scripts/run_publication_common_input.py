#!/usr/bin/env python3
"""Compare intact and rewired CNS graphs using exactly shared retinal movies."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys,time,traceback


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('/data/artifacts/by-project/malecns-gpu/experiments/bc-mechanism-controls'))
    p.add_argument('--source-queue',type=Path,default=Path('/data/artifacts/by-project/malecns-gpu/experiments/visual-motor-large-queue-20261009'))
    p.add_argument('--graph-helper',type=Path,default=Path('/home/xvhang/Projects/scratch/malecns-bc-mechanism/bootstrap_retina_extension_bc.py'))
    p.add_argument('--pixel-adapter',type=Path,default=Path('/home/xvhang/Projects/scratch/malecns-visual-illusions/pixel_illusion_adapter.py'))
    p.add_argument('--variant',choices=('original','shuffle_20261010','shuffle_20261011','shuffle_20261012'))
    return p.parse_args()


def cases(a):
    source=read(a.source_queue/'protocol.json')
    rows=[c for c in source['stimuli']['conditions'] if c['case_id'].startswith('speed_') and c['case_id'].split('_')[1] in ('static','positive60','negative60') and c['neural_seed'] in (1729,1730,1731)]
    rows.sort(key=lambda c:(c['neural_seed'],{'static':0,'positive60':1,'negative60':2}[c['case_id'].split('_')[1]]))
    if len(rows)!=9:raise ValueError('expected nine reference conditions')
    return source,rows


def pose_request(worker,seq,quaternion):
    q=quaternion.astype(float)*[1,-1,1,-1]
    roll,pitch,yaw=worker._quat_to_rpy_wxyz(q)
    return {'protocol':worker.PROTOCOL,'type':'frame','seq':seq,'time_s':seq*.005,'dt_s':.005,
        'pose':{'position_m':[0,0,.002], 'quaternion_wxyz':q.tolist(), 'yaw_rad':yaw,'pitch_rad':pitch,'roll_rad':roll,'velocity_mps':[0,0,0],'angular_velocity_radps':[0,0,0]}}


def child(a):
    import numpy as np
    from functools import partial
    sys.path.insert(0,str(a.project/'scripts'));sys.path.insert(0,str(a.pixel_adapter.parent))
    import malecns_gate,pixel_illusion_adapter
    malecns_gate.cuda_blockers=partial(malecns_gate.cuda_blockers,min_available_gib=0.0)
    pixel_illusion_adapter.install_worker_renderer()
    helper=module('publication_graph_helper',a.graph_helper);helper.SCRIPTS=a.project/'scripts';helper.PROJECT=a.project
    os.environ['BC_GRAPH_SHUFFLE_SEED']=a.variant.rsplit('_',1)[-1] if a.variant!='original' else '20261010'
    helper._patch_graph_loader('original' if a.variant=='original' else 'degree_preserving_target_shuffle')
    worker=module('publication_neural_worker',a.project/'scripts/flygym_online_neural_worker_eye_parity_state_v2.py')
    protocol,selected=cases(a);previous=None
    for case in selected:
        case_id=case['case_id'];run=a.out/'runs'/a.variant/case_id;run.mkdir(parents=True,exist_ok=True)
        if (run/'summary.json').exists():continue
        os.environ.update(FLYGYM_EXTENSION_SCHEDULE_PATH=case['schedule_path'],FLYGYM_EXTENSION_SCHEDULE_SHA256=case['schedule_sha256'],FLYGYM_EXTENSION_CASE_ID=case_id)
        argv=['--config',protocol['runtime']['worker_config'],'--stimulus-condition','angular','--angular-stimulus-spec',case['spec_path'],'--seed',str(case['neural_seed']),'--max-frames','2600','--static-warmup-s','1.0','--capture-mode','light','--deterministic-vision','--deterministic-neural-delivery','--out',str(run/'worker')]
        w=worker.OnlineNeuralWorker(worker._parse_cli(argv),reuse_from=previous)
        cache=a.out/'inputs'/f'{case_id}.npz';input_hash=hashlib.sha256();drive_hash=hashlib.sha256();drive_hash.update(np.ascontiguousarray(w.target_indices).tobytes())
        if a.variant=='original':
            with np.load(a.reference/'runs/original'/case_id/'trajectory.npz',allow_pickle=False) as n: quaternions=n['thorax_quat_wxyz'][:2600].copy()
            pixels=np.empty((2600,2,721),np.float32)
            render=w._render_columns
            def render_shared(pose,t):
                left,right,clock=render(pose,t);seq=round(t/.005);pixels[seq,0]=left;pixels[seq,1]=right;return left,right,clock
        else:
            with np.load(cache,allow_pickle=False) as n:pixels=n['pixels'].copy();quaternions=n['thorax_quat_wxyz'].copy()
            def render_shared(pose,t):
                seq=round(t/.005);return pixels[seq,0],pixels[seq,1],float(t)
        w._render_columns=render_shared
        original_step=w.engine.step_window
        def record_drive(sensory,ts,**kwargs):
            value=np.ascontiguousarray(sensory.channels['flyvis_t4t5']['values'],dtype=np.float32);drive_hash.update(value.tobytes());return original_step(sensory,ts,**kwargs)
        w.engine.step_window=record_drive
        records=[];start=time.monotonic()
        state=read(a.out/'state.json');state['current_run']=a.variant+'/'+case_id;write(a.out/'state.json',state)
        for seq in range(2600):
            reply=w.process_frame(pose_request(worker,seq,quaternions[seq]));input_hash.update(pixels[seq].tobytes());rates=reply['population_rates_hz'];named=w.engine.last_named_outputs
            records.append({'seq':seq,'frame_time_s':seq*.005,'DNp15_L_hz':rates['DNp15/L'],'DNp15_R_hz':rates['DNp15/R'],'raw_P':reply['leg_power'],'network_spikes':named['spike_count']})
            if seq and seq%500==0:print(json.dumps({'current':a.variant+'/'+case_id,'frames':seq+1}),flush=True)
        # Restore the bound method before resident reuse to avoid nested observers.
        w.engine.step_window=original_step
        if a.variant=='original':
            cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,pixels=pixels,thorax_quat_wxyz=quaternions,target_indices=w.target_indices)
        input_digest=input_hash.hexdigest();drive_digest=drive_hash.hexdigest()
        if a.variant!='original':
            baseline=read(a.out/'runs/original'/case_id/'summary.json')
            if input_digest!=baseline['retinal_movie_sha256'] or drive_digest!=baseline['fitted_drive_stream_sha256']:raise RuntimeError('shared retinal or fitted-current stream differs from intact reference')
        stop=w.stop()
        with (run/'neural_trace.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        active=records[200:]
        result={'variant':a.variant,'case_id':case_id,'condition':case_id.split('_')[1],'initialization_seed':case['neural_seed'],'rewiring_seed':None if a.variant=='original' else int(a.variant.rsplit('_',1)[-1]),'frames':2600,'retinal_movie_sha256':input_digest,'fitted_drive_stream_sha256':drive_digest,'matched_retina_and_fitted_drive':True,'DNp15_L_hz':float(np.mean([r['DNp15_L_hz'] for r in active])),'DNp15_R_hz':float(np.mean([r['DNp15_R_hz'] for r in active])),'DNp15_opponent_hz':float(np.mean([r['DNp15_L_hz']-r['DNp15_R_hz'] for r in active])),'mean_raw_P':float(np.mean([r['raw_P'] for r in active])),'mean_network_spikes_per_5ms':float(np.mean([r['network_spikes'] for r in active])),'wall_s':time.monotonic()-start,'neural_worker_stopped':stop.get('type')=='stopped','no_body_simulation_or_feedback':True}
        write(run/'network_intervention.json',helper.GRAPH_RECORD);write(run/'summary.json',result)
        state=read(a.out/'state.json');state['completed_runs']+=1;state['runs'].append(result);state['current_run']=None;write(a.out/'state.json',state)
        print(json.dumps({'completed':state['completed_runs'],'summary':result}),flush=True)
        previous=w


def parent(a):
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'state.json').exists():raise FileExistsError('output already contains a queue state; inspect before resuming')
    write(a.out/'protocol.json',{'design':'common-input replay of recorded intact-body retinal movies','stimuli':['static','positive60','negative60'],'initialization_seeds':[1729,1730,1731],'initial_variants':['original','shuffle_20261010'],'additional_rewiring_seeds':[20261011,20261012],'frames_per_run':2600,'dt_s':.005,'stimulus_window_s':[1,13],'pose_reflection':'quaternion (w,x,y,z) -> (w,-x,y,-z); angular-field eye translation does not affect rays','additional_graph_condition':'intact raw moving-minus-static contrasts have expected signs and first shuffled graph has mean absolute contrast below half intact','reference_closed_loop_data':str(a.reference),'shared_retinal_and_current_hashes_required':True,'no_body_actuation':True})
    write(a.out/'state.json',{'status':'running','completed_runs':0,'total_runs':18,'current_run':None,'runs':[]})
    variants=['original','shuffle_20261010'];i=0
    try:
        while i<len(variants):
            variant=variants[i]
            command=[sys.executable,str(Path(__file__).resolve()),'--project',str(a.project),'--out',str(a.out),'--reference',str(a.reference),'--source-queue',str(a.source_queue),'--graph-helper',str(a.graph_helper),'--pixel-adapter',str(a.pixel_adapter),'--variant',variant]
            subprocess.run(command,check=True)
            if i==1:
                state=read(a.out/'state.json');rows=state['runs'];lookup={(r['variant'],r['initialization_seed'],r['condition']):r for r in rows};effects={}
                for v in variants:
                    effects[v]={c:sum(lookup[v,s,c]['DNp15_opponent_hz']-lookup[v,s,'static']['DNp15_opponent_hz'] for s in (1729,1730,1731))/3 for c in ('positive60','negative60')}
                intact=effects['original'];null=effects['shuffle_20261010'];extend=intact['positive60']>0 and intact['negative60']<0 and sum(abs(x) for x in null.values())<.5*sum(abs(x) for x in intact.values())
                state['first_graph_effects_hz']=effects;state['additional_graphs_scheduled']=extend
                if extend:variants+=['shuffle_20261011','shuffle_20261012'];state['total_runs']=36
                write(a.out/'state.json',state)
            i+=1
        state=read(a.out/'state.json');state.update(status='complete',current_run=None);write(a.out/'state.json',state)
    except BaseException as error:
        state=read(a.out/'state.json');state.update(status='failed',error=str(error));write(a.out/'state.json',state);raise

if __name__=='__main__':
    a=args()
    if a.variant:child(a)
    else:parent(a)
