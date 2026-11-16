"""Configurable fresh workflow. Archived replay remains in notebook_workflow.

All simulation/optimization/receiver work is MPS FP32; CPU arrays are exported
for the existing plotting and archival pipeline only.
"""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import copy
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from . import mps_numerics as m
from .artifacts import write_json,sha
from .basis_source import SelectedBasis
from .delay_doppler import Channels
from .scaling_evidence import ScalingEvidence,prepare_scaling,load_scaling,GRIDS,SIZES
from .notebook_workflow import NotebookWorkflow,validate_config,NOMINAL
from .overlays import validate_review
from .training import DEFAULT_TRAINING,train,load_checkpoint,save_checkpoint,groups,coupling

DEFAULT_EVALUATION=dict(seed=303,matrix_channels=2,matrix_index=0,max_channels=8,min_channels=4,
    batch_channels=2,blocks_per_channel=2,snrs_db=(0.,10.,18.,25.),modulations=('QPSK','16QAM'),
    target_errors=100,max_seconds_per_point=30.,diagnostic_atoms=8,diagnostic_channels=8)


def integer(x,lo,hi,name):
    if type(x) is not int or not lo<=x<=hi: raise ValueError(f'{name} must be integer in [{lo},{hi}]')


def finite(x,lo,hi,name):
    if type(x) not in (int,float) or not math.isfinite(x) or not lo<=x<=hi:
        raise ValueError(f'{name} must be finite in [{lo},{hi}]')


def exact_keys(value,expected,name):
    if not isinstance(value,dict) or set(value)!=set(expected): raise ValueError(f'Unsupported/missing {name} keys; expected {sorted(expected)}')


def channel(model):
    # Reuse the existing DD law's strict range/family validation without sampling.
    validate_config('fresh',model,dict(basis_columns=(0,),snr_max_db=None),
        dict(seed=1,matrix_channels=1,matrix_index=0,ber_channels=2,blocks_per_channel=1,snrs_db=(0.,1.)))
    return {**model,'delay_range':tuple(model['delay_range']),'doppler_range':tuple(model['doppler_range'])}


def local_path(path,workspace):
    root=Path(workspace).resolve(); path=Path(path)
    if not path.is_absolute(): path=root/path
    path=path.resolve()
    if not path.is_relative_to(root/'local'): raise ValueError('Checkpoint/output paths must stay in this worktree local/ (including symlink targets)')
    return path


def validate(*,mode,evidence_mode,scenario,model,display,evaluation,training,k,support_policy,discovery,checkpoint,workspace,output_root,review):
    if mode not in ('load_existing','train_new'): raise ValueError('MODE must be load_existing or train_new')
    if scenario not in ('dd_v1','residual_cfo'): raise ValueError('SCENARIO must be dd_v1 or residual_cfo')
    if scenario=='residual_cfo':
        if mode=='train_new': raise NotImplementedError('Residual CFO train_new unsupported: archived staged W/R optimization is not reconstructed/qualified; no DD trainer substitution')
        raise ValueError('CFO uses start_cfo workflow configuration')
    if evidence_mode not in ('fresh','saved_replay'): raise ValueError('EVIDENCE_MODE must be fresh or saved_replay')
    model=channel(model); n=model['n']; integer(k,1,n,'K')
    if support_policy not in ('frozen','recalibrate'): raise ValueError('support_policy must be frozen or recalibrate')
    if mode=='train_new' and evidence_mode!='fresh': raise ValueError('train_new requires fresh evaluation; no historical BER reuse')
    if mode=='train_new' and support_policy!='frozen': raise ValueError('train_new supplies train-only support; use frozen policy')
    checkpoint=local_path(checkpoint,workspace) if checkpoint else None
    output_root=local_path(output_root,workspace)
    if checkpoint and checkpoint.suffix!='.npz': raise ValueError('Checkpoint requires .npz suffix')
    if mode=='train_new' and checkpoint and (checkpoint.exists() or checkpoint.with_suffix('.json').exists()): raise FileExistsError('Checkpoint exists; no overwrite/resume')
    if mode=='load_existing' and training is not None: raise ValueError('TRAINING must be None in load_existing; no ignored training knobs')
    if mode=='train_new':
        exact_keys(training,set(DEFAULT_TRAINING)|{'model','validation_model'},'TRAINING')
        training=copy.deepcopy(training)
        training['model']=channel(training['model']); training['validation_model']=channel(training['validation_model'])
        if training['model']['n']!=n or training['validation_model']['n']!=n: raise ValueError('Training/validation/evaluation N must agree; no resizing')
        for key,lo,hi in [('seed',0,2**31-1),('validation_seed',0,2**31-1),('iterations',1,10000),('layers',2,8),('batch_channels',2,128),('validation_channels',2,128),('validate_every',1,10000)]: integer(training[key],lo,hi,key)
        if training['validate_every']>training['iterations']: raise ValueError('validate_every exceeds iterations')
        finite(training['learning_rate'],1e-5,.2,'learning_rate'); finite(training['max_seconds'],.01,7200,'training max_seconds')
        finite(training['group_threshold'],.01,.9,'group_threshold')
        if training['initialization'] not in ('OFDM','OTFS','AFDM'): raise ValueError('Initialization must be OFDM/OTFS/AFDM')
        if n*n*(training['batch_channels']+training['validation_channels'])*training['model']['paths']>2**24: raise ValueError('Training channel memory cap exceeded')
    if evidence_mode=='saved_replay':
        if checkpoint or support_policy!='frozen' or k!=SIZES[n]: raise ValueError('Saved BER requires exact archived W/support/K; use fresh')
        if evaluation is not None or discovery is not None: raise ValueError('Replay EVALUATION/DISCOVERY must be None; no ignored controls')
        config=validate_config('saved_replay',model,display,None,review=review)
    else:
        exact_keys(evaluation,DEFAULT_EVALUATION,'EVALUATION'); evaluation=copy.deepcopy(evaluation)
        for key,lo,hi in [('seed',0,2**31-1),('matrix_channels',1,16),('max_channels',2,4096),('min_channels',2,4096),('batch_channels',2,16),('blocks_per_channel',1,64),('target_errors',0,1000000),('diagnostic_atoms',2,128),('diagnostic_channels',2,128)]: integer(evaluation[key],lo,hi,key)
        integer(evaluation['matrix_index'],0,evaluation['matrix_channels']-1,'matrix_index')
        if not evaluation['batch_channels']<=evaluation['min_channels']<=evaluation['max_channels']: raise ValueError('Require batch_channels <= min_channels <= max_channels')
        finite(evaluation['max_seconds_per_point'],.01,3600,'evaluation max_seconds_per_point')
        mods=evaluation['modulations']
        if not isinstance(mods,(list,tuple)) or not mods or len(set(mods))!=len(mods) or not set(mods)<= {'QPSK','16QAM'}: raise ValueError('Modulations must be unique QPSK/16QAM selections')
        if evaluation['max_channels']*evaluation['blocks_per_channel']*n*max(2 if x=='QPSK' else 4 for x in mods)>2**27: raise ValueError('Evaluation bit cap exceeded')
        proxy=dict(seed=evaluation['seed'],matrix_channels=evaluation['matrix_channels'],matrix_index=evaluation['matrix_index'],ber_channels=2,blocks_per_channel=1,snrs_db=evaluation['snrs_db'])
        config=validate_config('fresh',model,display,proxy,review=review)
        if training and len({training['seed'],training['validation_seed'],evaluation['seed']})!=3: raise ValueError('Train/validation/test seeds must be distinct')
        if support_policy=='recalibrate':
            exact_keys(discovery,{'model','seed','channels','group_threshold'},'DISCOVERY'); discovery=copy.deepcopy(discovery)
            discovery['model']=channel(discovery['model'])
            if discovery['model']['n']!=n: raise ValueError('Discovery N must match evaluation N')
            integer(discovery['seed'],0,2**31-1,'discovery seed'); integer(discovery['channels'],2,128,'discovery channels')
            finite(discovery['group_threshold'],.01,.9,'group_threshold')
            if discovery['seed']==evaluation['seed']: raise ValueError('Discovery/test seeds must differ')
        elif discovery is not None: raise ValueError('DISCOVERY must be None for frozen support')
        if mode=='load_existing' and checkpoint is None and support_policy=='frozen' and k!=SIZES[n]: raise ValueError('Archived frozen K differs; select recalibrate or train_new')
    config.update(basis_mode=mode,scenario=scenario,evaluation=evaluation,training=training,k=k,support_policy=support_policy,discovery=discovery)
    return config,checkpoint,output_root


class FreshWorkflow(NotebookWorkflow):
    """Same inherited plotting cells for loaded or trained W; no saved BER loader."""
    def __init__(self,config,workspace,save_run,selected,record,masks,output_root):
        self.config=config; self.environment=m.environment(); self.n=config['model']['n']; self.k=config['k']
        self.config['fixed_support_label']='Fixed train/discovery support'
        self.selected_basis=selected.validate(); self.saved_reference=False; self.scaling=None
        self.artifacts=Path(workspace)/'local'; self.reference_views=dict(labels=record['labels'],permutation=record['permutation'],selected=np.array(config['display']['basis_columns']))
        self.record=record
        self.ws={'Learned':selected.matrix.clone()}
        self.mps_ws={'Learned':m.pair(selected.matrix.numpy()),**{name:w for name,w in m.classical(self.n,GRIDS[self.n]).items() if name in ('OTFS','AFDM')}}
        self.ws.update({name:torch.from_numpy(m.array(w)) for name,w in self.mps_ws.items() if name!='Learned'})
        self._selected=self.ws['Learned'].clone()
        self.masks={name:torch.from_numpy(value.detach().cpu().numpy().copy()) for name,value in masks.items()}
        self.bases={name:w.numpy() for name,w in self.ws.items()}; self.bases['dft']=m.array(m.dft(self.n)); self.bases['labels']=record['labels']
        self.example_columns()
        self.channels=self.effective_channels=self.ber_table=None
        self.counts,self.images={},{}; self.matrix_index=config['evaluation']['matrix_index']; self.run_dir=None
        if save_run:
            self.run_dir=output_root/(config['basis_mode']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]); self.run_dir.mkdir(parents=True)
        self.provenance=dict(config=config,environment=self.environment,selected_learned=selected.describe(),
            fixed_support='Own fixed discovery support for every waveform; no per-realization/SNR switching. '+config['support_policy'],
            reference_group_views_applicable=False,BER_status='Not yet evaluated',
            scope='Fresh bounded uncoded DD-v1, raw slicing, Es/N0; one Learned per run, no BER guarantee',
            grouping=record.get('grouping','Frozen native grouping of selected basis'),
            bank_roles='train/discovery determines support/groups; validation selects W only; test diagnostics/matrix/BER use distinct role namespaces',
            uncertainty='Channel SE descriptive; zero-channel bound conservative. Error/time stopping intervals approximate; no training-seed uncertainty.')

    def summary(self):
        return dict(operation='Fresh MPS FP32 evaluation of one selected W; no saved BER counts',active_config=self.config,
                    selected_learned=self.selected_basis.describe(),device_dtype_threads=self.environment,
                    fixed_support=self.provenance['fixed_support'],grouping=self.provenance['grouping'],
                    group_count=int(len(np.unique(self.reference_views['labels']))),bank_roles=self.provenance['bank_roles'])

    def generate_channels(self):
        if self.channels is not None: return self.channels,self.effective_channels
        e=self.config['evaluation']; ns=('configurable-dd-v1','test',e['seed'],'matrix')
        h,info=m.sample(e['matrix_channels'],self.config['model'],ns)
        export=lambda x:torch.from_numpy(x.detach().cpu().numpy().copy())
        self.channels=Channels(torch.from_numpy(m.array(h)),torch.from_numpy(m.array(info['raw'])),export(info['tau']),export(info['nu']),torch.from_numpy(m.array(info['gains'])),export(info['scale']))
        self.effective_channels={name:torch.from_numpy(m.array(m.effective(w,h))) for name,w in self.mps_ws.items()}
        self.provenance['matrix_generation']=dict(namespace=ns,index=self.matrix_index)
        return self.channels,self.effective_channels

    def coupling_data(self):
        self.generate_channels()
        if not hasattr(self,'_coupling'):
            e=self.config['evaluation']; ns=('configurable-dd-v1','test',e['seed'],'diagnostics')
            op,_=m.sample(e['diagnostic_atoms'],self.config['model'],(*ns,'atoms'),atomic=True)
            h,_=m.sample(e['diagnostic_channels'],self.config['model'],(*ns,'multipath'))
            w=self.mps_ws['Learned']; f=m.dft(self.n)
            example=m.pair(self.channels.h[self.matrix_index].numpy()); a=m.effective(w,example)
            c=m.mm(f,w); hf=m.mm(m.mm(f,example),m.adj(f)); bridge=m.mm(m.mm(m.adj(c),hf),c)
            export=lambda x:x.detach().cpu().numpy().copy()
            self._coupling=dict(C=m.array(c),H_f=m.array(hf),A=m.array(a),A_spectral=m.array(bridge),
                labels=self.reference_views['labels'],permutation=self.reference_views['permutation'],
                Q_test=export(coupling(w,op)),mean_channel_power=export(coupling(w,h)),
                unitarity_max=m.unitarity(w),bridge_max=float(m.power((bridge[0]-a[0],bridge[1]-a[1])).sqrt().max()),
                channel_stat_label=f'Mean |A|² · {e["diagnostic_channels"]} evaluation-law channels',
                atom_stat_label=f'Q test · {e["diagnostic_atoms"]} evaluation-law unitary atoms',
                statistic_scope='Independent test diagnostics')
            self.record.update(q_test=self._coupling['Q_test'],ensemble_power=self._coupling['mean_channel_power'])
        return self._coupling

    def evaluate_ber(self):
        if self.ber_table is not None: return self.ber_table
        e=self.config['evaluation']; rows=[]
        for mod in e['modulations']:
            for snr in e['snrs_db']:
                rows.extend(self._point(mod,float(snr)))
        self.ber_table=pd.DataFrame(rows)
        self.provenance['BER_status']='NEW bounded MPS FP32 paired BER; descriptive channel-SE intervals, approximate under stopping. Zero errors unresolved.'
        return self.ber_table

    def _point(self,mod,snr):
        e=self.config['evaluation']; started=time.perf_counter(); count=0; batches=[]; mses=[]; infos=[]; seeds=[]; stop='channel_budget'
        bits_per=e['blocks_per_channel']*self.n*(2 if mod=='QPSK' else 4); variance=10**(-snr/10)
        while count<e['max_channels']:
            batch=len(batches); take=min(e['batch_channels'],e['max_channels']-count)
            ns=('configurable-dd-v1','test',e['seed'],'ber',batch)
            h,info=m.sample(take,self.config['model'],(*ns,'channel'))
            bits=m.draw((take,e['blocks_per_channel'],self.n,2 if mod=='QPSK' else 4),(*ns,'bits',mod,snr),bits=True)
            s=m.symbols(bits,mod)
            shape=(take,e['blocks_per_channel'],self.n)
            noise=(math.sqrt(variance/2)*m.draw(shape,(*ns,'noise-real',mod,snr),normal=True),math.sqrt(variance/2)*m.draw(shape,(*ns,'noise-imag',mod,snr),normal=True))
            errors=[]; mse=[]
            for name,w in self.mps_ws.items():
                a=m.effective(w,h)
                y=m.mm(m.mm(s,m.transpose(w)),m.transpose(h))
                y=y[0]+noise[0],y[1]+noise[1]
                z=m.mm(y,(w[0],-w[1]))
                arm=[]; arm_mse=[]
                for kind in ('fixed','full'):
                    support=self.masks[name].numpy() if kind=='fixed' else None
                    g=m.receiver(a,variance,m.tensor(support) if support is not None else None)
                    estimate=m.mm(z,m.transpose(g))
                    arm.append((m.decode(estimate,mod)!=bits).sum((1,2,3)).cpu().numpy())
                    ga=m.mm(g,a); ident=m.eye(self.n)
                    err=m.power((ga[0]-ident[0],ga[1])).sum(-1)+variance*m.power(g).sum(-1)
                    arm_mse.append(err.mean(-1).cpu().numpy())
                errors.append(arm); mse.append(arm_mse)
            errors=np.asarray(errors); batches.append(errors); mses.append(np.asarray(mse)); count+=take
            infos.append({**{key:value.detach().cpu().numpy() for key,value in info.items() if key in ('tau','nu','scale')},
                          'gain_real':info['gains'][0].detach().cpu().numpy(),
                          'gain_imag':info['gains'][1].detach().cpu().numpy()})
            seeds.append(dict(namespace=ns,count=take))
            if count>=e['min_channels'] and e['target_errors'] and np.concatenate(batches,axis=-1).sum(-1).min()>=e['target_errors']:
                stop='error_target'; break
            if time.perf_counter()-started>=e['max_seconds_per_point']:
                stop='time_budget'; break
        torch.mps.synchronize()
        errors=np.concatenate(batches,axis=-1); analytic=np.concatenate(mses,axis=-1)
        self.counts[f'{mod}_{snr:g}dB']=dict(errors=errors,analytic_mse=analytic,bits_per_channel=np.array(bits_per),
            waveforms=np.array(list(self.ws)),receivers=np.array(['fixed','full']),
            tau=np.concatenate([x['tau'] for x in infos]),nu=np.concatenate([x['nu'] for x in infos]),
            gain_real=np.concatenate([x['gain_real'] for x in infos]),gain_imag=np.concatenate([x['gain_imag'] for x in infos]),
            normalization=np.concatenate([x['scale'] for x in infos]),
            batch_namespaces_json=np.array(json.dumps(seeds)))
        rows=[]
        for wi,name in enumerate(self.ws):
            for ri,kind in enumerate(('fixed','full')):
                events=int(errors[wi,ri].sum()); bits=count*bits_per; rate=events/bits
                se=float((errors[wi,ri]/bits_per).std(ddof=1)/math.sqrt(count)) if count>1 else float('nan')
                rows.append(dict(N=self.n,K=self.k,scenario='configured_dd_v1',modulation=mod,snr_db=snr,waveform=name,receiver=kind,slicer='raw',
                    channels=count,bits=bits,errors=events,ber=rate,channel_se=se,
                    ci_low=max(0,rate-1.96*se) if events else np.nan,ci_high=min(1,rate+1.96*se) if events else np.nan,
                    zero_cluster_upper95=1-.05**(1/count) if events==0 else np.nan,detection_limit=1/bits,
                    analytic_mse=float(analytic[wi,ri].mean()),stop=stop,precision_target_met_for_curve=events>=e['target_errors'] if e['target_errors'] else False,
                    minimum_channels_met=count>=e['min_channels'],seconds=time.perf_counter()-started))
        return rows

    def required_figures(self):
        result=super().required_figures()
        return {name for name in result if not name.startswith('ber_') or ('QPSK' if 'QPSK' in name else '16QAM') in self.config['evaluation']['modulations']}

    def finish(self):
        result=super().finish()
        if self.run_dir:
            from .experiments import save_npz
            save_npz(self.run_dir/'spectral_coupling.npz',**self.coupling_data())
            save_npz(self.run_dir/'discovery.npz',**self.record)
            save_npz(self.run_dir/'grouped_spectra.npz',**self.grouped_spectra_data(),example_native_columns=self.example_columns())
            self.group_table().to_csv(self.run_dir/'groups.csv',index=False)
            self.example_table().to_csv(self.run_dir/'examples.csv',index=False)
            for kind in ('overlay8','half'): self.overlay_table(kind).to_csv(self.run_dir/(kind+'_mapping.csv'),index=False)
            self.provenance['source_sha256'].update({name:sha(Path(__file__).parent/name) for name in ('configurable.py','mps_numerics.py','training.py','entry.py')})
            self.provenance['files_sha256']={p.name:sha(p) for p in self.run_dir.iterdir() if p.is_file() and p.name!='run.json'}
            write_json(self.run_dir/'run.json',self.provenance)
        return result


def start(*,mode,evidence_mode,scenario,model,display,evaluation,training=None,k=None,support_policy='frozen',discovery=None,checkpoint=None,workspace,output_root='local/notebook_runs',save_run=True,scaling_artifacts=None,scaling_bundle=None,review=None):
    config,checkpoint,output_root=validate(mode=mode,evidence_mode=evidence_mode,scenario=scenario,model=model,display=display,evaluation=evaluation,training=training,k=k if k is not None else SIZES.get(model.get('n')),support_policy=support_policy,discovery=discovery,checkpoint=checkpoint,workspace=workspace,output_root=output_root,review=review)
    if type(save_run) is not bool: raise ValueError('SAVE_RUN must be boolean')
    if evidence_mode=='saved_replay':
        from .notebook_workflow import start_workflow
        return start_workflow(mode=mode,evidence_mode=evidence_mode,model=model,display=display,fresh=None,workspace=workspace,save_run=save_run,scaling_artifacts=scaling_artifacts,scaling_bundle=scaling_bundle,review=review,output_root=output_root)
    m.environment(); n=model['n']; k=config['k']; basis_record=None
    if mode=='train_new':
        selected,basis_record=train(config['training']['model'],config['training']['validation_model'],k,config['training'])
    elif checkpoint:
        selected,basis_record=load_checkpoint(checkpoint,n,k)
    else:
        if scaling_artifacts is None: raise FileNotFoundError('Provide local scaling artifacts for archived Learned; no source fallback')
        evidence=load_scaling(workspace,scaling_artifacts,scaling_bundle,n)
        selected=evidence.selected
        basis_record=dict(labels=evidence.records['Learned']['labels'].copy(),permutation=evidence.records['Learned']['permutation'].copy(),q_train=evidence.records['Learned']['q_nominal'].copy())
    bases={'Learned':m.pair(selected.matrix.numpy()),**{name:w for name,w in m.classical(n,GRIDS[n]).items() if name in ('OTFS','AFDM')}}
    masks={}
    if config['support_policy']=='recalibrate':
        d=config['discovery']; ops,_=m.sample(d['channels'],d['model'],('configurable-dd-v1','discovery',d['seed']))
        for name,w in bases.items(): masks[name]=m.mask(coupling(w,ops),k)
        q=coupling(bases['Learned'],ops).cpu().numpy(); labels=groups(q,d['group_threshold'])
        basis_record=dict(labels=labels,permutation=np.argsort(labels,kind='stable'),q_train=q,
            grouping='Connected components from configured discovery multipath coupling; threshold '+str(d['group_threshold']))
        selected=SelectedBasis(torch.from_numpy(m.array(bases['Learned'])),torch.from_numpy(masks['Learned'].cpu().numpy()),
            {**selected.provenance,'source':selected.provenance['source'],'arithmetic':'MPS_FP32','K':k,'support_recalibrated':d}).validate()
    elif mode=='train_new' or checkpoint:
        provenance=selected.provenance
        cfg=provenance['config']; ops,_=m.sample(cfg['batch_channels'],provenance['train_model'],tuple(provenance['train_namespace']))
        masks={name:m.mask(coupling(w,ops),k) for name,w in bases.items()}
        masks['Learned']=m.tensor(selected.fixed_q_mask.numpy()).bool()
        basis_record['grouping']='Train-only multipath connected components; threshold '+str(cfg['group_threshold'])
    else:
        masks={name:m.tensor(evidence.records[name]['mask_nominal']).bool() for name in bases}
    result=FreshWorkflow(config,workspace,save_run,selected,basis_record,masks,output_root)
    result.provenance['evaluation_matrix_content_sha256']={name:hashlib.sha256(m.array(w).tobytes()).hexdigest() for name,w in result.mps_ws.items()}
    result.provenance['precision_conversion']='Archived complex128 arrays, when used, explicitly converted to FP32 real pairs for fresh computation; plotting retains source array precision'
    if mode=='load_existing' and checkpoint is None:
        result.provenance['inputs_sha256']=evidence.input_hashes()
    if mode=='train_new':
        target=checkpoint or (result.run_dir/'selected_basis.npz' if result.run_dir else None)
        if target:
            save_checkpoint(target,selected,basis_record)
            result.provenance['checkpoint']=str(target)
    return result
