"""Bounded unitary DD concentration trainer, independent of historical recipes.

Selection: minimum held-out multipath leakage using train-only fixed row support.
One global W, including step zero; no BER/SNR-dependent selection.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from . import mps_numerics as m
from .artifacts import sha, write_json
from .basis_source import SelectedBasis
from .scaling_evidence import GRIDS

DEFAULT_TRAINING = dict(seed=101, validation_seed=202, iterations=20, learning_rate=.02,
    layers=4, initialization='OFDM', batch_channels=8, validation_channels=8,
    validate_every=5, max_seconds=120., group_threshold=.15)


def groups(q, threshold):
    """Connected components of symmetrized relative coupling; native IDs.

    Threshold is explicit. Components may be one diffuse group; no target count.
    """
    affinity=q+q.T
    np.fill_diagonal(affinity,0.)
    scale=affinity.max(axis=1)
    edges=(affinity>0)&(affinity>=threshold*np.maximum(scale[:,None],scale[None,:]))
    labels=np.full(len(q),-1,dtype=np.int64); g=0
    for j in range(len(q)):
        if labels[j]>=0: continue
        todo=[j]; labels[j]=g
        while todo:
            i=todo.pop()
            for k in np.flatnonzero(edges[i]):
                if labels[k]<0: labels[k]=g; todo.append(int(k))
        g+=1
    return labels


def synthesis(theta, initial):
    w=initial; f=m.dft(initial[0].shape[0])
    for layer,angles in enumerate(theta):
        p=m.phase(angles)
        if layer%2:
            w=m.mm(m.adj(f),m.multiply((p[0][:,None],p[1][:,None]),m.mm(f,w)))
        else:
            w=m.multiply((p[0][:,None],p[1][:,None]),w)
    return w


def coupling(w,operators):
    return m.power(m.effective(w,operators)).mean(0)


def leakage(q,support):
    return 1-(q*support).sum()/q.sum().clamp_min(1e-12)


def train(model, validation_model, k, cfg):
    started=time.perf_counter(); env=m.environment(); n=model['n']
    train_ns=('configurable-dd-v1','train',cfg['seed'])
    validation_ns=('configurable-dd-v1','validation',cfg['validation_seed'])
    operators,_=m.sample(cfg['batch_channels'],model,train_ns)
    validation,_=m.sample(cfg['validation_channels'],validation_model,validation_ns)
    initial=m.classical(n,GRIDS[n])[cfg['initialization']]
    theta=torch.nn.Parameter(.05*m.draw((cfg['layers'],n),(*train_ns,'initialization'),normal=True))
    optimizer=torch.optim.Adam([theta],lr=cfg['learning_rate'])
    trace=[]; best=None; best_score=float('inf'); completed=0; stop='iterations'

    def assess(step, train_loss=None):
        nonlocal best,best_score
        with torch.no_grad():
            w=synthesis(theta,initial); q=coupling(w,operators); support=m.mask(q,k)
            score=float(leakage(coupling(w,validation),support))
            unit=m.unitarity(w)
            if not np.isfinite(score) or unit>5e-4: raise RuntimeError('Training validation finite/unitarity gate failed')
            trace.append(dict(step=step,training_loss=float(leakage(q,support)),
                              last_update_loss=train_loss,
                              validation_leakage=score,unitarity_max=unit))
            if score<best_score:
                best_score=score
                best=dict(w=tuple(x.detach().clone() for x in w),q=q.clone(),mask=support.clone(),step=step,
                          theta=theta.detach().clone())
    assess(0)
    for step in range(1,cfg['iterations']+1):
        if time.perf_counter()-started>=cfg['max_seconds']:
            stop='time_budget'; break
        optimizer.zero_grad(set_to_none=True)
        q=coupling(synthesis(theta,initial),operators)
        loss=leakage(q,m.mask(q,k))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss; no fallback')
        loss.backward()
        if theta.grad is None or not torch.isfinite(theta.grad).all(): raise RuntimeError('Nonfinite/missing gradient')
        optimizer.step(); completed=step
        if step%cfg['validate_every']==0 or step==cfg['iterations']:
            assess(step,float(loss.detach()))
    if trace[-1]['step']!=completed: assess(completed)
    m.synchronize()
    w=m.array(best['w']); mask=best['mask'].cpu().numpy(); q=best['q'].cpu().numpy()
    meta=dict(source='new DD phase-stack unitary concentration training',basis_mode='train_new',
              arithmetic=m.DEVICE.type.upper()+'_FP32',N=n,K=k,training_status=f'Trained {completed} iterations; selected step {best["step"]}',
              objective='1 - captured mean normalized multipath coupling energy in train-only fixed row support',
              parameterization='alternating time/Fourier phase layers; unitary by construction',
              selection='minimum validation multipath leakage with train-only mask; earliest tie; includes step zero; no BER selection',
              config=cfg,train_model=model,validation_model=validation_model,train_namespace=train_ns,
              validation_namespace=validation_ns,steps_executed=completed,selected_step=best['step'],
              validation_leakage=best_score,stop=stop,seconds=time.perf_counter()-started,environment=env,
              trace=trace,limitations='Bounded surrogate fit, not historical training replay or BER guarantee')
    selected=SelectedBasis(torch.from_numpy(w),torch.from_numpy(mask),meta).validate()
    labels=groups(q,cfg['group_threshold'])
    record=dict(labels=labels,permutation=np.argsort(labels,kind='stable'),q_train=q)
    return selected,record


def save_checkpoint(path,selected,record):
    path=Path(path); sidecar=path.with_suffix('.json')
    if path.suffix!='.npz': raise ValueError('Checkpoint requires .npz suffix')
    if path.exists() or sidecar.exists(): raise FileExistsError('Checkpoint exists; choose a new path (no overwrite/resume)')
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create protects prior inference checkpoints, including concurrent runs.
    with path.open('xb') as stream:
        np.savez_compressed(stream,W=selected.matrix.numpy(),mask=selected.fixed_q_mask.numpy(),**record)
    write_json(sidecar,dict(format='configurable-dd-v1',sha256=sha(path),provenance=selected.provenance))
    return path


def load_checkpoint(path,n,k):
    path=Path(path); meta=json.loads(path.with_suffix('.json').read_text())
    if meta.get('format')!='configurable-dd-v1' or sha(path)!=meta.get('sha256'):
        raise ValueError('Checkpoint format/hash mismatch')
    provenance=meta['provenance']
    if provenance['N']!=n or provenance['K']!=k: raise ValueError('Checkpoint N/K differs; no resizing or support substitution')
    with np.load(path,allow_pickle=False) as data:
        selected=SelectedBasis(torch.from_numpy(data['W'].copy()),torch.from_numpy(data['mask'].copy()),
            {**provenance,'source':str(path),'basis_mode':'load_existing','checkpoint_sha256':meta['sha256']}).validate()
        record={key:data[key].copy() for key in ('labels','permutation','q_train')}
    if not np.array_equal(record['permutation'],np.argsort(record['labels'],kind='stable')):
        raise ValueError('Checkpoint group permutation differs')
    return selected,record
