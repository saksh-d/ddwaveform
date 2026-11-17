"""Explicit CPU FP32 qualification; same physical models, distinct backend RNG."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from waveforms import mps_numerics as m
from waveforms.configurable import start,DEFAULT_EVALUATION
from waveforms.training import DEFAULT_TRAINING,save_checkpoint
from waveforms.runtime_inputs import delivered_root,RuntimeEvidence
from waveforms.cfo_workflow import start_cfo

MODEL=dict(n=64,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
DISPLAY=dict(basis_columns=(0,1),example_columns=(0,),snr_max_db=25.)
EVAL={**DEFAULT_EVALUATION,'max_channels':2,'min_channels':2,'blocks_per_channel':1,
      'snrs_db':(0.,12.),'target_errors':0,'diagnostic_atoms':2,'diagnostic_channels':2}
TRAIN={**DEFAULT_TRAINING,'model':MODEL,'validation_model':MODEL,'iterations':3,
       'validate_every':1,'batch_channels':3,'validation_channels':3}


@pytest.fixture(autouse=True)
def explicit_cpu(monkeypatch):
    m.select_backend('cpu')
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    monkeypatch.setattr(torch.mps,'manual_seed',lambda *a:pytest.fail('CPU called MPS RNG'))
    monkeypatch.setattr(torch.mps,'synchronize',lambda *a:pytest.fail('CPU called MPS synchronization'))
    yield
    m.select_backend('mps')


def settings(tmp_path,**extra):
    return dict(mode='load_existing',evidence_mode='fresh',scenario='dd_v1',model=MODEL,
                display=DISPLAY,evaluation=EVAL,training=None,k=24,workspace=tmp_path,
                scaling_artifacts=delivered_root(),save_run=False,backend='cpu')|extra


@pytest.mark.parametrize('n',[64,128,256])
def test_cpu_fp32_physics_reference_and_seed_isolation(n):
    env=m.environment();assert env['device']=='cpu' and env['real_dtype']=='torch.float32' and not env['fallback']
    torch.random.default_generator.manual_seed(918);state=torch.get_rng_state().clone()
    a,info=m.sample(2,{**MODEL,'n':n},('cpu-reference',n))
    b,_=m.sample(2,{**MODEL,'n':n},('cpu-reference',n));np.testing.assert_array_equal(m.array(a),m.array(b))
    assert torch.equal(state,torch.get_rng_state())
    assert a[0].dtype==torch.float32 and a[0].device.type=='cpu'
    np.testing.assert_allclose(m.power(a).sum((-2,-1)),n,rtol=3e-6)
    k=np.arange(n);f=np.exp(-2j*np.pi*np.outer(k,k)/n)/np.sqrt(n)
    op=m.array(m.atoms(m.tensor([[1.25]]),m.tensor([[-.3]]),n))[0,0]
    expected=np.exp(2j*np.pi*.3*k/n)[:,None]*(f.conj().T@(np.exp(-2j*np.pi*1.25*k/n)[:,None]*f))
    np.testing.assert_allclose(op,expected,atol=1e-5,rtol=0)
    for w in m.classical(n, {64:(8,8),128:(8,16),256:(16,16)}[n]).values():assert m.unitarity(w)<1e-4
    w=m.pair(np.eye(n));a=m.effective(w,a);g=m.receiver(a,.2)
    x=m.array(a);ref=np.linalg.solve(x.conj().swapaxes(-1,-2)@x+.2*np.eye(n),x.conj().swapaxes(-1,-2))
    np.testing.assert_allclose(m.array(g),ref,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('n',[64,128,256])
def test_cpu_frozen_dd_fresh_budget_and_no_saved_ber(n,tmp_path,monkeypatch):
    monkeypatch.setattr(RuntimeEvidence,'ber',lambda *a:pytest.fail('CPU fresh used saved BER'))
    args=settings(tmp_path,model={**MODEL,'n':n,'paths':2,'delay_range':(.1,.9)},k=3*n//8)
    a=start(**args);b=start(**args)
    assert a.environment['device']=='cpu' and a.config['backend']=='cpu'
    assert all(w[0].device.type=='cpu' and w[0].dtype==torch.float32 for w in a.mps_ws.values())
    np.testing.assert_array_equal(a.ws['Learned'],RuntimeEvidence(delivered_root(),n).selected.matrix)
    frame=a.evaluate_ber();pd.testing.assert_frame_equal(frame.drop(columns='seconds'),b.evaluate_ber().drop(columns='seconds'))
    assert set(frame.channels)=={2} and set(frame[frame.modulation=='QPSK'].bits)=={4*n}
    assert set(frame.receiver)=={'fixed','full'} and a.coupling_data()['bridge_max']<2e-4
    assert 'CPU FP32' in a.provenance['BER_status']


def test_cpu_actual_training_update_split_and_checkpoint(tmp_path):
    args=settings(tmp_path,mode='train_new',training=TRAIN)
    a=start(**args);b=start(**args)
    np.testing.assert_array_equal(a.ws['Learned'],b.ws['Learned'])
    meta=a.selected_basis.provenance
    assert meta['arithmetic']=='CPU_FP32' and meta['environment']['device']=='cpu'
    assert meta['steps_executed']==3 and meta['train_namespace']!=meta['validation_namespace']
    assert abs(meta['trace'][0]['training_loss']-meta['trace'][-1]['training_loss'])>1e-7
    assert meta['selected_step']==min(meta['trace'],key=lambda x:x['validation_leakage'])['step']
    path=tmp_path/'local/cpu_basis.npz';save_checkpoint(path,a.selected_basis,a.record)
    c=start(**settings(tmp_path,checkpoint=path))
    np.testing.assert_array_equal(c.ws['Learned'],a.ws['Learned'])
    pd.testing.assert_frame_equal(a.evaluate_ber().drop(columns='seconds'),c.evaluate_ber().drop(columns='seconds'))
    changed=start(**(args|dict(model={**MODEL,'paths':1},evaluation={**EVAL,'seed':909})))
    np.testing.assert_array_equal(a.ws['Learned'],changed.ws['Learned'])
    assert not np.array_equal(a.generate_channels()[0].h,changed.generate_channels()[0].h)
    m.select_backend('mps')
    with pytest.raises(RuntimeError,match='Backend changed'):a.generate_channels()


def test_cpu_cfo_distinct_rectangular_model_and_noise(tmp_path):
    evaluation={k:v for k,v in EVAL.items() if k not in ('diagnostic_atoms','diagnostic_channels')}
    args=dict(mode='load_existing',evidence_mode='fresh',model=dict(n=64,streams=45,cfo_range=(.2,.3)),
              display=DISPLAY,evaluation=evaluation,workspace=tmp_path,artifacts=delivered_root(),bundle=None,save_run=False,backend='cpu')
    a=start_cfo(**args);b=start_cfo(**args);h,aa=a.generate_channels();w,r=m.array(a.w),m.array(a.r)
    assert w.shape==(64,45) and r.shape==(45,64) and not np.allclose(r,w.conj().T,atol=1e-3)
    expected=np.diag(np.exp(2j*np.pi*h.eps[0]*np.arange(64)/64))
    np.testing.assert_allclose(aa['Learned'][0],r@expected@w,atol=3e-6)
    frame=a.evaluate_ber();pd.testing.assert_frame_equal(frame.drop(columns='seconds'),b.evaluate_ber().drop(columns='seconds'))
    assert set(frame.receiver)=={'historical_R'} and set(frame[frame.modulation=='QPSK'].bits)=={180}
    assert 'CPU FP32' in a.provenance['BER_status']
    z=(m.draw((8192,64),('CPU-CFO-noise','real'),normal=True)/np.sqrt(2),m.draw((8192,64),('CPU-CFO-noise','imag'),normal=True)/np.sqrt(2))
    out=m.array(m.mm(z,m.transpose(a.r)))
    np.testing.assert_allclose(out.T@out.conj()/len(out),r@r.conj().T,atol=.07)


def test_cpu_not_implicit_and_backend_validation(tmp_path):
    with pytest.raises(RuntimeError,match='no CPU fallback'):start(**(settings(tmp_path)|dict(backend='mps')))
    with pytest.raises(ValueError,match='BACKEND'):start(**(settings(tmp_path)|dict(backend='cuda')))
    # Default replay remains CPU reference arithmetic even when MPS is unavailable.
    run=start(**settings(tmp_path,evidence_mode='saved_replay',model=MODEL,evaluation=None,backend='mps'))
    assert run.environment['device']=='cpu' and run.environment['real_dtype']=='torch.float64'
    assert len(run.evaluate_ber())==96
