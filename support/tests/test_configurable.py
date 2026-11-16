"""MPS numerical qualification plus configuration, splits, checkpoint and failure tests."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from waveforms import mps_numerics as m
from waveforms.configurable import start,validate,DEFAULT_EVALUATION
from waveforms.training import DEFAULT_TRAINING,train,save_checkpoint,load_checkpoint,synthesis,coupling,leakage,groups

MODEL=dict(n=64,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
DISPLAY=dict(basis_columns=(0,1),example_columns=(0,),snr_max_db=25.)
EVAL={**DEFAULT_EVALUATION,'max_channels':2,'min_channels':2,'batch_channels':2,'blocks_per_channel':1,'snrs_db':(0.,12.),'target_errors':0,'diagnostic_atoms':2,'diagnostic_channels':2}
TRAIN={**DEFAULT_TRAINING,'model':MODEL,'validation_model':MODEL,'iterations':3,'validate_every':1,'batch_channels':3,'validation_channels':3}


def settings(tmp_path,**overrides):
    return dict(mode='train_new',evidence_mode='fresh',scenario='dd_v1',model=MODEL,display=DISPLAY,evaluation=EVAL,training=TRAIN,k=24,support_policy='frozen',discovery=None,checkpoint=None,workspace=tmp_path,output_root='local/runs',review=None,**{})|overrides


def config(tmp_path,**overrides): return validate(**settings(tmp_path,**overrides))


@pytest.mark.parametrize('key,value,match',[
    ('model',{**MODEL,'n':96},'N=64'),('model',{**MODEL,'paths':7},'paths'),
    ('model',{**MODEL,'delay_range':(2,1)},'increasing'),('model',{**MODEL,'doppler_range':(-3,1)},'increasing'),
    ('k',0,'K'),('evidence_mode','saved_replay','requires fresh'),
    ('evaluation',{**EVAL,'seed':101},'seeds must be distinct'),
    ('evaluation',{**EVAL,'snrs_db':(12,0)},'strictly increasing'),
    ('evaluation',{**EVAL,'modulations':('BPSK',)},'Modulations'),
    ('evaluation',{**EVAL,'max_seconds_per_point':float('nan')},'finite'),
    ('training',{**TRAIN,'learning_rate':0},'learning_rate'),
    ('training',{**TRAIN,'validation_model':{**MODEL,'n':128}},'N must agree'),
    ('training',{**TRAIN,'unknown':1},'Unsupported'),
    ('checkpoint','../foreign.npz','worktree local'),('support_policy','realized','support_policy')])
def test_reject_invalid_before_work(tmp_path,key,value,match):
    with pytest.raises((ValueError,NotImplementedError),match=match): config(tmp_path,**{key:value})
    assert list(tmp_path.iterdir())==[]


def test_no_mps_fallback(monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(RuntimeError,match='Disable'): m.environment()
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK')
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no CPU fallback'): m.environment()


@pytest.mark.parametrize('n',[64,128,256])
def test_dd_deterministic_reference_and_classical_unitarity(n):
    m.environment(); f=m.array(m.dft(n)); index=np.arange(n)
    ref=np.exp(-2j*np.pi*np.outer(index,index)/n)/np.sqrt(n)
    np.testing.assert_allclose(f,ref,atol=3e-6,rtol=0)
    tau=m.tensor([[1.25]]); nu=m.tensor([[-.3]])
    a=m.array(m.atoms(tau,nu,n))[0,0]
    delay=ref.conj().T@(np.exp(-2j*np.pi*1.25*index/n)[:,None]*ref)
    expected=np.exp(2j*np.pi*.3*index/n)[:,None]*delay
    np.testing.assert_allclose(a,expected,atol=1e-5,rtol=0)
    from waveforms.scaling_evidence import GRIDS
    for w in m.classical(n,GRIDS[n]).values(): assert m.unitarity(w)<1e-4
    model={**MODEL,'n':n}; h,info=m.sample(2,model,('test','normalized',n))
    np.testing.assert_allclose(m.power(h).sum((-2,-1)).cpu(),n,rtol=3e-6)
    assert h[0].device.type=='mps' and h[0].dtype==torch.float32
    same,_=m.sample(2,model,('test','normalized',n)); np.testing.assert_array_equal(m.array(h),m.array(same))


def test_receiver_reference_noise_mse_and_roundtrip():
    m.environment(); n=8; i=np.arange(n)
    a=np.diag(.3+i/n)+.03j*np.ones((n,n)); ap=m.pair(a); variance=.2
    for support in (None,np.eye(n,dtype=bool)):
        g=m.receiver(ap,variance,m.tensor(support) if support is not None else None)
        model=a if support is None else a*support
        ref=np.linalg.solve(model.conj().T@model+variance*np.eye(n),model.conj().T)
        np.testing.assert_allclose(m.array(g),ref,atol=3e-6,rtol=0)
    for mod in ('QPSK','16QAM'):
        bits=m.draw((2,3,n,2 if mod=='QPSK' else 4),('roundtrip',mod),bits=True)
        assert torch.equal(m.decode(m.symbols(bits,mod),mod),bits)
    # Full receiver average MSE invariant under unitary change of coordinates.
    h,_=m.sample(2,MODEL,('full-MSE-invariance',))
    from waveforms.scaling_evidence import GRIDS
    metrics=[]
    for w in m.classical(64,GRIDS[64]).values():
        a=m.effective(w,h); g=m.receiver(a,variance); ga=m.mm(g,a)
        metrics.append(float((m.power((ga[0]-m.eye(64)[0],ga[1])).sum(-1)+variance*m.power(g).sum(-1)).mean()))
    assert max(metrics)-min(metrics)<3e-5


def test_phase_gradient_finite_difference_and_actual_update():
    m.environment(); theta=m.draw((2,64),('gradient',),normal=True)*.05; theta.requires_grad_()
    initial=m.classical(64,(8,8))['OFDM']; op,_=m.sample(3,MODEL,('grad-bank',),atomic=True)
    def loss(t):
        q=coupling(synthesis(t,initial),op)
        return leakage(q,m.mask(q,24))
    out=loss(theta); out.backward(); analytic=float(theta.grad[0,7]); delta=.01
    plus=theta.detach().clone();minus=theta.detach().clone();plus[0,7]+=delta;minus[0,7]-=delta
    numerical=float((loss(plus)-loss(minus))/(2*delta))
    assert np.isfinite(analytic) and abs(analytic-numerical)<1e-4
    a,_=train(MODEL,MODEL,24,{**DEFAULT_TRAINING,'iterations':2,'validate_every':1,'batch_channels':3,'validation_channels':3})
    trace=a.provenance['trace']; assert len(trace)==3 and trace[-1]['step']==2
    assert abs(trace[0]['training_loss']-trace[-1]['training_loss'])>1e-7


def test_train_checkpoint_seeds_split_and_evaluation_equivalence(tmp_path):
    args=settings(tmp_path); a=start(**args,save_run=False)
    b=start(**args,save_run=False)
    np.testing.assert_array_equal(a.ws['Learned'],b.ws['Learned'])
    assert a.selected_basis.provenance['trace']==b.selected_basis.provenance['trace']
    assert a.selected_basis.provenance['selected_step']==min(a.selected_basis.provenance['trace'],key=lambda x:x['validation_leakage'])['step']
    assert a.selected_basis.provenance['train_namespace']!=a.selected_basis.provenance['validation_namespace']
    checkpoint=tmp_path/'local/selected.npz'; save_checkpoint(checkpoint,a.selected_basis,a.record)
    loaded,record=load_checkpoint(checkpoint,64,24)
    np.testing.assert_array_equal(loaded.matrix,a.ws['Learned']); np.testing.assert_array_equal(loaded.fixed_q_mask,a.masks['Learned'])
    with pytest.raises(FileExistsError): save_checkpoint(checkpoint,a.selected_basis,a.record)
    with pytest.raises(ValueError,match='N/K differs'): load_checkpoint(checkpoint,128,48)
    c=start(**settings(tmp_path,mode='load_existing',training=None,checkpoint=checkpoint),save_run=False)
    pd.testing.assert_frame_equal(a.evaluate_ber().drop(columns='seconds'),c.evaluate_ber().drop(columns='seconds'))
    for key in a.counts: np.testing.assert_array_equal(a.counts[key]['errors'],c.counts[key]['errors'])
    np.testing.assert_array_equal(a.generate_channels()[0].h,c.generate_channels()[0].h)
    # Evaluation changes cannot alter W, training selection or train-only grouping.
    modified={**MODEL,'paths':2,'delay_range':(.1,.9),'doppler_range':(-.1,.2)}
    d=start(**settings(tmp_path,model=modified,evaluation={**EVAL,'seed':909,'snrs_db':(1.,11.)}),save_run=False)
    np.testing.assert_array_equal(d.ws['Learned'],a.ws['Learned'])
    np.testing.assert_array_equal(d.reference_views['labels'],a.reference_views['labels'])
    assert not np.array_equal(d.generate_channels()[0].h,a.channels.h)
    assert set(d.evaluate_ber().snr_db)=={1.,11.}
    values=d.counts['QPSK_1dB']; assert values['tau'].shape==(2,2) and np.max(values['tau'])<=.9
    meta=json.loads(checkpoint.with_suffix('.json').read_text()); meta['sha256']='bad'; checkpoint.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='hash mismatch'): load_checkpoint(checkpoint,64,24)


@pytest.mark.parametrize('n',[128,256])
def test_short_actual_training_scaling(n,tmp_path):
    model={**MODEL,'n':n}; t={**TRAIN,'model':model,'validation_model':model,'iterations':2,'validate_every':1,'batch_channels':2,'validation_channels':2}
    a=start(**settings(tmp_path,model=model,training=t,k=3*n//8),save_run=False)
    assert a.ws['Learned'].shape==(n,n) and a.selected_basis.provenance['steps_executed']==2
    assert a.masks['Learned'].sum(-1).unique().item()==3*n//8
    assert a.coupling_data()['bridge_max']<2e-4
    frame=a.evaluate_ber(); assert frame.shape[0]==24 and set(frame[frame.modulation=='16QAM'].bits)=={2*n*4}


def test_error_time_channel_budgets_and_zero_bounds(tmp_path):
    a=start(**settings(tmp_path,evaluation={**EVAL,'max_channels':4,'target_errors':1,'snrs_db':(0.,1.)}),save_run=False)
    frame=a.evaluate_ber(); assert set(frame.stop)=={'error_target'} and set(frame.channels)=={2}
    for _,point in frame.groupby(['modulation','snr_db']): assert point.channels.nunique()==point.bits.nunique()==1
    b=start(**settings(tmp_path,evaluation={**EVAL,'max_channels':4,'min_channels':4,'max_seconds_per_point':.01}),save_run=False)
    assert 'time_budget' in set(b.evaluate_ber().stop)
    c=start(**settings(tmp_path,evaluation={**EVAL,'snrs_db':(30.,40.),'modulations':('QPSK',)},display={**DISPLAY,'snr_max_db':40.}),save_run=False)
    frame=c.evaluate_ber(); zeros=frame[frame.errors==0]
    assert not zeros.empty and zeros.ci_low.isna().all() and (zeros.zero_cluster_upper95>zeros.detection_limit).all()


@pytest.mark.artifacts
def test_archived_fresh_recalibration_and_replay_guard(tmp_path,scaling_artifacts,monkeypatch):
    # Fresh may never call the saved BER loader, including changed channel/K.
    from waveforms.scaling_evidence import ScalingEvidence
    monkeypatch.setattr(ScalingEvidence,'ber',lambda self:pytest.fail('stale saved BER used'))
    args=settings(Path(__file__).resolve().parents[2],mode='load_existing',training=None,scaling_artifacts=scaling_artifacts)
    a=start(**args,save_run=False); a.evaluate_ber()
    with pytest.raises(ValueError,match='Archived frozen K'): start(**(args|{'k':16}),save_run=False)
    discovery=dict(model=MODEL,seed=404,channels=3,group_threshold=.15)
    b=start(**(args|{'k':16,'support_policy':'recalibrate','discovery':discovery}),save_run=False)
    assert b.masks['Learned'].sum(-1).unique().item()==16
    assert b.selected_basis.provenance['support_recalibrated']['seed']==404
    for kwargs in (dict(k=16),dict(checkpoint=tmp_path/'local/a.npz'),dict(support_policy='recalibrate')):
        with pytest.raises(ValueError,match='Saved BER requires'):
            config(tmp_path,mode='load_existing',training=None,evidence_mode='saved_replay',evaluation=None,**kwargs)


def test_group_count_derived_and_native_mapping():
    q=np.eye(6);q[0,1]=q[1,0]=.5;q[2,3]=q[3,2]=.5;q[4,5]=q[5,4]=.5
    np.testing.assert_array_equal(groups(q,.2),[0,0,1,1,2,2])
    assert len(np.unique(groups(np.ones((6,6)),.2)))==1
    assert len(np.unique(groups(np.eye(6),.2)))==6


def test_training_channel_pathcount_validation_law_seed_and_k_propagate(tmp_path):
    a=start(**settings(tmp_path,k=16),save_run=False)
    assert np.all(a.selected_basis.fixed_q_mask.numpy().sum(-1)==16)
    b=start(**settings(tmp_path,k=16,training={**TRAIN,'model':{**MODEL,'paths':1}}),save_run=False)
    assert np.max(np.abs(a.record['q_train']-b.record['q_train']))>1e-4
    c=start(**settings(tmp_path,k=16,training={**TRAIN,'validation_model':{**MODEL,'doppler_range':(-1.5,1.5)}}),save_run=False)
    assert abs(a.selected_basis.provenance['trace'][0]['validation_leakage']-c.selected_basis.provenance['trace'][0]['validation_leakage'])>1e-6
    d=start(**settings(tmp_path,k=16,training={**TRAIN,'seed':505}),save_run=False)
    assert not np.array_equal(a.ws['Learned'],d.ws['Learned'])


def test_training_time_budget_retains_selected_state(tmp_path):
    a=start(**settings(tmp_path,training={**TRAIN,'iterations':1000,'max_seconds':.01}),save_run=False)
    assert a.selected_basis.provenance['stop']=='time_budget'
    assert a.selected_basis.provenance['steps_executed']<1000
    assert np.isfinite(a.ws['Learned'].numpy()).all()


def test_fresh_group_plot_does_not_magnify_unitary_roundoff(tmp_path):
    from waveforms.coupling import plot_groups
    import matplotlib.pyplot as plt
    run=start(**settings(tmp_path),save_run=False)
    data=run.coupling_data(); fig=plot_groups(data)
    profile=fig.axes[5]
    assert profile.get_ylim()[0]==0
    assert profile.yaxis.get_major_formatter().get_useOffset() is False
    assert 'evaluation-law channels' in fig.axes[2].get_title()
    plt.close(fig)


def test_single_row_complex_basis_labels_fit_canvas(tmp_path):
    from waveforms.figures import plot_basis,style
    import matplotlib.pyplot as plt
    run=start(**settings(tmp_path),save_run=False);style()
    fig=plot_basis(run.bases,(0,1),True);fig.canvas.draw();renderer=fig.canvas.get_renderer()
    artists=[fig._supylabel,fig._supxlabel]+[ax.title for ax in fig.axes]
    for artist in artists:
        box=artist.get_window_extent(renderer)
        assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.x1 and box.y1<=fig.bbox.y1
    plt.close(fig)
