"""Public input qualification: runs without optional ignored audit archives."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
import torch
from waveforms.runtime_inputs import RuntimeEvidence, delivered_root, verify_runtime, load_cfo
from waveforms.notebook_workflow import NotebookWorkflow, validate_config
from waveforms.configurable import start, DEFAULT_EVALUATION

ROOT=Path(__file__).resolve().parents[2]
MODEL=dict(n=64,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
DISPLAY=dict(basis_columns=(0,1),example_columns=(0,),snr_max_db=25.)
EVAL={**DEFAULT_EVALUATION,'max_channels':2,'min_channels':2,'blocks_per_channel':1,
      'snrs_db':(0.,12.),'target_errors':0,'diagnostic_atoms':2,'diagnostic_channels':2}
IDENTITIES={64:'11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f',
            128:'cf19e5eb5d02607a68f412c164e979a79654099f33a6f7ac741d55e9b0ca1bdf',
            256:'52de3ac3fca6c34716cd1f7ea9b0c515d34cb1199cee1cd9e5d077c650cf6b6c'}


def replay(evidence,tmp_path):
    model={**MODEL,'n':evidence.n}
    return NotebookWorkflow(validate_config('saved_replay',model,DISPLAY),evidence.root,tmp_path,
                            False,selected_basis=evidence.selected,scaling=evidence)


@pytest.mark.parametrize('n,sizes',[(64,[23,21,20]),(128,[25,44,24,35]),(256,[63,62,79,52])])
def test_delivered_replay_physics_and_uncertainty(n,sizes,tmp_path,monkeypatch):
    import waveforms.notebook_workflow as flow
    monkeypatch.setattr(flow,'ber_point',lambda *a,**kw:pytest.fail('Replay simulated new BER'))
    evidence=RuntimeEvidence(delivered_root(),n);run=replay(evidence,tmp_path)
    assert evidence.identity['original_file_sha256']==IDENTITIES[n]
    h,a=run.generate_channels();data=run.coupling_data()
    assert np.bincount(data['labels']).tolist()==sizes
    assert data['bridge_max']<1e-13 and data['unitarity_max']<1e-13
    np.testing.assert_allclose(data['Q_test'].sum(0),1.,atol=1e-13)
    np.testing.assert_allclose(data['mean_channel_power'].sum(),n,atol=1e-11)
    for name,w in run.ws.items():
        torch.testing.assert_close(a[name],w.mH@h.h@w,atol=0,rtol=0)
        assert (run.masks[name].sum(-1)==3*n//8).all()
    frame=run.evaluate_ber()
    assert len(frame)==96 and set(frame.bits)=={4882432} and set(frame.channels)=={4768}
    assert 'precision_target_met_for_curve' in frame and frame.errors.min()<300
    assert frame.groupby(['modulation','snr_db']).bits.nunique().max()==1
    if n==64:
        rows=frame[(frame.modulation=='QPSK')&(frame.snr_db==25)&(frame.receiver=='full')].set_index('waveform')
        assert rows.loc['Learned','ber']>rows.loc['OTFS','ber']


@pytest.mark.parametrize('n',[64,128,256])
def test_full_audit_and_delivered_outputs_exact(n,scaling_artifacts,tmp_path):
    from waveforms.scaling_evidence import ScalingEvidence
    lean=RuntimeEvidence(delivered_root(),n);full=ScalingEvidence(scaling_artifacts,n)
    a,b=replay(lean,tmp_path),replay(full,tmp_path)
    np.testing.assert_array_equal(a.ws['Learned'],b.ws['Learned'])
    for name in lean.records:
        np.testing.assert_array_equal(lean.records[name]['mask_nominal'],full.records[name]['mask_nominal'])
    for key in lean.records['Learned']:
        if key!='mask_nominal':np.testing.assert_array_equal(lean.records['Learned'][key],full.records['Learned'][key])
    for name,values in a.generate_channels()[1].items():np.testing.assert_array_equal(values,b.generate_channels()[1][name])
    for left,right in ((a.coupling_data(),b.coupling_data()),(a.grouped_spectra_data(),b.grouped_spectra_data())):
        assert left.keys()==right.keys()
        for key in left:np.testing.assert_array_equal(left[key],right[key])
    pd.testing.assert_frame_equal(a.evaluate_ber(),b.evaluate_ber())


@pytest.mark.parametrize('damage',['missing','corrupt','symlink','format','inventory'])
def test_delivered_missing_or_altered_fails_without_fallback(tmp_path,damage):
    root=tmp_path/'data';shutil.copytree(delivered_root(),root)
    file=root/'dd_N64.npz'
    if damage=='missing':file.unlink()
    elif damage=='corrupt':file.write_bytes(file.read_bytes()[:-1]+b'x')
    elif damage=='symlink':file.unlink();file.symlink_to(delivered_root()/'dd_N64.npz')
    else:
        manifest=json.loads((root/'manifest.json').read_text())
        if damage=='format':manifest['format']='unknown'
        else:manifest['files'].pop('dd_N256.npz')
        (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Missing or altered|format/file inventory'):verify_runtime(root)


def test_delivered_size_rejects_resize():
    with pytest.raises(ValueError,match='Available sizes'):RuntimeEvidence(delivered_root(),512)


@pytest.mark.parametrize('n',[64,128,256])
def test_fresh_delivered_never_reads_replay_and_config_changes_propagate(n,tmp_path,monkeypatch):
    monkeypatch.setattr(RuntimeEvidence,'ber',lambda *a:pytest.fail('Fresh used historical BER'))
    kwargs=dict(mode='load_existing',evidence_mode='fresh',scenario='dd_v1',
                model={**MODEL,'n':n,'paths':2,'delay_range':(.1,.9),'doppler_range':(-.1,.2)},
                display=DISPLAY,evaluation=EVAL,training=None,k=3*n//8,workspace=tmp_path,
                scaling_artifacts=delivered_root(),save_run=False)
    a=start(**kwargs);b=start(**kwargs)
    frame=a.evaluate_ber();second=b.evaluate_ber()
    pd.testing.assert_frame_equal(frame.drop(columns='seconds'),second.drop(columns='seconds'))
    assert set(frame.snr_db)=={0.,12.} and set(frame.channels)=={2}
    assert set(frame[frame.modulation=='QPSK'].bits)=={4*n}
    assert a.generate_channels()[0].tau.shape[-1]==2
    with pytest.raises(ValueError,match='Saved BER'):
        start(**(kwargs|dict(evidence_mode='saved_replay',evaluation=None)))


def test_delivered_cfo_rectangular_model_and_fresh_budget(tmp_path):
    from waveforms.cfo_workflow import start_cfo
    from waveforms import numerics as m
    w,r,checks=load_cfo(delivered_root())
    assert w.shape==(64,45) and r.shape==(45,64) and not torch.allclose(r,w.mH,atol=1e-3,rtol=0)
    assert checks['clean_identity_loss']<.001
    evaluation={k:v for k,v in EVAL.items() if k not in ('diagnostic_atoms','diagnostic_channels')}
    run=start_cfo(mode='load_existing',evidence_mode='fresh',model=dict(n=64,streams=45,cfo_range=(.2,.3)),
                  display=DISPLAY,evaluation=evaluation,workspace=tmp_path,artifacts=delivered_root(),bundle=None,save_run=False)
    h,a=run.generate_channels()
    expected=np.diag(np.exp(2j*np.pi*h.eps[0]*np.arange(64)/64))
    np.testing.assert_allclose(a['Learned'][0],r.numpy()@expected@w.numpy(),atol=3e-6)
    frame=run.evaluate_ber()
    assert set(frame.receiver)=={'historical_R'} and set(frame[frame.modulation=='QPSK'].bits)=={180}
    assert run.w[0].device.type=='cpu' and run.w[0].dtype==torch.float32


def test_delivered_cfo_exact_original_payload(artifacts):
    from waveforms.residual_cfo import load_qualified
    a,r,checks=load_cfo(delivered_root());b,s,original=load_qualified(artifacts)
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(r,s,atol=0,rtol=0)
    assert all(checks[key]==value for key,value in original.items())


def test_training_and_own_checkpoint_never_touch_delivered_inputs(tmp_path,monkeypatch):
    from waveforms import runtime_inputs
    from waveforms.training import DEFAULT_TRAINING,save_checkpoint
    monkeypatch.setattr(runtime_inputs,'verify_runtime',lambda *a:pytest.fail('Training/checkpoint read pretrained inputs'))
    training={**DEFAULT_TRAINING,'model':MODEL,'validation_model':MODEL,'iterations':2,
              'validate_every':1,'batch_channels':2,'validation_channels':2}
    args=dict(mode='train_new',evidence_mode='fresh',scenario='dd_v1',model=MODEL,display=DISPLAY,
              evaluation=EVAL,training=training,k=24,workspace=tmp_path,save_run=False)
    a=start(**args);path=tmp_path/'local/new.npz';save_checkpoint(path,a.selected_basis,a.record)
    b=start(**(args|dict(mode='load_existing',training=None,checkpoint=path)))
    pd.testing.assert_frame_equal(a.evaluate_ber().drop(columns='seconds'),b.evaluate_ber().drop(columns='seconds'))
    np.testing.assert_array_equal(a.ws['Learned'],b.ws['Learned'])
