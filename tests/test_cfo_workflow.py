from pathlib import Path
import numpy as np
import pytest
import torch
from advisor import mps_numerics as m
from advisor.cfo_workflow import start_cfo,operator
from advisor.configurable import DEFAULT_EVALUATION

MODEL=dict(n=64,streams=45,cfo_range=(-.1,.1))
DISPLAY=dict(basis_columns=(0,22,44),snr_max_db=25.)
EVAL={key:value for key,value in DEFAULT_EVALUATION.items() if key not in ('diagnostic_atoms','diagnostic_channels')}
EVAL.update(max_channels=2,min_channels=2,blocks_per_channel=1,snrs_db=(0.,12.),target_errors=0)


def test_cfo_rejections_before_io(tmp_path):
    args=dict(mode='load_existing',evidence_mode='fresh',model=MODEL,display=DISPLAY,evaluation=EVAL,workspace=tmp_path,artifacts=tmp_path/'local/missing',bundle=None)
    for extra,exception,match in [(dict(mode='train_new'),NotImplementedError,'unsupported'),
        (dict(evidence_mode='saved_replay'),ValueError,'requires fresh'),
        (dict(model={**MODEL,'n':128}),ValueError,'dimensions'),
        (dict(model={**MODEL,'streams':24}),ValueError,'dimensions'),
        (dict(model={**MODEL,'cfo_range':(-1.,1.)}),ValueError,'finite'),
        (dict(k=24),ValueError,'excludes DD'),(dict(review={}),ValueError,'excludes DD'),
        (dict(display={**DISPLAY,'basis_columns':(45,)}),ValueError,'stream')]:
        with pytest.raises(exception,match=match): start_cfo(**(args|extra))
    assert list(tmp_path.iterdir())==[]


@pytest.mark.artifacts
def test_cfo_actual_rectangular_pipeline_noise_and_reproducibility(artifacts,tmp_path):
    # Local provisioned fixture is read-only; path confined to worktree local/.
    root=Path(__file__).resolve().parents[1]
    args=dict(mode='load_existing',evidence_mode='fresh',model=MODEL,display=DISPLAY,evaluation=EVAL,workspace=root,artifacts=artifacts,bundle=None,save_run=False)
    run=start_cfo(**args); h,a=run.generate_channels()
    w=m.array(run.w); r=m.array(run.r); eps=h.eps
    expected=np.diag(np.exp(2j*np.pi*eps[0]*np.arange(64)/64))
    np.testing.assert_allclose(h.h[0],expected,atol=1e-6)
    np.testing.assert_allclose(a['Learned'][0],r@expected@w,atol=3e-6)
    assert w.shape==(64,45) and r.shape==(45,64) and not np.allclose(r,w.conj().T,atol=1e-3)
    assert np.linalg.norm(r@r.conj().T-np.eye(45))>.1
    frame=run.evaluate_ber(); assert set(frame.receiver)=={'historical_R'} and len(frame)==4
    assert set(frame[frame.modulation=='QPSK'].bits)=={180}
    repeated=start_cfo(**args); repeated.evaluate_ber()
    for key,values in run.counts.items():
        np.testing.assert_array_equal(values['errors'],repeated.counts[key]['errors'])
        assert np.all((values['eps']>=-.1)&(values['eps']<=.1))
    changed=start_cfo(**(args|{'model':{**MODEL,'cfo_range':(.2,.3)}}))
    assert not torch.equal(changed.generate_channels()[0].h,h.h)
    # Exact positive CFO diagonal independent of the DD negative-sign model.
    np.testing.assert_allclose(m.array(operator(m.tensor([0.])))[0],np.eye(64),atol=0)


@pytest.mark.artifacts
def test_cfo_noise_covariance_is_distinct_analysis(artifacts):
    root=Path(__file__).resolve().parents[1]
    run=start_cfo(mode='load_existing',evidence_mode='fresh',model=MODEL,display=DISPLAY,evaluation=EVAL,
                  workspace=root,artifacts=artifacts,bundle=None,save_run=False)
    noise=(m.draw((8192,64),('CFO-noise-covariance','real'),normal=True)/np.sqrt(2),
           m.draw((8192,64),('CFO-noise-covariance','imag'),normal=True)/np.sqrt(2))
    analyzed=m.mm(noise,m.transpose(run.r))
    # Rows are samples; conjugate-transpose covariance is transpose(R R^H).
    measured=m.mm(m.adj(analyzed),analyzed)
    expected=m.mm(run.r,m.adj(run.r))
    np.testing.assert_allclose(m.array(measured)/8192,m.array(expected).T,atol=.055,rtol=.08)
