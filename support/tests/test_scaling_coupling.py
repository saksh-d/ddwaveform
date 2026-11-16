"""Independent small reference checks; no campaign/training."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from waveforms.coupling import analyze, spectrum
from waveforms.core import configure
from waveforms.scaling_evidence import ScalingEvidence, prepare_scaling
from waveforms.notebook_workflow import NotebookWorkflow, validate_config
from waveforms.artifacts import EXPECTED_W, sha, provision, verify


@pytest.mark.parametrize('n',[64,128,256])
def test_dtft_matches_complex_coefficients_and_energy(n):
    gen=np.random.default_rng(42)
    x=gen.normal(size=(n,2))+1j*gen.normal(size=(n,2))
    freq,s=spectrum(x)
    # Independent finite-sum check at off-grid frequencies, preserving phase.
    indices=np.array([0,3,11,n*4+1])
    direct=np.exp(-2j*np.pi*np.outer(freq[indices],np.arange(n)))@x/np.sqrt(n)
    np.testing.assert_allclose(s[indices],direct,atol=3e-13,rtol=1e-12)
    np.testing.assert_allclose(s[::8],np.fft.fftshift(np.fft.fft(x,axis=0),axes=0)/np.sqrt(n),atol=1e-13)
    np.testing.assert_allclose(np.sum(np.abs(s)**2,axis=0)/8,np.sum(np.abs(x)**2,axis=0),atol=1e-11)


@pytest.mark.artifacts
@pytest.mark.parametrize('n,sizes',[(64,[23,21,20]),(128,[25,44,24,35]),(256,[63,62,79,52])])
def test_own_basis_group_bridge_and_paired_evidence(n,sizes,scaling_artifacts,artifacts,tmp_path):
    evidence=ScalingEvidence(scaling_artifacts,n)
    config=validate_config('saved_replay',dict(n=n,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75)),dict(basis_columns=(0,),snr_max_db=25.))
    run=NotebookWorkflow(config,artifacts,tmp_path,False,selected_basis=evidence.selected,scaling=evidence)
    channels,aa=run.generate_channels(); w=run.bases['Learned']; d=run.coupling_data()
    assert d['unitarity_max']<1e-13 and d['bridge_max']<1e-13
    assert list(np.bincount(d['labels']))==sizes
    if n==64: assert sha(evidence.root/evidence.identity['file'])==EXPECTED_W
    for name in run.ws:
        np.testing.assert_allclose(aa[name][0].numpy(),evidence.records[name]['example'],atol=1e-12)
        for key in ('example_tau','example_nu','example_gains'):
            np.testing.assert_array_equal(evidence.records[name][key],evidence.records['Learned'][key])
    j=0
    np.testing.assert_allclose(w@d['A'][:,j],channels.h[0].numpy()@w[:,j],atol=1e-13)
    # Complex group contributions reconstruct total; their powers omit cross terms.
    pieces=np.stack([d['C'][:,d['labels']==g]@d['A'][d['labels']==g,j] for g in range(len(sizes))])
    np.testing.assert_allclose(pieces.sum(0),d['H_f']@d['C'][:,j],atol=1e-13)
    assert np.max(np.abs(np.abs(pieces.sum(0))**2-np.sum(np.abs(pieces)**2,axis=0)))>1e-8
    assert not np.allclose(d['Q_test'],d['mean_channel_power'],atol=1e-7)
    np.testing.assert_allclose(d['Q_test'].sum(0),1.,atol=1e-13)
    np.testing.assert_allclose(d['Q_test'].sum(1),1.,atol=1e-13)
    np.testing.assert_allclose(d['mean_channel_power'].sum(),n,atol=1e-11)
    assert np.all(evidence.selected.fixed_q_mask.sum(-1).numpy()==3*n//8)
    frame=run.evaluate_ber()
    assert set(frame.bits)=={4882432} and set(frame.channels)=={4768}
    assert frame.groupby(['modulation','snr_db']).bits.nunique().max()==1
    # Read archived per-channel counts, do not rerun source evaluator.
    with np.load(evidence.root/f'communication/N{n}_nominal/per_channel.npz') as data:
        errors=data['errors']
        for _,row in frame.iterrows():
            si=[0,5,10,12,15,18,20,25].index(int(row.snr_db))
            wi=['Learned','OTFS','AFDM'].index(row.waveform)
            ri=0 if row.receiver=='fixed' else 2
            mi=['QPSK','16QAM'].index(row.modulation)
            assert int(errors[si,wi,ri,0,mi].sum())==row.errors


@pytest.mark.artifacts
def test_scaling_bundle_provision_and_missing_size_rejection(scaling_artifacts,tmp_path):
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'support/configs/scaling_artifacts.json').read_text())
    bundle=scaling_artifacts.parent/'advisor-scaling-artifacts.tar.gz'
    if not bundle.is_file(): pytest.skip('Bundle provisioning check requires explicit local bundle')
    dest=tmp_path/'local/scaling'
    assert provision(bundle,dest,manifest)['files']==64
    assert verify(dest,manifest)['files']==64
    with pytest.raises(ValueError,match='Available sizes'): ScalingEvidence(dest,512)
    with pytest.raises(FileNotFoundError,match='No N64 substitution'):
        prepare_scaling(root,root/'local/intentionally-absent-scaling',None)


@pytest.mark.artifacts
@pytest.mark.parametrize('n',[128,256])
def test_fresh_ber_shape_support_and_budget_propagation(n,scaling_artifacts,artifacts,tmp_path):
    evidence=ScalingEvidence(scaling_artifacts,n)
    model=dict(n=n,paths=2,delay_range=(.1,.2),doppler_range=(-.1,.2))
    fresh=dict(seed=54,matrix_channels=2,matrix_index=1,ber_channels=2,blocks_per_channel=1,snrs_db=(0.,12.))
    config=validate_config('fresh',model,dict(basis_columns=(n-1,),snr_max_db=25.),fresh)
    run=NotebookWorkflow(config,artifacts,tmp_path,False,selected_basis=evidence.selected,scaling=evidence)
    h,aa=run.generate_channels(); table=run.evaluate_ber()
    assert h.h.shape==(2,n,n) and run.matrix_index==1
    assert table.shape[0]==24 and set(table.snr_db)=={0.,12.}
    assert set(table[table.modulation=='QPSK'].bits)=={2*n*2}
    assert set(table[table.modulation=='16QAM'].bits)=={2*n*4}
    assert ((h.tau>=.1)&(h.tau<=.2)).all()
    torch.testing.assert_close(aa['Learned'],run.ws['Learned'].mH@h.h@run.ws['Learned'])


def test_unsupported_or_missing_size_never_substitutes_n64(tmp_path):
    from waveforms.notebook_workflow import start_workflow
    display=dict(basis_columns=(0,),snr_max_db=25.)
    model=dict(n=512,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
    with pytest.raises(ValueError,match="own verified learned basis required"):
        validate_config('saved_replay',model,display)
    with pytest.raises(FileNotFoundError,match='no N64 substitution'):
        start_workflow(mode='load_existing',evidence_mode='saved_replay',model={**model,'n':128},
                       display=display,fresh=None,workspace=tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_grouped_view_preserves_absolute_energy_and_separate_amplitude_normalization():
    from waveforms.coupling import grouped_spectral_data, plot_grouped_spectra
    import matplotlib.pyplot as plt
    n=16; k=np.arange(n)
    w=np.exp(2j*np.pi*np.outer(k,k)/n)/np.sqrt(n)
    labels=np.array([2,0,1,0,2,1,0,2,1,0,2,1,0,2,1,0])
    order=np.argsort(labels,kind='stable')
    original=w.copy(); data=grouped_spectral_data(w,labels,order)
    np.testing.assert_allclose(data['absolute_energy'],np.eye(n),atol=1e-13)
    np.testing.assert_allclose(data['absolute_energy'].sum(0),1,atol=1e-13)
    assert data['dense_frequency'][0]==0 and data['dense_frequency'][-1]<1
    np.testing.assert_allclose(data['peak_normalized_magnitude_db'].max(0),0,atol=1e-13)
    # Different column gains must change absolute energy, not normalized shapes.
    gains=np.linspace(1,2,n)
    scaled=grouped_spectral_data(w*gains,labels,order)
    np.testing.assert_allclose(scaled['absolute_energy'],data['absolute_energy']*gains**2,atol=1e-13)
    np.testing.assert_allclose(scaled['peak_normalized_magnitude_db'],data['peak_normalized_magnitude_db'],atol=1e-11)
    j,index=3,101
    f=data['dense_frequency'][index]
    direct=np.abs(np.exp(-2j*np.pi*f*k)@w[:,j]/np.sqrt(n))
    expected=max(-60,20*np.log10(direct/data['individual_peak_magnitude'][j]))
    np.testing.assert_allclose(data['peak_normalized_magnitude_db'][index,j],expected,atol=1e-12)
    np.testing.assert_array_equal(w,original)
    fig=plot_grouped_spectra(data)
    assert len(fig.axes)==6
    for row,g in enumerate(np.unique(labels)):
        members=order[labels[order]==g]
        left,right=fig.axes[2*row:2*row+2]
        assert len(left.lines)==len(members)+1 and len(right.lines)==len(members)
        np.testing.assert_allclose(left.lines[-1].get_ydata(),data['absolute_energy'][:,members].mean(1))
        assert left.lines[-1].get_color()=='black'
    plt.close(fig)


@pytest.mark.artifacts
@pytest.mark.parametrize('n,counts',[(64,[23,21,20]),(128,[25,44,24,35]),(256,[63,62,79,52])])
def test_grouped_view_uses_all_actual_size_members(n,counts,scaling_artifacts):
    from waveforms.coupling import grouped_spectral_data
    evidence=ScalingEvidence(scaling_artifacts,n);record=evidence.records['Learned']
    data=grouped_spectral_data(record['w'],record['labels'],record['permutation'])
    assert np.bincount(data['labels']).tolist()==counts
    np.testing.assert_array_equal(data['permutation'],record['permutation'])
    np.testing.assert_allclose(data['absolute_energy'],record['spectral_power'],atol=1e-13)
    np.testing.assert_allclose(data['complex_spectrum'][::32],data['dft_coefficients'],atol=1e-13)
    np.testing.assert_allclose(data['absolute_energy'].sum(0),np.ones(n),atol=1e-13)
    assert data['peak_normalized_magnitude_db'].shape==(32*n,n)


def test_example_indices_validate_native_ids_and_freeze_display_order():
    from waveforms.coupling import example_indices
    labels=np.array([1,0,1,0,1,0]);order=np.argsort(labels,kind='stable')
    np.testing.assert_array_equal(example_indices(labels,order,(4,1,0)),[1,0,4])
    with pytest.raises(ValueError,match='two examples per coupling group'):
        example_indices(labels,order,(0,2,4))
    with pytest.raises(ValueError,match='outside selected N'):
        example_indices(labels,order,(6,))
    with pytest.raises(ValueError,match='unique'):
        example_indices(labels,order,(1,1))


@pytest.mark.artifacts
@pytest.mark.parametrize('n',[64,128,256])
def test_selected_examples_keep_group_native_rank_and_same_channel(n,scaling_artifacts,tmp_path):
    from waveforms.coupling import plot_basis_examples,plot_grouped_spectra
    import matplotlib.pyplot as plt
    evidence=ScalingEvidence(scaling_artifacts,n);record=evidence.records['Learned']
    labels,order=record['labels'],record['permutation']
    # Choose another member of every group; input tuple deliberately reverse-ordered.
    requested=tuple(int(np.flatnonzero(labels==g)[1]) for g in np.unique(labels))[::-1]
    display=dict(basis_columns=(0,),example_columns=requested,snr_max_db=25.)
    model=dict(n=n,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
    config=validate_config('saved_replay',model,display)
    run=NotebookWorkflow(config,scaling_artifacts,tmp_path,False,selected_basis=evidence.selected,scaling=evidence)
    selected=run.example_table();h,aa=run.generate_channels();data=run.coupling_data();spectral=run.grouped_spectra_data()
    expected=order[np.isin(order,requested)]
    np.testing.assert_array_equal(selected.native_basis_index,expected)
    np.testing.assert_array_equal(selected.group,labels[expected])
    np.testing.assert_array_equal(order[selected.permuted_rank],expected)
    baseline=NotebookWorkflow(validate_config('saved_replay',model,{**display,'example_columns':(0,)}),
                              scaling_artifacts,tmp_path,False,selected_basis=evidence.selected,scaling=evidence)
    np.testing.assert_array_equal(baseline.generate_channels()[0].h,h.h)
    np.testing.assert_array_equal(run.bases['Learned'],baseline.bases['Learned'])
    for name in aa:np.testing.assert_array_equal(aa[name],baseline.effective_channels[name])
    import pandas as pd
    pd.testing.assert_frame_equal(run.evaluate_ber(),baseline.evaluate_ber())
    fig=plot_basis_examples(data,spectral,run.bases['Learned'],requested)
    assert len(fig.axes)==3*len(expected)
    for row,j in enumerate(expected):
        ax=fig.axes[row*3]
        assert f'native j={j}' in ax.get_title() and f'group {labels[j]}' in ax.get_title()
        np.testing.assert_array_equal(ax.lines[0].get_ydata(),record['w'][:,j].real)
        bars=fig.axes[row*3+2].patches
        np.testing.assert_allclose([bar.get_height() for bar in bars],np.abs(data['A'][order,j])**2)
    plt.close(fig)
    # Labeled overview highlights show the exact same selected spectra.
    fig=plot_grouped_spectra(spectral,requested)
    for row,g in enumerate(np.unique(labels)):
        ax=fig.axes[2*row+1];j=next(int(j) for j in expected if labels[j]==g)
        highlighted=next(line for line in ax.lines if f'Native j={j} ' in line.get_label())
        np.testing.assert_array_equal(highlighted.get_ydata(),spectral['peak_normalized_magnitude_db'][:,j])
    plt.close(fig)
    # Invalid group density fails before run creation, rather than silently dropping members.
    three=tuple(int(x) for x in np.flatnonzero(labels==0)[:3])
    invalid=validate_config('saved_replay',model,{**display,'example_columns':three})
    with pytest.raises(ValueError,match='two examples per coupling group'):
        NotebookWorkflow(invalid,scaling_artifacts,tmp_path,True,selected_basis=evidence.selected,scaling=evidence)
    assert list(tmp_path.iterdir())==[]
