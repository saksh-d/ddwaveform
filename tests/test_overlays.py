"""Review selections/layouts stay faithful to frozen complex basis columns."""
import numpy as np
import pytest
from waveforms.coupling import grouped_spectral_data
from waveforms.overlays import validate_review,overlay_values,plot_overlay,OVERLAY_VARIANTS


def review(n):
    return dict(variants=(*OVERLAY_VARIANTS,'grouped_spectra'),
                overlay8_columns=tuple(range(0,n,n//8)),half_columns=tuple(range(0,n,2)))


@pytest.mark.parametrize('n',[64,128,256])
def test_overlay_selection_sizes_and_invalid_controls(n):
    cfg=review(n);valid=validate_review(cfg,n)
    assert len(valid['overlay8_columns'])==8 and len(valid['half_columns'])==n//2
    with pytest.raises(ValueError,match='exactly'):
        validate_review({**cfg,'half_columns':tuple(range(n//2-1))},n)
    with pytest.raises(ValueError,match='exactly'):
        validate_review({**cfg,'overlay8_columns':(0,)*8},n)
    with pytest.raises(ValueError,match='integers'):
        validate_review({**cfg,'overlay8_columns':tuple(range(7))+(n,)},n)
    with pytest.raises(ValueError,match='PLOT_VARIANTS'):
        validate_review({**cfg,'variants':('made_up',)},n)


@pytest.mark.parametrize('representation',['complex','energy_magnitude'])
def test_overlays_plot_all_selected_original_curves_with_complete_key(representation):
    import matplotlib.pyplot as plt
    n=16;x=np.arange(n)
    w=np.exp(2j*np.pi*np.outer(x,x)/n)/np.sqrt(n)
    labels=np.array([2,0,1,0,2,1,0,2,1,0,2,1,0,2,1,0]);order=np.argsort(labels,kind='stable')
    spectral=grouped_spectral_data(w,labels,order)
    columns=(0,2,4,6,8,10,12,14)
    values=overlay_values(w,spectral,columns);ids=order[np.isin(order,columns)]
    np.testing.assert_array_equal(values['indices'],ids)
    np.testing.assert_array_equal(values['time'],w[:,ids])
    np.testing.assert_allclose(values['absolute_energy'],np.eye(n)[:,ids],atol=1e-13)
    fig=plot_overlay(w,spectral,columns,representation)
    panels=fig.axes[:-1]
    assert len(panels)==(4 if representation=='complex' else 2)
    assert all(len(ax.lines)==8 for ax in panels)
    legend=fig.axes[-1].get_legend();text=[t.get_text() for t in legend.get_texts()]
    assert len(text)==8 and all(f'j={j} ' in ' '.join(text) for j in ids)
    for i,j in enumerate(ids):
        if representation=='complex':
            np.testing.assert_array_equal(panels[0].lines[i].get_ydata(),w[:,j].real)
            np.testing.assert_array_equal(panels[1].lines[i].get_ydata(),w[:,j].imag)
            np.testing.assert_array_equal(panels[2].lines[i].get_ydata(),spectral['complex_spectrum'][:,j].real)
            np.testing.assert_array_equal(panels[3].lines[i].get_ydata(),spectral['complex_spectrum'][:,j].imag)
        else:
            np.testing.assert_array_equal(panels[0].lines[i].get_ydata(),spectral['absolute_energy'][:,j])
            np.testing.assert_array_equal(panels[1].lines[i].get_ydata(),spectral['peak_normalized_magnitude_db'][:,j])
    plt.close(fig)


@pytest.mark.artifacts
@pytest.mark.parametrize('n',[64,128,256])
def test_actual_half_overlay_count_mapping_and_config_selection(n,scaling_artifacts,tmp_path):
    from waveforms.scaling_evidence import ScalingEvidence
    from waveforms.notebook_workflow import NotebookWorkflow,validate_config
    import matplotlib.pyplot as plt
    e=ScalingEvidence(scaling_artifacts,n);cfg=review(n)
    model=dict(n=n,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
    display=dict(basis_columns=(0,),example_columns=(0,),snr_max_db=25.)
    run=NotebookWorkflow(validate_config('saved_replay',model,display,review=cfg),scaling_artifacts,tmp_path,False,
                         selected_basis=e.selected,scaling=e)
    assert len(run.required_figures())==13
    table=run.overlay_table('half');assert len(table)==n//2
    labels=e.records['Learned']['labels'];order=e.records['Learned']['permutation']
    np.testing.assert_array_equal(order[table.permuted_rank],table.native_basis_index)
    np.testing.assert_array_equal(labels[table.native_basis_index],table.group)
    data=run.grouped_spectra_data()
    fig=plot_overlay(run.bases['Learned'],data,cfg['half_columns'],'complex')
    assert all(len(ax.lines)==n//2 for ax in fig.axes[:-1])
    assert len(fig.axes[-1].get_legend().get_texts())==n//2
    plt.close(fig)
    other={**cfg,'variants':('overlay8_complex',),'overlay8_columns':tuple(range(1,n,n//8)),
           'half_columns':tuple(range(1,n,2))}
    changed=NotebookWorkflow(validate_config('saved_replay',model,display,review=other),scaling_artifacts,tmp_path,False,
                            selected_basis=e.selected,scaling=e)
    assert len(changed.required_figures())==9 and set(changed.overlay_table('overlay8').native_basis_index)==set(other['overlay8_columns'])
    np.testing.assert_array_equal(run.generate_channels()[0].h,changed.generate_channels()[0].h)
    for name in run.ws:np.testing.assert_array_equal(run.ws[name],changed.ws[name])
    import pandas as pd
    pd.testing.assert_frame_equal(run.evaluate_ber(),changed.evaluate_ber())
