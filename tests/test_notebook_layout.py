"""Presentation-only contracts; no fitting or numerical simulations needed."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from waveforms import presentation as view

ROOT = Path(__file__).resolve().parents[1]


def test_single_root_entry_and_compact_markdown():
    notebooks = list(ROOT.glob('*.ipynb'))
    assert [p.name for p in notebooks] == ['Waveform_Exploration.ipynb']
    assert not (ROOT/'notebooks/results.ipynb').exists()
    assert not (ROOT/'advisor').exists()
    notebook = json.loads(notebooks[0].read_text())
    markdown = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='markdown']
    assert max(map(len,markdown)) < 600
    assert sum('configuration' in c.get('metadata',{}).get('tags',[]) for c in notebook['cells']) == 1
    assert 'JSON(' not in '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert all(c.get('outputs',[])==[] and c.get('execution_count') is None for c in notebook['cells'] if c['cell_type']=='code')
    assert [s.splitlines()[0] for s in markdown[1:]] == [
        '## 1. Configure the experiment','## 2. Run overview','## 3. Physical and effective channel',
        '## 4. Waveform structure','## 5. Overlay comparisons','## 6. Error-rate evaluation','## 7. Run record']


def test_run_summary_is_compact_with_full_collapsed_metadata(monkeypatch):
    outputs=[];monkeypatch.setattr(view,'display',outputs.append)
    info={'selected_learned':{'selected_step':3,'steps_executed':5},'full_trace':['retained']*200}
    workflow=SimpleNamespace(config={'scenario':'dd_v1','basis_mode':'train_new','evidence_mode':'fresh',
        'model':dict(paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))},
        summary=lambda:info,n=64,k=24,environment={'device':'mps:0','real_dtype':'torch.float32'})
    view.show_run_summary(workflow)
    assert len(outputs)==2
    assert outputs[0].data.count('<tr')==6  # header plus five setting rows
    assert 'Validation-selected step 3 / 5' in outputs[0].data
    assert 'retained' not in outputs[0].data
    assert outputs[1].data.startswith('<details>') and 'retained' in outputs[1].data
    assert '<script' not in ''.join(x.data for x in outputs)


def test_display_respects_active_plot_selection(monkeypatch):
    outputs=[];calls=[];monkeypatch.setattr(view,'display',outputs.append)
    workflow=SimpleNamespace(required_figures=lambda:{'basis_time'},show=calls.append)
    view.show_figure(workflow,'basis_time','4.1 Time','Complex coefficients.')
    view.show_figure(workflow,'grouped_spectra','4.4 Groups','All members.')
    assert calls==['basis_time'] and len(outputs)==1
    assert outputs[0].data.startswith('### 4.1 Time')
    with pytest.raises(ValueError): view.show_figure(workflow,'typo','Heading','Caption')


def test_ber_summary_preserves_full_counts_and_flags(monkeypatch):
    outputs=[];monkeypatch.setattr(view,'display',outputs.append)
    frame=pd.DataFrame(dict(modulation=['QPSK','QPSK'],channels=[2,4],bits=[512,1024],
        errors=[0,4],ber=[0.,4/1024],stop=['time_budget','channel_budget'],
        zero_cluster_upper95=[.77,float('nan')],minimum_channels_met=[False,True]))
    original=frame.copy(deep=True)
    assert view.show_ber_summary(SimpleNamespace(evaluate_ber=lambda:frame)) is frame
    pd.testing.assert_frame_equal(frame,original)
    assert 'Zero-error points' in outputs[0].data and '2–4' in outputs[0].data
    assert outputs[1].data.startswith('<details>')
    assert all(key in outputs[1].data for key in ('errors','zero_cluster_upper95','minimum_channels_met'))


def test_completion_record_renders_html_without_expanded_json():
    result={'essential_inline_figures':13,'optional_local_archive':'local/example'}
    record=view._RunRecord(result,'local/example')._repr_mimebundle_()
    assert record[view.RUN_MIME] is result
    assert 'text/html' in record and 'application/json' not in record
