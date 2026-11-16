"""Notebook reading flow and compact summaries; no simulation or fitting logic."""
from html import escape
import json
from pathlib import Path
import pandas as pd
from IPython.display import HTML, Markdown, display
from .notebook_workflow import ESSENTIALS
from .overlays import OVERLAY_VARIANTS

RUN_MIME = 'application/vnd.waveforms.run+json'


def details(title, content):
    """Standard HTML details, supported without JavaScript or custom styling."""
    display(HTML(f'<details><summary>{escape(title)}</summary>\n{content}\n</details>'))


def _table(rows):
    return pd.DataFrame(rows, columns=['Setting', 'Value']).to_html(index=False, escape=True)


def show_run_summary(workflow):
    info = workflow.summary()
    config = getattr(workflow, 'config', {})
    scenario = config.get('scenario', 'residual_cfo' if hasattr(workflow, 'streams') else 'dd_v1')
    model = config.get('model', getattr(workflow, 'model', {}))
    basis = info.get('selected_learned', {})
    evidence = config.get('evidence_mode', info.get('evidence_mode', 'saved_replay'))
    mode = config.get('basis_mode', info.get('basis_mode', 'load_existing'))
    env = workflow.environment
    if scenario == 'residual_cfo':
        dimensions = '64 samples · 45 streams; W[64,45], R[45,64]'
        channel = f"Residual CFO ε ∈ {tuple(model['cfo_range'])}"
        selection = 'Verified frozen W/R; direct R analysis'
    else:
        dimensions = f'N={workflow.n} · K={workflow.k} per receive row'
        channel = f"P={model['paths']} · delay {tuple(model['delay_range'])} · Doppler {tuple(model['doppler_range'])}"
        selection = (f"Validation-selected step {basis['selected_step']} / {basis['steps_executed']}"
                     if 'selected_step' in basis else 'Verified size-specific archived export')
    arithmetic = env['device'] + ' · ' + env.get('real_dtype', '')
    display(HTML(_table([
        ('Run', f'{scenario} · {mode} · {evidence}'),
        ('Dimensions', dimensions), ('Channel law', channel),
        ('Learned selection', selection), ('Arithmetic', arithmetic),
    ])))
    details('Configuration and provenance', '<pre>' + escape(json.dumps(info, indent=2, default=str)) + '</pre>')


def show_channel_details(workflow):
    matrices = workflow.matrix_summary().to_html(index=False, escape=True)
    paths = workflow.path_table().to_html(index=False, escape=True)
    details('Matrix dimensions and displayed channel draw', matrices + paths)


def _active_figures(workflow):
    if hasattr(workflow, 'required_figures'):
        return workflow.required_figures()
    return {'channel_matrices_native', 'basis_time', 'basis_frequency'} | {
        f'ber_{mod}_raw' for mod in workflow.e['modulations']}


def show_figure(workflow, name, heading, caption):
    if name not in set(ESSENTIALS) | set(OVERLAY_VARIANTS):
        raise ValueError('Unknown research figure')
    if name not in _active_figures(workflow):
        return
    display(Markdown(f'### {heading}\n\n{caption}'))
    workflow.show(name)


def show_coupling(workflow):
    if not hasattr(workflow, 'coupling_data'):
        return
    data = workflow.coupling_data()
    groups = workflow.group_table()
    counts = groups.groupby('group', sort=True).size().rename('Bases').reset_index()
    display(Markdown('### 4.3 Coupling structure\n\nGroups come from the selected basis’s coupling statistics; their count is derived. '
                     'Sample power, multipath mean power and atomic Q remain distinct.'))
    display(HTML(counts.to_html(index=False)))
    details('Native basis IDs, group membership and numerical checks',
            groups.to_html(index=False) + '<p>' + escape(
                f"max |WᴴW−I| = {data['unitarity_max']:.3g}; max complex bridge error = {data['bridge_max']:.3g}") + '</p>')
    workflow.show('coupling_groups')


def show_overlay_key(workflow):
    if not hasattr(workflow, 'overlay_table') or not any(
            name in _active_figures(workflow) for name in OVERLAY_VARIANTS):
        return
    content = '<p>Native ID → group → frozen coupling rank. Every selected curve remains in its overlay.</p>'
    for kind, title in [('overlay8', 'Eight bases'), ('half', 'N/2 bases')]:
        content += f'<h4>{title}</h4>' + workflow.overlay_table(kind).to_html(index=False)
    details('Complete overlay index keys', content)


def show_examples(workflow):
    if not hasattr(workflow, 'example_table'):
        return
    display(Markdown('### 4.5 Individual basis examples\n\nEach row links one native basis to its time samples, '
                     'normalized spectrum and exact coupling column in the displayed H.'))
    display(HTML(workflow.example_table().to_html(index=False)))
    workflow.show('basis_examples')


def show_ber_summary(workflow):
    frame = workflow.evaluate_ber()
    rows = []
    for mod, values in frame.groupby('modulation', sort=False):
        def span(column):
            low, high = int(values[column].min()), int(values[column].max())
            return f'{low:,}' if low == high else f'{low:,}–{high:,}'
        stops = ', '.join(sorted(str(x) for x in values['stop'].unique())) if 'stop' in values else 'saved evidence'
        rows.append({'Modulation': mod, 'Channels / point': span('channels'),
                     'Bits / arm / point': span('bits'), 'Zero-error points': int((values.errors == 0).sum()),
                     'Stopping': stops})
    display(HTML(pd.DataFrame(rows).to_html(index=False)))
    details('Per-point counts, BER intervals, zero-error bounds and stopping flags',
            frame.to_html(index=False, float_format=lambda x: f'{x:.5g}'))
    return frame


class _RunRecord:
    """HTML for readers; a small machine-readable record for local audit tools."""
    def __init__(self, result, archive):
        self.result, self.archive = result, archive

    def _repr_mimebundle_(self, include=None, exclude=None):
        count = self.result.get('essential_inline_figures', self.result.get('inline_figures'))
        rows = [('Inline figures', str(count)), ('Local archive', self.archive or 'Not saved')]
        return {'text/html': _table(rows), 'text/plain': f'Completed: {count} inline figures. Archive: {self.archive}',
                RUN_MIME: self.result}


def show_run_record(workflow, result):
    archive = str(workflow.run_dir) if workflow.run_dir else None
    if archive:
        root = Path.cwd().resolve()
        try:
            archive = str(Path(archive).relative_to(root))
        except ValueError:
            pass
    if workflow.run_dir:
        from .artifacts import sha, write_json
        path = workflow.run_dir / 'run.json'
        metadata = json.loads(path.read_text())
        metadata['source_sha256']['presentation.py'] = sha(__file__)
        notebook = Path(__file__).resolve().parents[1] / 'Waveform_Exploration.ipynb'
        metadata['notebook_sha256'] = sha(notebook)
        write_json(path, metadata)
    display(_RunRecord(result, archive))
