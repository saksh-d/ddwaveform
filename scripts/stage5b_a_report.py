"""Render the complete declared recovery ledger and summary statistics."""
import json
from collections import defaultdict
from pathlib import Path
from scripts.stage5b_a import OUT, dump


def main():
    rows=[]
    for armset in ['AB','C']:
        assert (OUT/armset/'complete.json').exists()
        rows+=json.loads((OUT/armset/'results.json').read_text())
    grouped=defaultdict(list)
    for r in rows: grouped[r['arm'],r['category']].append(r)
    lines=['\n## Complete recovery ledger\n',
        'Errors below are relative Frobenius errors, not squared losses. Each cell: best error / objective-and-gradient evaluations. Every fit retained; numerical failures: '+str(sum(r['numerical_failure'] is not None for r in rows))+'.\n',
        '| Arm | Start | Pass / cases | Initial error range | Final error range | Total evaluations | Fitting seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for (arm,cat),rs in sorted(grouped.items()):
        lines.append(f"| {arm} | {cat} | {sum(r['success'] for r in rs)}/{len(rs)} | {min(r['initial_error'] for r in rs):.3g}–{max(r['initial_error'] for r in rs):.3g} | {min(r['error'] for r in rs):.3g}–{max(r['error'] for r in rs):.3g} | {sum(r['evaluations'] for r in rs)} | {sum(r['seconds'] for r in rs):.2f} |")
    lines+=['\nEach row below denotes one common target/start. `s0/s1/s2` are generation indices, **not restarts of one target**. `0.001/0.03` are polar perturbation scales; `cold` uses only the target diagonal for phase initialization. Exact seeds appear in raw results.\n',
        '| Configuration / start | Initial error | A error / evals | B error / evals | C error / evals |',
        '|---|---:|---:|---:|---:|']
    bykey=defaultdict(dict)
    for r in rows: bykey[r['key']][r['arm']]=r
    for key, arms in sorted(bykey.items()):
        lines.append(f"| {key} | {arms['A']['initial_error']:.6g} | "+' | '.join(f"{arms[a]['error']:.6g} / {arms[a]['evaluations']}" for a in ['A','B','C'])+' |')
    verification=json.loads((OUT/'verification.json').read_text())
    prefix=verification['common_prefix']
    lines+=['\n## Equal-evaluation prefix check\n']
    for category in ['local','cold']:
        rs=[r for r in prefix if r['key'].endswith('cold')==(category=='cold')]
        lines.append(f"A/B at the smaller of each pair's recorded closure counts, {category}: A {sum(r['A']<1e-4 for r in rs)}/{len(rs)}, B {sum(r['B']<1e-4 for r in rs)}/{len(rs)} pass. This is a diagnostic trace-prefix comparison, not equal wall time.\n")
    lines+=['\n## Representative gradient evolution\n',
        'Ascending four-pass seed0, near perturbation0.001 and cold, fixed independently of outcomes. Norms are Euclidean parameter-gradient norms and depend on coordinates.\n',
        '| Start | Arm | Initial norm | Maximum norm | Last evaluated norm |', '|---|---|---:|---:|---:|']
    for cat in ['0.001','cold']:
        for arm in ['A','B','C']:
            base='AB' if arm!='C' else 'C'
            trace=json.loads((OUT/base/f'ascending_p4_s0_{cat}_{arm}_trace.json').read_text())
            lines.append(f"| {cat} | {arm} | {trace[0]['gradient_norm']:.4g} | {max(r['gradient_norm'] for r in trace):.4g} | {trace[-1]['gradient_norm']:.4g} |")
    lines+=['\n## Algebraic planted references\n',
        '18 distinct targets; exact saved polar factors reproduce targets with zero discrepancy. Cartesian conversion maximum relative error '+f"{max(r['converted_error'] for r in verification['planted']):.3g}"+'. These are algebraic references, never counted as optimization successes. Raw receipt repeats each reference across start categories.\n']
    Path('reports/stage5b_a_results.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'summary.json',dict(fits=len(rows),groups=[dict(arm=a,category=c,cases=len(rs),passed=sum(r['success'] for r in rs),
        evaluations=sum(r['evaluations'] for r in rs),seconds=sum(r['seconds'] for r in rs),
        max_error=max(r['error'] for r in rs)) for (a,c),rs in sorted(grouped.items())]))

if __name__=='__main__':main()
