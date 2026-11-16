"""Prove cleanup equivalence against locally retained pre-cleanup runs and source."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from waveforms.artifacts import sha


class NamingOnly(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='selected_frame': node.id='advisor_frame'
        return node

    def visit_alias(self,node):
        if node.name=='selected_frame':node.name='advisor_frame'
        return node

    def visit_FunctionDef(self,node):
        if node.name=='selected_frame':node.name='advisor_frame'
        node=self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_Module(self,node):
        return self._strip_docstring(self.generic_visit(node))

    @staticmethod
    def _strip_docstring(node):
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body=node.body[1:]
        return node

    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=node.value.replace('waveforms.entry','advisor.entry')
            node.value=node.value.replace('Selected basis differs','Advisor basis differs').replace('Saved selected A','Saved advisor A')
            if node.value=='research':node.value='advisor'  # Figure manifest audience label only.
            if node.value=='waveforms':node.value='advisor'  # Notebook root discovery directory only.
        return node


def normalized(source):
    return ast.dump(NamingOnly().visit(ast.parse(source)),include_attributes=False)


def run_record(notebook):
    nb=json.loads(Path(notebook).read_text())
    for output in nb['cells'][-1]['outputs']:
        for key in ('application/vnd.waveforms.run+json','application/json'):
            if key in output.get('data',{}):
                result=output['data'][key]
                return Path(result.get('optional_local_archive',result.get('archive')))
    raise ValueError('Missing completion record')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',default='local/presentation-cleanup/before/runs')
    parser.add_argument('--after',default='local/presentation-cleanup/after/runs')
    parser.add_argument('--base',default='aca235e627c13319287fd0c2c17ae931b7c7cc81')
    args=parser.parse_args();before=(ROOT/args.before).resolve();after=(ROOT/args.after).resolve()
    if not all(p.is_relative_to(ROOT/'local') for p in (before,after)):raise ValueError('Evidence must stay under local/')
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',args.base,'advisor'],cwd=ROOT,text=True).splitlines()
    modules=[];identical=[]
    for name in tracked:
        if not name.endswith('.py'):continue
        old=subprocess.check_output(['git','show',f'{args.base}:{name}'],cwd=ROOT,text=True)
        new=(ROOT/'waveforms'/Path(name).name).read_text()
        assert normalized(old)==normalized(new),f'Non-presentation source change: {name}'
        modules.append(name)
        if old==new:identical.append(name)
    old_nb=json.loads((ROOT/'local/presentation-cleanup/before/results.ipynb').read_text())
    new_nb=json.loads((ROOT/'Waveform_Exploration.ipynb').read_text())
    config=lambda nb:''.join(next(c for c in nb['cells'] if 'configuration' in c.get('metadata',{}).get('tags',[]))['source'])
    assert normalized(config(old_nb))==normalized(config(new_nb)),'Configuration expressions changed'
    rows=[]
    for case in ('replay64','train64','cfo'):
        a=run_record(before/(case+'.ipynb'));b=run_record(after/(case+'.ipynb'))
        arrays=0
        assert {p.name for p in a.glob('*.npz')}=={p.name for p in b.glob('*.npz')}
        for path in a.glob('*.npz'):
            with np.load(path,allow_pickle=False) as left,np.load(b/path.name,allow_pickle=False) as right:
                assert set(left.files)==set(right.files),path.name
                for key in left.files:
                    np.testing.assert_array_equal(left[key],right[key],err_msg=f'{case}/{path.name}:{key}')
                    assert left[key].dtype==right[key].dtype
                    arrays+=1
        for path in a.glob('*.csv'):
            left=pd.read_csv(path);right=pd.read_csv(b/path.name)
            # Timing is expected to vary with presentation/runtime overhead.
            pd.testing.assert_frame_equal(left.drop(columns='seconds',errors='ignore'),right.drop(columns='seconds',errors='ignore'))
        pngs=list(a.glob('*.png'));assert {p.name for p in pngs}=={p.name for p in b.glob('*.png')}
        for path in pngs: assert sha(path)==sha(b/path.name),f'Figure changed: {case}/{path.name}'
        old=json.loads((a/'run.json').read_text());new=json.loads((b/'run.json').read_text())
        for key in ('selected_step','steps_executed','validation_leakage','trace'):
            assert old['selected_learned'].get(key)==new['selected_learned'].get(key),(case,key)
        rows.append(dict(case=case,array_fields_exact=arrays,figure_count=len(pngs),all_png_bytes_exact=True,
                         csv_values_exact_except_seconds=True,training_selection_exact=True,before=str(a),after=str(b)))
    result=dict(source_modules_equivalent=len(modules),source_modules_byte_identical=len(identical),
                normalization='Only docstrings, selected_frame name, two selection error labels, package/root discovery and figure audience label',
                config_expressions_identical=True,results=rows)
    (after.parent/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
