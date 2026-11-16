"""Extract the public runtime dataset from explicitly supplied verified audit snapshots.

Maintainer utility only; the notebook never runs this or searches source projects.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from waveforms.artifacts import sha,verify,write_json


def array_hash(a):
    a=np.ascontiguousarray(a)
    digest=hashlib.sha256(str(a.dtype).encode())
    digest.update(np.asarray(a.shape,dtype=np.int64).tobytes());digest.update(a.tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scaling',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'waveforms/data')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh runtime dataset destination')
    sm=json.loads((ROOT/'support/configs/scaling_artifacts.json').read_text())
    lm=json.loads((ROOT/'support/configs/artifacts.json').read_text())
    verify(args.scaling,sm);verify(args.legacy,lm)
    args.out.mkdir(parents=True)
    manifest=dict(format='waveforms-runtime-v1',files={},dd={},
        source_bundles={'scaling':sm['bundle_sha256'],'legacy':lm['bundle_sha256']},
        scope='Exact archived Learned/support/statistic/replay subset; no new fit or BER simulation',
        exclusions='Raw per-channel campaigns, optimizer histories, source snapshots and internal studies remain optional audit archives',
        dd_protocol=json.loads((args.scaling/'config.json').read_text()))
    identities=json.loads((args.scaling/'matrices/MANIFEST.json').read_text())
    for n in (64,128,256):
        identity=next(x for x in identities if x['N']==n)
        values={};origins={}
        for name in ('Learned','OTFS','AFDM'):
            original=f'structure/N{n}_nominal_{name}.npz'
            with np.load(args.scaling/original,allow_pickle=False) as a:
                keys=('w','q_nominal','q_test','ensemble_power','labels','permutation',
                      'example_tau','example_nu','example_gains') if name=='Learned' else ()
                for key in (*keys,'mask_nominal'):
                    target=key if key!='mask_nominal' else f'{name}_mask'
                    values[target]=a[key].copy()
                    origins[target]=dict(file=original,file_sha256=sha(args.scaling/original),
                        key=key,array_sha256=array_hash(values[target]))
            if name=='Learned':
                w=np.load(args.scaling/identity['file'],allow_pickle=False)
                np.testing.assert_array_equal(w,values['w'])
        filename=f'dd_N{n}.npz'
        np.savez_compressed(args.out/filename,**values)
        manifest['dd'][str(n)]=dict(N=n,K=3*n//8,file=filename,identity=identity,arrays=origins,
            example='Original structural-channel index0 paths/gains; effective A regenerated, never outcome-selected',
            statistics='q_test:2048 nominal unitary DD atoms; ensemble_power:512 nominal normalized multipath channels',
            selection='Historical verified export N64; one nominal validation-selected trial per higher N; no BER selection')
    original=args.scaling/'communication_summary.csv'
    frame=pd.read_csv(original)
    manifest['replay']=dict(file='dd_replay.csv',original_file='communication_summary.csv',
        original_sha256=sha(original),dtypes={k:str(v) for k,v in frame.dtypes.items()},
        selection='nominal; raw slicer; fixed_nominal/full receivers; original rows, counts, intervals and stopping flags',rows=0)
    with original.open(newline='') as src,(args.out/'dd_replay.csv').open('w',newline='') as dst:
        rows=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=['_source_row',*rows.fieldnames],lineterminator='\n');writer.writeheader()
        for i,row in enumerate(rows):
            if row['N'] in ('64','128','256') and row['scenario']=='nominal' and row['slicer']=='raw' and row['receiver'] in ('fixed_nominal','full'):
                writer.writerow({'_source_row':i,**row});manifest['replay']['rows']+=1
    assert manifest['replay']['rows']==288
    original=args.legacy/'residual_cfo/stage1_checkpoint.pt'
    shutil.copyfile(original,args.out/'cfo_stage1.pt')
    manifest['cfo']=dict(file='cfo_stage1.pt',original_file='residual_cfo/stage1_checkpoint.pt',
        original_sha256=sha(original),W_shape=[64,45],R_shape=[45,64],
        physical='positive residual CFO, A=RHW; distinct R; no DD unitary substitution',
        scope='Frozen genuine stage1 checkpoint; original acceptance/payload checks retained; no CFO training or saved BER reuse')
    for file in sorted(args.out.iterdir()):
        manifest['files'][file.name]=dict(bytes=file.stat().st_size,sha256=sha(file))
    write_json(args.out/'manifest.json',manifest)
    print(json.dumps({'files':{p.name:p.stat().st_size for p in sorted(args.out.iterdir())},
                      'total_bytes':sum(p.stat().st_size for p in args.out.iterdir())},indent=2))


if __name__=='__main__':main()
