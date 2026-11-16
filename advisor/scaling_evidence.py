"""Qualified nominal size-specific frozen evidence; no training/source imports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .artifacts import verify, provision, sha
from .basis_source import SelectedBasis
from .delay_doppler import classical

SIZES = {64:24, 128:48, 256:96}
GRIDS = {64:(8,8), 128:(8,16), 256:(16,16)}


def prepare_scaling(workspace, root, bundle):
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    manifest=json.loads((workspace/'configs/scaling_artifacts.json').read_text())
    if not root.exists():
        if not root.is_relative_to(workspace/'local'):
            raise ValueError('Provision scaling evidence under this checkout local/')
        if bundle is None or not Path(bundle).is_file():
            raise FileNotFoundError('Qualified scaling bundle missing; set SCALING_BUNDLE. No N64 substitution.')
        provision(bundle,root,manifest)
    verify(root,manifest)
    return root


class ScalingEvidence:
    def __init__(self,root,n):
        if n not in SIZES: raise ValueError('Available sizes: N64/K24, N128/K48, N256/K96')
        self.root,self.n,self.k=Path(root),n,SIZES[n]
        self.records={}
        for name in ('Learned','OTFS','AFDM'):
            path=self.root/f'structure/N{n}_nominal_{name}.npz'
            meta=json.loads(path.with_suffix('.json').read_text())
            if sha(path)!=meta['npz_sha256']: raise ValueError('Altered size-specific structure')
            with np.load(path,allow_pickle=False) as a: self.records[name]={k:a[k].copy() for k in a.files}
        self.metadata=json.loads((self.root/'config.json').read_text())
        matrices=json.loads((self.root/'matrices/MANIFEST.json').read_text())
        self.identity=next(x for x in matrices if x['N']==n)
        path=self.root/self.identity['file']
        if sha(path)!=self.identity['file_sha256']: raise ValueError('Learned matrix checksum differs')
        w=np.load(path,allow_pickle=False)
        if not np.array_equal(w,self.records['Learned']['w']): raise ValueError('Learned orientation/content mismatch')
        baseline=classical(n,*GRIDS[n])
        for name in ('OTFS','AFDM'):
            np.testing.assert_allclose(self.records[name]['w'],baseline[name].numpy(),atol=1e-13,rtol=0)
        for name,data in self.records.items():
            labels,order=data['labels'],data['permutation']
            assert np.array_equal(np.sort(order),np.arange(n))
            if name=='Learned':
                assert np.array_equal(order,np.argsort(labels,kind='stable'))
            else:
                assert np.array_equal(order,np.argsort(labels,kind='stable')) or np.array_equal(order,np.arange(n))
            assert data['mask_nominal'].dtype==np.bool_ and np.all(data['mask_nominal'].sum(-1)==self.k)
            for key in ('q_nominal','q_test','ensemble_power'):
                assert data[key].dtype==np.float64 and np.isfinite(data[key]).all() and (data[key]>=0).all()
        self.selected=SelectedBasis(torch.from_numpy(w.copy()),torch.from_numpy(self.records['Learned']['mask_nominal']),
            {'source':self.identity['file'],'file_sha256':self.identity['file_sha256'],
             'provenance':self.identity['provenance'],'N':n,'K':self.k,'basis_mode':'load_existing',
             'training_status':'Not run here; completed archived training/export',
             'selection':'Historical export at N64; validation-selected tail, one trial each at N128/N256; no BER selection'}).validate()

    def matches(self,basis):
        return torch.equal(basis.matrix,self.selected.matrix) and torch.equal(basis.fixed_q_mask,self.selected.fixed_q_mask)

    def ber(self):
        a=pd.read_csv(self.root/'communication_summary.csv')
        a=a[(a.N==self.n)&(a.scenario=='nominal')&(a.slicer=='raw')&a.receiver.isin(['fixed_nominal','full'])].copy()
        a['receiver']=a.receiver.replace({'fixed_nominal':'fixed'})
        if len(a)!=96 or not set(a.waveform)=={'Learned','OTFS','AFDM'}: raise ValueError('Incomplete saved BER curves')
        # Preserve budget-censoring and precision flags, not only plotted means.
        return a
