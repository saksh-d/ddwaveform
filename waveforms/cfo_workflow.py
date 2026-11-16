"""Frozen rectangular residual-CFO W/R: fresh direct-analysis BER, no DD reuse.

The archived staged CFO trainer and its pilot/multipath extensions are excluded.
No invented unitary, sparse-LMMSE or OTFS/AFDM equivalence is claimed here.
"""
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from . import mps_numerics as m
from .artifacts import provision,verify,write_json,sha
from .configurable import integer,finite,exact_keys,local_path,DEFAULT_EVALUATION
from .residual_cfo import load_qualified
from .figures import style


def operator(eps,n=64):
    samples=torch.arange(n,device=m.DEVICE,dtype=m.DTYPE)
    p=m.phase(2*math.pi*eps[...,None]*samples/n)
    return torch.diag_embed(p[0]),torch.diag_embed(p[1])


class CFOWorkflow:
    def __init__(self,model,display,evaluation,w,r,checks,workspace,output_root,save_run):
        self.n,self.streams=64,45; self.model=model; self.display=display; self.e=evaluation
        self.environment=m.environment(); self.w=m.pair(w.numpy()); self.r=m.pair(r.numpy())
        self.ws={'Learned':w}; self.selected=checks; self.channels=None; self.ber_table=None
        self.images,self.counts={},{}; self.matrix_index=evaluation['matrix_index']; self.run_dir=None
        if save_run:
            self.run_dir=output_root/('cfo_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]); self.run_dir.mkdir(parents=True)
        self.provenance=dict(scenario='residual_cfo',basis_mode='load_existing',evidence_mode='fresh',model=model,evaluation=evaluation,display=display,
            environment=self.environment,selected_learned=checks,BER_status='Not yet evaluated',
            physical='H=diag(exp(+j2pi eps n/64)); W[64,45], R[45,64], A=RHW; R is not W^H',
            receiver='Historical raw direct R analysis (no added LMMSE). Physical white noise gives sigma² R R^H.',
            SNR='New evaluation Es/N0, unit energy data symbols; historical Eb/N0 tables are not reused',
            limitations='Frozen stage1 W/R only; no CFO training, pilots, compensation estimator, DD multipath, support budget, or classical comparator qualification')

    def summary(self): return self.provenance

    def generate_channels(self):
        if self.channels is None:
            from types import SimpleNamespace
            lo,hi=self.model['cfo_range']; eps=lo+(hi-lo)*m.draw((self.e['matrix_channels'],),('configurable-cfo-v1','test',self.e['seed'],'matrix'))
            h=operator(eps); a=m.effective(self.w,h,self.r)
            self.channels=SimpleNamespace(h=torch.from_numpy(m.array(h)),raw=torch.from_numpy(m.array(h)),eps=eps.cpu().numpy())
            self.a=torch.from_numpy(m.array(a))
        return self.channels,{'Learned':self.a}

    def matrix_summary(self):
        self.generate_channels()
        return pd.DataFrame([dict(matrix=name,shape=value.shape,dtype=str(value.dtype)) for name,value in [('H',self.channels.h),('W',self.ws['Learned']),('R',torch.from_numpy(m.array(self.r))),('A=RHW',self.a)]])

    def path_table(self):
        self.generate_channels()
        return pd.DataFrame({'residual CFO (DFT-bin units)':self.channels.eps})

    def show_basis_analysis(self):
        self.show('basis_time'); self.show('basis_frequency')
        print('CFO shows distinct W/R columns. DD groups, unitary bridge and N/2 DD overlays are inapplicable.')

    def show_ber_figures(self):
        from IPython.display import display
        frame=self.evaluate_ber(); print(self.provenance['BER_status']); display(frame)
        for mod in self.e['modulations']: self.show(f'ber_{mod}_raw')

    def evaluate_ber(self):
        if self.ber_table is not None: return self.ber_table
        rows=[]; e=self.e
        for mod in e['modulations']:
            for snr in e['snrs_db']:
                start=time.perf_counter(); errors=[]; eps_values=[]; count=0; stop='channel_budget'
                variance=10**(-snr/10); bps=2 if mod=='QPSK' else 4
                while count<e['max_channels']:
                    take=min(e['batch_channels'],e['max_channels']-count); batch=len(errors)
                    ns=('configurable-cfo-v1','test',e['seed'],'ber',batch)
                    lo,hi=self.model['cfo_range']; eps=lo+(hi-lo)*m.draw((take,),(*ns,'eps'))
                    bits=m.draw((take,e['blocks_per_channel'],45,bps),(*ns,'bits',mod,snr),bits=True)
                    s=m.symbols(bits,mod); y=m.mm(m.mm(s,m.transpose(self.w)),m.transpose(operator(eps)))
                    shape=(take,e['blocks_per_channel'],64)
                    y=(y[0]+math.sqrt(variance/2)*m.draw(shape,(*ns,'noise-real',mod,snr),normal=True),
                       y[1]+math.sqrt(variance/2)*m.draw(shape,(*ns,'noise-imag',mod,snr),normal=True))
                    estimate=m.mm(y,m.transpose(self.r))
                    errors.append((m.decode(estimate,mod)!=bits).sum((1,2,3)).cpu().numpy()); eps_values.append(eps.cpu().numpy()); count+=take
                    if count>=e['min_channels'] and e['target_errors'] and np.concatenate(errors).sum()>=e['target_errors']: stop='error_target'; break
                    if time.perf_counter()-start>=e['max_seconds_per_point']: stop='time_budget'; break
                values=np.concatenate(errors); total=int(values.sum()); bpc=e['blocks_per_channel']*45*bps; bits=count*bpc; mean=total/bits
                se=(values/bpc).std(ddof=1)/math.sqrt(count) if count>1 else np.nan
                rows.append(dict(modulation=mod,snr_db=snr,waveform='Learned',receiver='historical_R',channels=count,bits=bits,errors=total,ber=mean,
                    ci_low=max(0,mean-1.96*se) if total else np.nan,ci_high=min(1,mean+1.96*se) if total else np.nan,
                    zero_cluster_upper95=1-.05**(1/count) if not total else np.nan,detection_limit=1/bits,stop=stop,minimum_channels_met=count>=e['min_channels'],seconds=time.perf_counter()-start))
                self.counts[f'{mod}_{snr:g}dB']=dict(errors=values,eps=np.concatenate(eps_values),bits_per_channel=np.array(bpc))
        self.ber_table=pd.DataFrame(rows); self.provenance['BER_status']='NEW MPS FP32 CFO direct-analysis BER; descriptive channel-SE intervals, approximate under stopping; zero errors unresolved'
        return self.ber_table

    def show(self,name):
        from IPython.display import Image,display
        import matplotlib.pyplot as plt
        style()
        if name=='channel_matrices_native':
            self.generate_channels(); i=self.matrix_index
            items=[('H[64,64] · positive CFO',self.channels.h[i].numpy()),('W[64,45] · synthesis',m.array(self.w)),('R[45,64] · distinct analysis',m.array(self.r)),('A[45,45] = R H W',self.a[i].numpy())]
            fig,axes=plt.subplots(1,4,figsize=(21,6),layout='constrained')
            for ax,(title,value) in zip(axes,items):
                im=ax.imshow(10*np.log10(np.maximum(np.abs(value)**2,1e-8)),origin='lower',vmin=-60,vmax=0,aspect='auto'); ax.set_title(title); ax.set_xlabel('Input index')
            axes[0].set_ylabel('Output index'); fig.colorbar(im,ax=axes,shrink=.6,label='Entry power (dB re 1)')
        elif name in ('basis_time','basis_frequency'):
            columns=self.display['basis_columns']; fig,axes=plt.subplots(len(columns),1,squeeze=False,figsize=(17,3.6*len(columns)),layout='constrained')
            w=m.array(self.w); v=m.array(m.adj(self.r))
            for ax,j in zip(axes.flat,columns):
                for label,a in [('Transmit W[:,j]',w),('Receive V[:,j]=R^H[:,j]',v)]:
                    x=np.arange(64); value=a[:,j]
                    if name=='basis_frequency':
                        x=np.fft.fftshift(np.fft.fftfreq(512)); value=np.fft.fftshift(np.fft.fft(value,n=512))/8
                    line,=ax.plot(x,value.real,label=label+' Re'); ax.plot(x,value.imag,ls='--',color=line.get_color(),label=label+' Im')
                ax.set_title(f'Residual CFO · native stream j={j} · distinct W and R'); ax.set_xlabel('Frequency f/Fs' if name=='basis_frequency' else 'Time sample'); ax.set_ylabel('Complex coefficient'); ax.grid(alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside upper center',ncol=2)
        elif name.startswith('ber_'):
            mod='QPSK' if name=='ber_QPSK_raw' else '16QAM'; a=self.evaluate_ber(); a=a[a.modulation==mod]
            if self.display['snr_max_db'] is not None: a=a[a.snr_db<=self.display['snr_max_db']]
            fig,ax=plt.subplots(figsize=(12,7),layout='constrained'); y=np.where(a.errors>0,a.ber,a.detection_limit)
            ax.semilogy(a.snr_db,y,'o-',label='Learned W/R · direct raw R analysis')
            ax.fill_between(a.snr_db,a.ci_low,a.ci_high,alpha=.2); zero=a.errors==0; ax.scatter(a.snr_db[zero],y[zero],marker='v')
            ax.set_title(f'Fresh residual CFO · {mod} · 64 samples / 45 streams\nNo DD support receiver or historical BER reuse'); ax.set_xlabel('Es/N0 (dB)'); ax.set_ylabel('Uncoded BER'); ax.grid(alpha=.2); ax.legend()
            if zero.any(): fig.supxlabel('▽ zero errors at 1/bits; unresolved (channel bound in table)')
        else: raise ValueError('CFO supports matrices, complex W/R bases and direct-analysis BER only; DD grouping unsupported')
        buf=BytesIO(); fig.savefig(buf,format='png',dpi=300); plt.close(fig); self.images[name]=buf.getvalue()
        if self.run_dir: (self.run_dir/(name+'.png')).write_bytes(buf.getvalue())
        display(Image(data=buf.getvalue(),width=1100))

    def finish(self):
        expected={'channel_matrices_native','basis_time','basis_frequency'}|{f'ber_{mod}_raw' for mod in self.e['modulations']}
        if set(self.images)!=expected: raise ValueError('Run all CFO figure cells before finishing')
        if self.run_dir:
            from .experiments import save_npz
            save_npz(self.run_dir/'matrices.npz',H=self.channels.h,W=m.array(self.w),R=m.array(self.r),A=self.a,eps=self.channels.eps,
                     receive_noise_covariance_unit=m.array(m.mm(self.r,m.adj(self.r))))
            self.ber_table.to_csv(self.run_dir/'ber.csv',index=False)
            for name,values in self.counts.items(): save_npz(self.run_dir/(name+'.npz'),**values)
            self.provenance['source_sha256']={name:sha(Path(__file__).parent/name) for name in ('cfo_workflow.py','mps_numerics.py','residual_cfo.py')}
            self.provenance['files_sha256']={p.name:sha(p) for p in self.run_dir.iterdir() if p.is_file()}
            write_json(self.run_dir/'run.json',self.provenance)
        return dict(basis_mode='load_existing',scenario='residual_cfo',inline_figures=len(self.images),archive=str(self.run_dir),training='Unsupported; not run')


def start_cfo(*,mode,evidence_mode,model,display,evaluation,workspace,artifacts,bundle,save_run=True,output_root='local/notebook_runs',training=None,k=None,support_policy=None,discovery=None,checkpoint=None,review=None):
    if mode=='train_new': raise NotImplementedError('Residual CFO train_new unsupported: staged W/R trainer not reconstructed or qualified; no DD trainer substitution')
    if mode!='load_existing': raise ValueError('MODE must be load_existing or train_new')
    if evidence_mode!='fresh': raise ValueError('CFO requires fresh evaluation; archived BER tables are not qualified for configurable replay')
    if any(x is not None for x in (training,k,support_policy,discovery,checkpoint,review)):
        raise ValueError('CFO excludes DD training/K/support/group/checkpoint controls; set them None')
    exact_keys(model,{'n','streams','cfo_range'},'CFO MODEL')
    if type(model['n']) is not int or type(model['streams']) is not int or (model['n'],model['streams'])!=(64,45): raise ValueError('Qualified CFO dimensions only W[64,45], R[45,64]; no resizing')
    bounds=model['cfo_range']
    if not isinstance(bounds,(list,tuple)) or len(bounds)!=2: raise ValueError('cfo_range needs two limits')
    for x in bounds: finite(x,-.5,.5,'cfo_range')
    if bounds[0]>=bounds[1]: raise ValueError('cfo_range must be increasing within [-.5,.5]')
    exact_keys(evaluation,set(DEFAULT_EVALUATION)-{'diagnostic_atoms','diagnostic_channels'},'CFO EVALUATION')
    # Validate common evaluation settings through the DD schema; no DD simulation.
    from .configurable import validate
    nominal=dict(n=64,paths=3,delay_range=(0.,4.),doppler_range=(-.75,.75))
    config,_,out=validate(mode='load_existing',evidence_mode='fresh',scenario='dd_v1',model=nominal,display=display,evaluation={**evaluation,'diagnostic_atoms':2,'diagnostic_channels':2},training=None,k=24,support_policy='frozen',discovery=None,checkpoint=None,workspace=workspace,output_root=output_root,review=None)
    if any(j>=45 for j in config['display']['basis_columns']): raise ValueError('CFO basis column must be a stream in [0,44]')
    if type(save_run) is not bool: raise ValueError('SAVE_RUN must be boolean')
    m.environment(); artifacts=local_path(artifacts,workspace)
    manifest=json.loads((Path(workspace)/'configs/artifacts.json').read_text())
    if not artifacts.exists():
        if bundle is None or not Path(bundle).is_file(): raise FileNotFoundError('CFO requires local legacy bundle')
        provision(bundle,artifacts,manifest)
    verify(artifacts,manifest); w,r,checks=load_qualified(artifacts)
    return CFOWorkflow(model,config['display'],evaluation,w,r,checks,workspace,out,save_run)
