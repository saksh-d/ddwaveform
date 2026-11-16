"""Frozen coupling groups linked to complex spectra and one physical channel."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

GROUP_COLORS = ('#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9')


def analyze(w,h,record):
    n=len(w); f=np.fft.fft(np.eye(n),axis=0)/np.sqrt(n)
    c=f@w; hf=f@h@f.conj().T; a=w.conj().T@h@w
    labels,order=record['labels'],record['permutation']
    if not np.array_equal(order,np.argsort(labels,kind='stable')):
        raise ValueError('Group permutation must be the frozen stable discovery ordering')
    return {'C':c,'H_f':hf,'A':a,'A_spectral':c.conj().T@hf@c,
            'labels':labels,'permutation':order,'Q_test':record['q_test'],
            'mean_channel_power':record['ensemble_power'],
            'unitarity_max':float(np.max(np.abs(w.conj().T@w-np.eye(n)))),
            'bridge_max':float(np.max(np.abs(c.conj().T@hf@c-a)))}


def spectrum(w,oversample=8):
    """Finite-block DTFT samples; original DFT knots equal FW. No window/peak scaling."""
    n=w.shape[0]; length=oversample*n
    return (np.fft.fftshift(np.fft.fftfreq(length)),
            np.fft.fftshift(np.fft.fft(w,n=length,axis=0),axes=0)/np.sqrt(n))


def grouped_spectral_data(w, labels, permutation, oversample=32):
    """Absolute unitary-DFT energy and separately peak-normalized DTFT magnitude."""
    n = len(w)
    labels, order = np.asarray(labels), np.asarray(permutation)
    if w.shape != (n,n) or not np.isfinite(w).all():
        raise ValueError('Expected finite synthesis W[N,N]')
    if labels.shape != (n,) or order.shape != (n,) or not np.array_equal(np.sort(order),np.arange(n)):
        raise ValueError('Group labels and permutation must map every native basis index')
    if not np.array_equal(order,np.argsort(labels,kind='stable')):
        raise ValueError('Use the same frozen group permutation as the coupling matrix')
    if type(oversample) is not int or oversample < 1:
        raise ValueError('oversample must be a positive integer')
    c = np.fft.fft(w,axis=0)/np.sqrt(n)
    dense = np.fft.fft(w,n=oversample*n,axis=0)/np.sqrt(n)
    peaks = np.abs(dense).max(axis=0)
    if (peaks <= 0).any():
        raise ValueError('Cannot peak-normalize a zero basis column')
    # 20 log10 amplitude; no peak normalization enters the absolute DFT energy.
    relative_db = 20*np.log10(np.maximum(np.abs(dense)/peaks,1e-3))
    return {'dft_frequency':np.arange(n)/n, 'dense_frequency':np.arange(oversample*n)/(oversample*n),
            'dft_coefficients':c, 'absolute_energy':np.abs(c)**2,
            'complex_spectrum':dense, 'peak_normalized_magnitude_db':relative_db,
            'individual_peak_magnitude':peaks, 'labels':labels.copy(), 'permutation':order.copy(),
            'oversample':np.array(oversample), 'display_floor_db':np.array(-60.)}


def example_indices(labels, permutation, columns):
    """Validate native IDs; display in frozen coupling order, at most two per group."""
    labels, order = np.asarray(labels), np.asarray(permutation)
    columns = tuple(columns)
    if not 1 <= len(columns) <= 8 or len(set(columns)) != len(columns):
        raise ValueError('Choose1–8 unique example_columns (native basis IDs)')
    if any(type(j) is not int or not 0 <= j < len(labels) for j in columns):
        raise ValueError('Example native basis index outside selected N')
    if not np.array_equal(np.sort(order),np.arange(len(labels))):
        raise ValueError('Examples require the frozen coupling permutation')
    if any(np.count_nonzero(labels[list(columns)]==g)>2 for g in np.unique(labels) if g>=0):
        raise ValueError('Choose at most two examples per coupling group to keep panels readable')
    return order[np.isin(order,columns)]


def plot_grouped_spectra(data, columns=()):
    """Every member, one row per frozen coupling group, same IDs/order/colors."""
    labels, order = data['labels'],data['permutation']
    groups = list(dict.fromkeys(labels[order].tolist()))
    fig, axes = plt.subplots(len(groups),2,squeeze=False,figsize=(17,3.6*len(groups)),
                             layout='constrained',sharex=True,sharey='col')
    energy_limit = float(data['absolute_energy'].max())*1.05
    for row,g in enumerate(groups):
        ranks = np.flatnonzero(labels[order]==g); members = order[ranks]
        color = GROUP_COLORS[g]
        left,right = axes[row]
        left.plot(data['dft_frequency'],data['absolute_energy'][:,members],color=color,alpha=.3,lw=.75)
        left.plot(data['dft_frequency'],data['absolute_energy'][:,members].mean(axis=1),color='black',lw=2)
        right.plot(data['dense_frequency'],data['peak_normalized_magnitude_db'][:,members],color=color,alpha=.3,lw=.65)
        chosen = example_indices(labels,order,columns) if len(columns) else []
        for index,j in enumerate(j for j in chosen if labels[j]==g):
            rank = int(np.flatnonzero(order==j)[0])
            right.plot(data['dense_frequency'],data['peak_normalized_magnitude_db'][:,j],
                       color='black',ls=('-', '--')[index],lw=1.4,
                       label=f'Native j={j} · rank {rank}')
        if any(labels[j]==g for j in chosen):
            right.legend(loc='best')
        label = f'Group {g} · {len(members)} bases · ranks {ranks[0]}–{ranks[-1]}'
        left.set_title(label+'\nAbsolute N-point DFT energy',color=color)
        right.set_title(label+'\nIndividually peak-normalized magnitude',color=color)
        left.set_ylim(0,energy_limit); right.set_ylim(-60,2)
        left.set_ylabel('|C[k,j]|² (absolute energy)')
        right.set_ylabel('Relative magnitude (dB)')
        left.legend(handles=[Line2D([],[],color=color,lw=1,label='Every member'),
                             Line2D([],[],color='black',lw=2,label='Group mean')],loc='best')
        for ax in (left,right):
            ax.set_xlim(0,1); ax.set_xticks([0,.25,.5,.75,1]); ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Normalized frequency f/Fs = k/N · periodic [0,1)')
    axes[-1,1].set_xlabel('Normalized frequency f/Fs · periodic [0,1)')
    return fig


def plot_basis_examples(data, spectral, w, columns):
    """Individually labeled time/spectral shapes beside their actual coupling columns."""
    labels, order = data['labels'],data['permutation']
    chosen = example_indices(labels,order,columns)
    fig,axes = plt.subplots(len(chosen),3,squeeze=False,figsize=(23,3.8*len(chosen)),layout='constrained')
    for row,j in enumerate(chosen):
        g = int(labels[j]); rank = int(np.flatnonzero(order==j)[0]); color = GROUP_COLORS[g]
        time,frequency,coupling = axes[row]
        identity = f'Learned · group {g} · native j={j} · rank {rank}'
        time.plot(np.arange(len(w)),w[:,j].real,color=color,label=f'Re w[:,{j}]')
        time.plot(np.arange(len(w)),w[:,j].imag,color=color,ls='--',label=f'Im w[:,{j}]')
        time.set_title(identity+'\nTime basis',color=color)
        time.set_xlabel('Time sample n'); time.set_ylabel('Complex basis coefficient'); time.legend(loc='best')
        frequency.plot(spectral['dense_frequency'],spectral['peak_normalized_magnitude_db'][:,j],
                       color=color,label=f'Native j={j}')
        frequency.set_title(identity+'\nIndividually peak-normalized magnitude',color=color)
        frequency.set_xlabel('Frequency f/Fs · periodic [0,1)'); frequency.set_ylabel('Relative magnitude (dB)')
        frequency.set_xlim(0,1); frequency.set_ylim(-60,2); frequency.legend(loc='best')
        coupling.bar(np.arange(len(w)),np.abs(data['A'][order,j])**2,
                     color=[GROUP_COLORS[int(x)] for x in labels[order]],width=1)
        coupling.set_title(identity+'\nSame sample H · transmit j → receive i',color=color)
        coupling.set_xlabel('Receive i (frozen rank; native IDs in table)'); coupling.set_ylabel('|A[i,j]|² (sample power)')
        _strip(coupling,labels[order])
        for ax in (time,frequency,coupling):
            ax.grid(alpha=.2)
    return fig


def _boundaries(ax,labels):
    sizes=np.bincount(labels); edges=np.cumsum(sizes)[:-1]-.5
    for e in edges:
        ax.axvline(e,color='white',ls='--',lw=.8)
        ax.axhline(e,color='white',ls='--',lw=.8)


def _strip(ax,labels,axis='x'):
    cm=ListedColormap(GROUP_COLORS[:len(np.unique(labels))])
    if axis=='x':
        sub=ax.inset_axes([0,1.01,1,.025]); sub.imshow(labels[None,:],aspect='auto',cmap=cm,interpolation='nearest',vmin=0,vmax=len(cm.colors)-1)
    else:
        sub=ax.inset_axes([-.035,0,.025,1]); sub.imshow(labels[:,None],aspect='auto',origin='lower',cmap=cm,interpolation='nearest',vmin=0,vmax=len(cm.colors)-1)
    sub.set_axis_off()


def plot_groups(data):
    a=data['A']; order=data['permutation']; labels=data['labels']; n=len(a)
    fig,axes=plt.subplots(2,3,figsize=(22,13),layout='constrained')
    panels=[('One sample |A|² · native indices',np.abs(a)**2,False),
            ('Same sample |A|² · frozen permutation',np.abs(a)**2,True),
            ('Mean multipath |A|² · 512 nominal channels',data['mean_channel_power'],True),
            ('Q test · 2048 nominal unitary DD atoms',data['Q_test'],True)]
    for ax,(title,power,reorder) in zip(axes.flat,panels):
        value=power[np.ix_(order,order)] if reorder else power
        im=ax.imshow(10*np.log10(np.maximum(value,1e-8)),origin='lower',vmin=-60,vmax=0,cmap='viridis')
        ax.set_title(title,pad=22); ax.set_xlabel('Transmit j (permuted rank)' if reorder else 'Transmit basis j')
        ax.set_ylabel('Receive i (permuted rank)' if reorder else 'Receive basis i')
        _strip(ax,labels[order] if reorder else labels); _strip(ax,labels[order] if reorder else labels,'y')
        if reorder: _boundaries(ax,labels)
        fig.colorbar(im,ax=ax,shrink=.7).set_label('Power (dB re 1)')
    ax=axes[1,1]; power=np.abs(data['C'])**2
    im=ax.imshow(10*np.log10(np.maximum(power[:,order],1e-8)),origin='lower',vmin=-60,vmax=0,cmap='viridis',aspect='auto')
    ax.set_title('Learned DFT power |C[k,j]|² · same columns',pad=22)
    ax.set_xlabel('Basis j (permuted rank)'); ax.set_ylabel('DFT bin k (periodic)'); _strip(ax,labels[order])
    fig.colorbar(im,ax=ax,shrink=.7).set_label('DFT coefficient power (dB re 1)')
    ax=axes[1,2]
    for g in np.unique(labels):
        # Mean per basis column: each curve sums to one over the N DFT bins.
        profile=power[:,labels==g].mean(-1)
        ax.plot(np.arange(n)/n,profile,color=GROUP_COLORS[g],label=f'Group {g} · {np.sum(labels==g)} columns')
    ax.set_title('Group-mean DFT energy · leaky / possibly split regions')
    ax.set_xlabel('Frequency k/N (periodic)'); ax.set_ylabel('Mean |C[k,j]|² per group column')
    ax.legend(loc='best'); ax.grid(alpha=.2)
    return fig


def plot_propagation(data,w,h,column):
    a,c=data['A'],data['C']; labels,order=data['labels'],data['permutation']; n=len(w)
    y=h@w[:,column]; freq,ys=spectrum(y)
    fig,axes=plt.subplots(2,2,figsize=(21,12),layout='constrained')
    ax=axes[0,0]
    ax.bar(np.arange(n),np.abs(a[order,column])**2,color=[GROUP_COLORS[g] for g in labels[order]],width=1)
    ax.set_title(f'Learned transmit j={column}, group {labels[column]} → received bases',pad=22)
    ax.set_xlabel('Receive i (frozen permuted rank; native IDs in table)'); ax.set_ylabel('Sample coupling |A[i,j]|²')
    _strip(ax,labels[order])
    ax=axes[0,1]
    for g in np.unique(labels):
        ix=np.flatnonzero(labels[order]==g)
        ax.plot(ix,data['mean_channel_power'][order[ix],column],color=GROUP_COLORS[g],ls='-')
        ax.plot(ix,data['Q_test'][order[ix],column],color=GROUP_COLORS[g],ls='--')
    ax.set_title(f'Same transmit j={column} · frozen nominal statistics',pad=22)
    ax.set_xlabel('Receive i (same permuted rank)'); ax.set_ylabel('Mean squared complex coupling')
    ax.legend(handles=[Line2D([],[],color='black',label='512-channel mean |A|²'),Line2D([],[],color='black',ls='--',label='2048-atom Q test')])
    _strip(ax,labels[order])
    ax=axes[1,0]
    ax.plot(freq,ys.real,color='black',label='Re DTFT(Hw[j])')
    ax.plot(freq,ys.imag,color='black',ls='--',label='Im DTFT(Hw[j])')
    reconstruction=w@a[:,column]
    error=np.max(np.abs(reconstruction-y))
    ax.set_title(f'Complex received spectrum · synthesis error {error:.1e}')
    ax.set_xlabel('Frequency f/Fs'); ax.set_ylabel('Complex coefficient (1/√N)'); ax.legend(loc='best')
    ax=axes[1,1]
    ax.plot(freq,np.abs(ys)**2,color='black',lw=2,label='Total |DTFT(Hw[j])|²')
    for g in np.unique(labels):
        mask=labels==g
        _,part=spectrum(w[:,mask]@a[mask,column])
        ax.plot(freq,np.abs(part)**2,color=GROUP_COLORS[g],label=f'Group {g} contribution')
    ax.set_title('Received group spectra · powers are NOT additive')
    ax.set_xlabel('Frequency f/Fs'); ax.set_ylabel('Squared magnitude (power)'); ax.legend(loc='best')
    for ax in axes.flat: ax.grid(alpha=.2)
    return fig
