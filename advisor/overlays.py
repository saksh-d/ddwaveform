"""Display-only basis overlay alternatives; no fitting or new evidence selection."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OVERLAY_VARIANTS = ('overlay8_complex','overlay_half_complex','overlay8_energy_magnitude','overlay_half_energy_magnitude')
REVIEW_VARIANTS = (*OVERLAY_VARIANTS,'grouped_spectra')


def validate_review(review,n):
    if review is None:
        review={'variants':('grouped_spectra',),'overlay8_columns':tuple(range(0,n,n//8)),
                'half_columns':tuple(range(0,n,2))}
    if set(review)!={'variants','overlay8_columns','half_columns'}:
        raise ValueError('REVIEW needs variants, overlay8_columns and half_columns')
    variants=review['variants']
    if not isinstance(variants,(tuple,list)) or len(set(variants))!=len(variants) or any(x not in REVIEW_VARIANTS for x in variants):
        raise ValueError('Choose unique supported PLOT_VARIANTS')
    result={'variants':tuple(variants)}
    for key,count in [('overlay8_columns',8),('half_columns',n//2)]:
        ids=review[key]
        if not isinstance(ids,(tuple,list)) or len(ids)!=count or len(set(ids))!=count:
            raise ValueError(f'{key} requires exactly {count} unique native basis indices')
        if any(type(j) is not int or not 0<=j<n for j in ids):
            raise ValueError(f'{key} indices must be integers in [0,{n-1}]')
        result[key]=tuple(ids)
    return result


def ordered_columns(order,columns):
    """Same discovery order as the coupling view; never sort by spectral appearance."""
    selected=np.asarray(order)[np.isin(order,columns)]
    if len(selected)!=len(columns):
        raise ValueError("Overlay indices must be unique members of the frozen permutation")
    return selected


def curve_key(n,j,rank):
    # Native-ID color stays fixed when the selected subset or plot variant changes.
    return plt.get_cmap('turbo')((j+.5)/n), ('-','--','-.',':')[rank%4]


def overlay_mapping(labels,order,columns):
    selected=ordered_columns(order,columns)
    inverse=np.argsort(order)
    return [{'native_basis_index':int(j),'group':int(labels[j]),'permuted_rank':int(inverse[j])}
            for j in selected]


def overlay_values(w,spectral,columns):
    ids=ordered_columns(spectral['permutation'],columns)
    return {'indices':ids,'time':w[:,ids],
            'complex_spectrum':spectral['complex_spectrum'][:,ids],
            'absolute_energy':spectral['absolute_energy'][:,ids],
            'magnitude_db':spectral['peak_normalized_magnitude_db'][:,ids]}


def plot_overlay(w,spectral,columns,representation):
    if representation not in ('complex','energy_magnitude'):
        raise ValueError('Unknown overlay representation')
    values=overlay_values(w,spectral,columns);ids=values['indices'];n=len(w);count=len(ids)
    inverse=np.argsort(spectral['permutation'])
    legend_cols=4 if count<=8 else 6
    key_height=.38*math.ceil(count/legend_cols)+.35
    graph_rows=2 if representation=='complex' else 1
    fig=plt.figure(figsize=(21,3.7*graph_rows+key_height),layout='constrained')
    grid=fig.add_gridspec(graph_rows+1,2,height_ratios=[3.7]*graph_rows+[key_height])
    axes=[fig.add_subplot(grid[r,c]) for r in range(graph_rows) for c in range(2)]
    handles=[]
    for index,j in enumerate(ids):
        rank=int(inverse[j]);group=int(spectral['labels'][j]);color,line=curve_key(n,int(j),rank)
        handles.append(Line2D([],[],color=color,ls=line,lw=2,label=f'j={j} · G{group} · r={rank}'))
        if representation=='complex':
            series=[(np.arange(n),values['time'][:,index].real),
                    (np.arange(n),values['time'][:,index].imag),
                    (spectral['dense_frequency'],values['complex_spectrum'][:,index].real),
                    (spectral['dense_frequency'],values['complex_spectrum'][:,index].imag)]
        else:
            series=[(spectral['dft_frequency'],values['absolute_energy'][:,index]),
                    (spectral['dense_frequency'],values['magnitude_db'][:,index])]
        for ax,(x,y) in zip(axes,series):
            ax.plot(x,y,color=color,ls=line,lw=1 if count<=8 else .65,alpha=.85 if count<=8 else .55)
    if representation=='complex':
        titles=['Time · real','Time · imaginary','Complex DTFT · real','Complex DTFT · imaginary']
        for ax,title in zip(axes,titles):
            ax.set_title(f'Learned · {count} of N={n} basis columns\n'+title)
        extent=max(float(np.abs(values['time'].real).max()),float(np.abs(values['time'].imag).max()))*1.05
        frequency_extent=max(float(np.abs(values['complex_spectrum'].real).max()),float(np.abs(values['complex_spectrum'].imag).max()))*1.05
        for ax in axes[:2]:
            ax.set_xlabel('Time sample n');ax.set_ylabel('Complex basis coefficient');ax.set_ylim(-extent,extent)
        for ax in axes[2:]:
            ax.set_xlabel('Frequency f/Fs · periodic [0,1)');ax.set_ylabel('Complex coefficient (1/√N)')
            ax.set_xlim(0,1);ax.set_ylim(-frequency_extent,frequency_extent)
    else:
        for ax,title in zip(axes,['Absolute N-point DFT coefficient energy','Individually peak-normalized spectral magnitude']):
            ax.set_title(f'Learned · {count} of N={n} basis columns\n'+title)
            ax.set_xlabel('Frequency f/Fs · periodic [0,1)');ax.set_xlim(0,1)
        axes[0].set_ylabel('|C[k,j]|² (absolute energy)');axes[0].set_ylim(bottom=0)
        axes[1].set_ylabel('Relative magnitude (dB)');axes[1].set_ylim(-60,2)
    for ax in axes:ax.grid(alpha=.2)
    key=fig.add_subplot(grid[-1,:]);key.set_axis_off()
    key.legend(handles=handles,ncol=legend_cols,loc='center',frameon=False,
               title='Curve key: native j · coupling group G · frozen rank r',
               handlelength=2,columnspacing=1.1,labelspacing=.5)
    return fig
