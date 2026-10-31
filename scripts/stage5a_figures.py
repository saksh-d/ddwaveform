"""Re-render compact Stage-5A figures from frozen numerical artifacts."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "outputs/stage5a"
FIG=OUT / "figures"


def read(name):
    with (OUT / name).open() as h:
        return list(csv.DictReader(h))


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.labelsize":18,
                         "xtick.labelsize":15,"ytick.labelsize":15,
                         "legend.fontsize":16,"legend.title_fontsize":16,
                         "axes.titlesize":17,"figure.dpi":100,"savefig.dpi":300})
    FIG.mkdir(exist_ok=True)


def save(fig,name):
    fig.savefig(FIG / (name+".png"),dpi=300)
    plt.close(fig)


def main():
    style()
    fit=read("fit_results.csv")
    synthetic=read("synthetic_qualification.csv")
    fig,axes=plt.subplots(1,2,figsize=(11.9,5.9),sharex=True,sharey=True)
    colors={"LEARNED_0":"#0072b2","HAAR_1":"#d55e00","HAAR_2":"#009e73","SYNTHETIC":"#777777"}
    for ax,topology in zip(axes,("ascending","descending")):
        for target,label in (("LEARNED_0","Learned 0"),("HAAR_1","Haar 1"),("HAAR_2","Haar 2")):
            low,high=[],[]
            for passes in (1,2,4):
                errors=[float(r["native_relative_frobenius"]) for r in fit if r["target"]==target and r["topology"]==topology and int(r["passes"])==passes]
                low.append(min(errors));high.append(max(errors))
            ax.plot((448,832,1600),low,"o-",color=colors[target],label=label)
            ax.fill_between((448,832,1600),low,high,color=colors[target],alpha=.15)
        low=[min(float(r["best_loss"])**.5 for r in synthetic if r["topology"]==topology and int(r["passes"])==p) for p in (1,2,4)]
        ax.plot((448,832,1600),low,"s--",color="#777777",label="Known-family target")
        ax.axhline(1e-4,color="black",ls=":",lw=1)
        ax.set_title(topology.capitalize()+" scale order")
        ax.set_yscale("log");ax.set_xticks((448,832,1600));ax.grid(alpha=.2)
    axes[0].set_ylabel("Relative Frobenius error")
    fig.supxlabel("Nominal stored real parameters",fontsize=18,y=.20)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.34,top=.91,wspace=.10)
    save(fig,"compressibility")

    rep=read("representation_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(11.9,5.7))
    for ax,metric,title in zip(axes,("db_mean","da_mean"),("512 fractional DD atoms","512 complete multipath channels")):
        for topology,color in (("ascending","#0072b2"),("descending","#d55e00")):
            values=[float(next(r for r in rep if r["name"]==f"FIT_{topology}_p{p}")[metric]) for p in (1,2,4)]
            ax.plot((448,832,1600),values,"o-",color=color,label=topology.capitalize())
        ax.set_title(title);ax.set_xticks((448,832,1600));ax.grid(alpha=.25)
        ax.set_ylim(0,1.4)
    axes[0].set_ylabel("Mean normalized\nsquared discrepancy")
    fig.supxlabel("Nominal stored real parameters",fontsize=18,y=.13)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
    fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.90,wspace=.20)
    save(fig,"complex_operator_fidelity")

    comm=read("communication_summary.csv")
    palette=[("LEARNED_0","Learned teacher","black","o"),
             ("FIT_ascending_p4","4-pass ascending","#0072b2","s"),
             ("FIT_descending_p4","4-pass descending","#009e73","^"),
             ("OFDM","OFDM","#cc79a7","v"),("OTFS","OTFS","#e69f00","D"),
             ("AFDM","AFDM","#d55e00","x")]
    fig,axes=plt.subplots(1,2,figsize=(11.9,6.2))
    for ax,mod in zip(axes,("QPSK","16QAM")):
        for name,label,color,marker in palette:
            rows=sorted([r for r in comm if r["name"]==name and r["modulation"]==mod and r["detector"]=="legacy"],key=lambda r:int(r["snr_db"]))
            ax.semilogy([int(r["snr_db"]) for r in rows],[float(r["ber"]) for r in rows],color=color,marker=marker,label=label)
        ax.set_title(mod+", complete LMMSE")
        ax.set_xlabel("SNR (dB)");ax.set_xticks((12,15,18,20));ax.grid(alpha=.25,which="both")
    axes[0].set_ylabel("Bit error rate")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.27,top=.92,wspace=.22)
    save(fig,"full_receiver_ber")

    latency=read("latency_raw.csv")
    fig,axes=plt.subplots(1,2,figsize=(11.9,5.8))
    for ax,batch in zip(axes,("1","32")):
        for topology,color in (("ascending","#0072b2"),("descending","#d55e00")):
            native,dense=[],[]
            for p in (1,2,4):
                rows=[r for r in latency if r["name"]==f"FIT_{topology}_p{p}" and r["batch"]==batch and r["direction"]=="forward"]
                native.append(np.median([float(r["nanoseconds"])/1000 for r in rows if r["implementation"]=="native_cached"]))
                dense.append(np.median([float(r["nanoseconds"])/1000 for r in rows if r["implementation"]=="dense"]))
            ax.plot((448,832,1600),native,"o-",color=color,label=topology.capitalize()+", native")
            ax.plot((448,832,1600),dense,"o--",color=color,label=topology.capitalize()+", dense")
        ax.set_title("Batch = "+batch);ax.set_yscale("log");ax.grid(alpha=.25)
        ax.set_xticks((448,832,1600))
    axes[0].set_ylabel("Median forward latency (µs)")
    fig.supxlabel("Nominal stored real parameters",fontsize=18,y=.20)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.34,top=.91,wspace=.2)
    save(fig,"native_application_cost")
    (FIG / "style.json").write_text(json.dumps({"font":"DejaVu Sans","labels":18,"ticks":15,
                                                "panel_labels":17,"legend":16,"dpi":300,
                                                "overall_title":False,"format":"PNG"},indent=2)+"\n")


if __name__=="__main__":
    main()
