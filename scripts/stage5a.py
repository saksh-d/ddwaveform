"""Bounded Stage-5A campaign; each phase refuses to overwrite its artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch

from src.channel import complex_awgn, dft_matrix, make_generator, noise_variance, sample_dd_v1
from src.metrics import constellation_encode, nearest_neighbor_decode, qam16_constellation, qpsk_constellation, random_bits
from src.receiver import mismatched_lmmse_equalizer
from src.stage2a import qualification_waveforms
from src.stage3b import atomic_operators
from src.stage5a import (ExactGivens, MultiscaleUnitary, fidelity, fit_matrix,
                         operator_discrepancy, role_seed)
from src.waveforms import random_unitary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage5a"
PROTOCOL = ROOT / "reports/stage5a_protocol.json"
SOURCE = ROOT / ".context/stage5a_prior/full_receiver_A_to_W_2026_10_18/inputs/learned_w.npy"
TEACHER_SHA = "11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def inputs():
    assert sha(SOURCE) == TEACHER_SHA, "archived target changed"
    w = torch.from_numpy(np.load(SOURCE).copy())
    controls = qualification_waveforms(w)
    return {"LEARNED_0": w, "HAAR_1": random_unitary(64, role_seed("haar", 1)),
            "HAAR_2": random_unitary(64, role_seed("haar", 2)),
            **{k: v for k, v in controls.items() if k != "Learned"}}


def provenance():
    paths = [*sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "reports").glob("stage[0-4]*")),
             ROOT / "researchPlan.MD", SOURCE,
             ROOT / ".context/stage5a_prior/archive_provenance.json", PROTOCOL]
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "executed_runner_sha256": sha(__file__),
            "missing": ["LEARNED_1", "LEARNED_2", "LEARNED_3", "LEARNED_4", "Stage3B Q-only"],
            "input_note": "Recovered archived array, not original .pt bytes; manifest verified. No retraining."}


def summary(values):
    x = np.asarray(values)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)),
            "tail95": float(np.sort(x)[-max(1, int(np.ceil(.05 * len(x)))):].mean()),
            "max": float(x.max())}


def fit_campaign():
    if OUT.exists():
        raise RuntimeError("preserve nonempty Stage-5A run; use a new explicitly named namespace")
    (OUT / "checkpoints").mkdir(parents=True)
    (OUT / "traces").mkdir()
    cfg = json.loads(PROTOCOL.read_text())
    write_json(OUT / "protocol.json", cfg)
    write_json(OUT / "provenance_fit.json", provenance())
    targets = inputs()
    np.save(OUT / "teacher.npy", targets["LEARNED_0"].numpy())
    exact_rows, structures, fits, qualifications = [], [], [], []
    for name, w in targets.items():
        exact = ExactGivens(w)
        restored = exact.matrix()
        probe = random_unitary(64, role_seed("exact-probe", name))[:3]
        exact_rows.append({"target": name, "rotations": len(exact.gates),
                           "relative_frobenius": float(torch.linalg.vector_norm(restored-w)/8),
                           "adjoint_max_error": float((exact.apply_W_hermitian(probe)-probe@w.conj()).abs().max()),
                           "elimination_residual": exact.elimination_residual})
        v = dft_matrix(64) @ w
        energy = abs(v).numpy()**2
        indices = np.arange(64)
        dist = abs(indices[:,None] - indices[None,:])
        dist = np.minimum(dist, 64-dist)
        singular = torch.linalg.svdvals(torch.roll(v, 1, 0)-torch.roll(v, -1, 1)).square()
        structures.append({"target": name,
                           "constant_modulus_lower_bound": float(torch.linalg.vector_norm(w.abs()-1/8)/8),
                           "frequency_band3_energy": float(energy[dist<=3].sum()/64),
                           "frequency_top8_per_column_energy": float(np.sort(energy,axis=0)[-8:].sum()/64),
                           "frequency_participation_mean": float(np.mean(1/np.sum(energy**2,axis=0))),
                           "cyclic_displacement_rank95": int(torch.searchsorted(singular.cumsum(0),.95*singular.sum())+1),
                           "pulse_peak_mean_median": float(np.median((w.abs().square()*64).max(0).values.numpy()))})
    write_csv(OUT / "exact_reference.csv", exact_rows)
    write_csv(OUT / "structural_signatures.csv", structures)
    for topology, passes in itertools.product(cfg["topologies"],cfg["passes"]):
        teacher = MultiscaleUnitary(64, passes, topology, role_seed("synthetic-target",topology,passes))
        w = teacher.matrix().detach()
        for restart in range(cfg["restarts"]):
            name = f"SYNTHETIC_{topology}_p{passes}_r{restart}"
            model = MultiscaleUnitary(64, passes, topology, role_seed("synthetic-fit",topology,passes,restart))
            model.initialize(w)
            model, trace, stats = fit_matrix(model,w,adam_steps=cfg["adam_steps"],lbfgs_steps=cfg["lbfgs_steps"])
            write_csv(OUT / "traces" / f"{name}.csv",trace)
            torch.save({"state":model.state_dict(),"target":teacher.state_dict()}, OUT / "checkpoints" / f"{name}.pt")
            qualifications.append({"name":name, "topology":topology, "passes":passes, "restart":restart,
                                    **stats, "qualified":stats["best_loss"]<1e-8})
            print(name, stats["best_loss"], flush=True)
    write_csv(OUT / "synthetic_qualification.csv",qualifications)
    for name in ("LEARNED_0","HAAR_1","HAAR_2","OFDM","OTFS","AFDM"):
        w = targets[name]
        for topology, restart in itertools.product(cfg["topologies"],range(cfg["restarts"])):
            incumbent = None
            for passes in cfg["passes"]:
                label = f"{name}_{topology}_p{passes}_r{restart}"
                model = MultiscaleUnitary(64,passes,topology,role_seed("fit",name,topology,passes,restart))
                model.initialize(w,incumbent)
                model,trace,stats = fit_matrix(model,w,adam_steps=cfg["adam_steps"],lbfgs_steps=cfg["lbfgs_steps"])
                incumbent = model
                torch.save({"n":64,"passes":passes,"topology":topology,"state":model.state_dict()},
                           OUT / "checkpoints" / f"{label}.pt")
                write_csv(OUT / "traces" / f"{label}.csv",trace)
                measured = fidelity(w,model.matrix().detach())
                fits.append({"name":label,"target":name,"topology":topology,"passes":passes,
                             "restart":restart,**stats,**{k:v for k,v in measured.items() if k!="column_errors"},
                             **{k:v for k,v in model.compile().ledger().items() if k not in ("workspace","identifiability")}})
                write_json(OUT / "traces" / f"{label}_columns.json",measured["column_errors"])
                write_csv(OUT / "fit_results.csv",fits)
                print(label, stats["best_loss"], flush=True)
    write_json(OUT / "fit_complete.json",{"fits":len(fits),"synthetic_fits":len(qualifications),
                                          "protocol_sha256":sha(PROTOCOL),"completed":time.time()})


def load_selected():
    complete=json.loads((OUT / "fit_complete.json").read_text())
    if complete["protocol_sha256"] != sha(PROTOCOL):
        raise RuntimeError("fitting protocol changed")
    with (OUT / "fit_results.csv").open() as handle:
        fits = list(csv.DictReader(handle))
    expected=set(itertools.product(("LEARNED_0","HAAR_1","HAAR_2","OFDM","OTFS","AFDM"),
                                    ("ascending","descending"),(1,2,4),(0,1)))
    actual=[(r["target"],r["topology"],int(r["passes"]),int(r["restart"])) for r in fits]
    if len(actual)!=72 or set(actual)!=expected or complete["fits"]!=72:
        raise RuntimeError("incomplete or duplicate fit grid; cannot freeze evaluation")
    matrices, actions, selected = inputs(), {}, []
    matrices["EXACT_GIVENS"]=ExactGivens(matrices["LEARNED_0"]).matrix()
    for topology, passes in itertools.product(("ascending","descending"),(1,2,4)):
        options=[r for r in fits if r["target"]=="LEARNED_0" and r["topology"]==topology and int(r["passes"])==passes]
        row=min(options,key=lambda r:float(r["best_loss"]))
        label=f"FIT_{topology}_p{passes}"
        payload=torch.load(OUT / "checkpoints" / f"{row['name']}.pt",weights_only=True)
        model=MultiscaleUnitary(64,passes,topology)
        model.load_state_dict(payload["state"])
        matrices[label]=model.matrix().detach()
        compilation=[]
        for _ in range(11):
            start=time.perf_counter_ns()
            compiled=model.compile()
            compilation.append(time.perf_counter_ns()-start)
        compiled.compilation_nanoseconds=compilation[1:]
        actions[label]=compiled
        selected.append({"label":label,"checkpoint":row["name"],"objective":float(row["best_loss"]),
                         "checkpoint_sha256":sha(OUT / "checkpoints" / f"{row['name']}.pt")})
    return matrices,actions,selected


def evaluate():
    if (OUT / "selection_frozen.json").exists():
        raise RuntimeError("evaluation already frozen; no final-bank-driven reselection")
    matrices,actions,selected=load_selected()
    write_json(OUT / "selection_frozen.json",{"candidates":selected,"protocol_sha256":sha(PROTOCOL),
                                              "diagnostic_status":"cold synthetic fitter unqualified; evaluate attained matrices, no capacity or successful-compression inference",
                                              "note":"frozen before generating any final bank","time":time.time()})
    write_json(OUT / "provenance_evaluation.json",provenance())
    source_dir=OUT / "executed_evaluation_sources"
    source_dir.mkdir()
    for source in [ROOT / "scripts/stage5a.py", *sorted((ROOT / "src").glob("*.py"))]:
        shutil.copy2(source,source_dir / (source.parent.name+"_"+source.name))
    cfg=json.loads(PROTOCOL.read_text())
    teacher=matrices["LEARNED_0"]
    generator=make_generator(role_seed("final-atoms"))
    delays=4*torch.rand(512,generator=generator,dtype=torch.float64)
    dopplers=1.5*torch.rand(512,generator=generator,dtype=torch.float64)-.75
    atoms=atomic_operators(delays,dopplers)
    channels=sample_dd_v1(512,role_seed("final-operator-channels")).matrices
    np.savez(OUT / "operator_banks.npz",delays=delays.numpy(),dopplers=dopplers.numpy(),channels=channels.numpy())
    metrics,raw=[],{}
    q0=(teacher.mH@atoms@teacher).abs().square().mean(0)
    mask0=torch.topk(q0,24,dim=-1).indices
    for name,w in matrices.items():
        db=operator_discrepancy(teacher,w,atoms)
        da=operator_discrepancy(teacher,w,channels)
        raw[f"{name}_db"],raw[f"{name}_da"]=db,da
        q=(w.mH@atoms@w).abs().square().mean(0)
        idx=torch.topk(q,24,dim=-1).indices
        support=sum(len(set(a.tolist())&set(b.tolist())) for a,b in zip(mask0,idx))/(64*24)
        metrics.append({"name":name,**{k:v for k,v in fidelity(teacher,w).items() if k!="column_errors"},
                        **{f"db_{k}":v for k,v in summary(db).items()},
                        **{f"da_{k}":v for k,v in summary(da).items()},
                        "q_eta24":float(torch.topk(q,24,dim=-1).values.sum()/q.sum()),
                        "q_support_overlap_teacher":support})
    write_csv(OUT / "representation_summary.csv",metrics)
    np.savez(OUT / "operator_discrepancies.npz",**raw)
    timing(matrices,actions)
    communication(matrices,actions,cfg)
    write_json(OUT / "evaluation_complete.json",{"completed":time.time(),"candidates":len(selected)})


def timing(matrices,actions):
    rows=[]
    for name in ("LEARNED_0",*actions):
        w=matrices[name]
        representations={"dense":(lambda x:x@w.T,lambda x:x@w.conj())}
        if name in actions:
            a=actions[name]
            representations["native_cached"]=(a.apply_W,a.apply_W_hermitian)
        for batch in (1,32):
            x=random_unitary(64,role_seed("timing",batch))[:batch]
            for implementation,(forward,adjoint) in representations.items():
                for direction,call in (("forward",forward),("adjoint",adjoint)):
                    reference=x@(w.T if direction=="forward" else w.conj())
                    error=float((call(x)-reference).abs().max())
                    for _ in range(5): call(x)
                    for repeat in range(41):
                        start=time.perf_counter_ns()
                        call(x)
                        elapsed=time.perf_counter_ns()-start
                        rows.append({"name":name,"batch":batch,"implementation":implementation,
                                     "direction":direction,"repeat":repeat,"nanoseconds":elapsed,
                                     "max_absolute_error":error})
    write_csv(OUT / "latency_raw.csv",rows)
    ledgers=[]
    for name,action in actions.items():
        ledgers.append({"name":name,**action.ledger(),
                        "compilation_nanoseconds_raw":action.compilation_nanoseconds,
                        "compilation_median_nanoseconds":float(np.median(action.compilation_nanoseconds)),
                        "deployment_note":"parameter file may be discarded after compiling; cache coefficients plus indices counted; PyTorch tensor object overhead excluded"})
    write_json(OUT / "cost_ledger.json",ledgers)
    # Complete physical LMMSE chain for one reused channel, with setup separated.
    h=sample_dd_v1(1,role_seed("timing-physical-channel")).matrices[0]
    constellation=qam16_constellation()
    receiver_setup=[]
    for repeat in range(21):
        start=time.perf_counter_ns()
        g=mismatched_lmmse_equalizer(h,noise_variance(18))
        receiver_setup.append(time.perf_counter_ns()-start)
    full_rows=[]
    for name in ("LEARNED_0",*actions):
        w=matrices[name]
        for batch in (1,32):
            x=random_unitary(64,role_seed("timing-full",batch))[:batch]
            noise=torch.ones_like(x)*.001
            for implementation in ("dense","native_cached"):
                if implementation=="native_cached" and name not in actions:
                    continue
                action=actions.get(name)
                def chain():
                    tx=action.apply_W(x) if implementation=="native_cached" else x@w.T
                    y=tx@h.T+noise
                    filtered=y@g.T
                    z=action.apply_W_hermitian(filtered) if implementation=="native_cached" else filtered@w.conj()
                    return nearest_neighbor_decode(z,constellation)
                for _ in range(5): chain()
                for repeat in range(41):
                    start=time.perf_counter_ns()
                    chain()
                    full_rows.append({"name":name,"batch":batch,"implementation":implementation,
                                      "repeat":repeat,"nanoseconds":time.perf_counter_ns()-start,
                                      "receiver_setup_median_nanoseconds":float(np.median(receiver_setup[1:])),
                                      "scope":"TX,H propagation,physical full equalizer,analysis,legacy 16QAM demap; channel reused; H generation excluded"})
    write_csv(OUT / "complete_chain_latency_raw.csv",full_rows)
    write_json(OUT / "receiver_setup_latency_raw.json",receiver_setup)


def communication(matrices,actions,cfg):
    settings=cfg["communication"]
    bank=sample_dd_v1(settings["count"],role_seed("final-communication"))
    np.savez(OUT / "communication_bank.npz",H=bank.matrices.numpy(),delays=bank.delays.numpy(),
             dopplers=bank.dopplers.numpy(),gains=bank.pre_normalization_gains.numpy(),
             normalization_scales=bank.normalization_scales.numpy())
    names=[k for k in matrices if not k.startswith("HAAR")]
    rows,geometry=[],[]
    n,blocks=64,settings["blocks_per_channel"]
    for constellation,snr in itertools.product((qpsk_constellation(),qam16_constellation()),settings["snrs_db"]):
        sigma2=noise_variance(snr)
        symbols_rng=make_generator(role_seed("communication-symbols",constellation.name,snr))
        noise_rng=make_generator(role_seed("communication-noise",constellation.name,snr))
        for start in range(0,len(bank.matrices),64):
            h=bank.matrices[start:start+64]
            bits=random_bits((len(h),blocks,n,constellation.bits_per_symbol),symbols_rng)
            symbols,true_idx=constellation_encode(bits,constellation)
            noise=complex_awgn(symbols.shape,sigma2,noise_rng)
            g=mismatched_lmmse_equalizer(h,sigma2)
            gh=g@h
            for name in names:
                w=matrices[name]
                native=actions.get(name)
                x=native.apply_W(symbols) if native else symbols@w.T
                y=torch.einsum("cij,cbj->cbi",h,x)+noise
                physical=torch.einsum("cij,cbj->cbi",g,y)
                estimate=native.apply_W_hermitian(physical) if native else physical@w.conj()
                gains=(w.mH@gh@w).diagonal(dim1=-2,dim2=-1).real
                if float(gains.min())<=0:
                    raise RuntimeError("invalid full-receiver gain")
                if constellation.name=="QPSK":
                    for offset,stream in enumerate(1-gains):
                        geometry.append({"name":name,"snr_db":snr,"channel_index":start+offset,
                                         "stream_mse_mean":float(stream.mean()),
                                         "stream_mse_cvar95":float(torch.topk(stream,4).values.mean()),
                                         "minimum_gain":float(gains[offset].min())})
                for detector,z in (("legacy",estimate),("gain-aware",estimate/gains[:,None,:])):
                    decoded,indices=nearest_neighbor_decode(z,constellation)
                    errors=(decoded!=bits).sum((1,2,3))
                    symbol_errors=(indices!=true_idx).sum((1,2))
                    for offset in range(len(h)):
                        rows.append({"name":name,"modulation":constellation.name,"snr_db":snr,
                                     "detector":detector,"channel_index":start+offset,
                                     "bit_errors":int(errors[offset]),"total_bits":blocks*n*constellation.bits_per_symbol,
                                     "symbol_errors":int(symbol_errors[offset]),"total_symbols":blocks*n})
        write_csv(OUT / "communication_raw.csv",rows)
        write_csv(OUT / "stream_geometry.csv",geometry)
        print("communication",constellation.name,snr,"complete",flush=True)
    aggregate_communication(rows,settings)


def aggregate_communication(rows,settings):
    groups={}
    for row in rows:
        groups.setdefault((row["modulation"],row["snr_db"],row["detector"],row["name"]),[]).append(row)
    summaries,comparisons=[],[]
    for (mod,snr,det,name),samples in groups.items():
        errors=sum(r["bit_errors"] for r in samples)
        bits=sum(r["total_bits"] for r in samples)
        summaries.append({"name":name,"modulation":mod,"snr_db":snr,"detector":det,
                          "bit_errors":errors,"total_bits":bits,"channels":len(samples),"ber":errors/bits})
        teacher=groups[(mod,snr,det,"LEARNED_0")]
        cand=np.array([r["bit_errors"]/r["total_bits"] for r in samples])
        base=np.array([r["bit_errors"]/r["total_bits"] for r in teacher])
        assert [r["channel_index"] for r in samples]==[r["channel_index"] for r in teacher]
        rng=np.random.default_rng(role_seed("bootstrap",mod,snr,det))
        idx=rng.integers(0,len(samples),size=(settings["bootstrap_resamples"],len(samples)))
        delta=(cand-base)[idx].mean(1)
        ni=(cand-1.1*base)[idx].mean(1)
        comparisons.append({"name":name,"modulation":mod,"snr_db":snr,"detector":det,
                            "delta_ber":float((cand-base).mean()),"delta_ci95_low":float(np.quantile(delta,.025)),
                            "delta_ci95_high":float(np.quantile(delta,.975)),
                            "noninferiority_statistic":float((cand-1.1*base).mean()),
                            "noninferiority_upper95":float(np.quantile(ni,.95)),
                            "pointwise_noninferior":bool(np.quantile(ni,.95)<0),
                            "error_bearing_channels":int(np.sum(cand>0)),"paired_channels":len(samples)})
    write_csv(OUT / "communication_summary.csv",summaries)
    write_csv(OUT / "communication_paired.csv",comparisons)


def main():
    global OUT, SOURCE
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","evaluate"))
    parser.add_argument("--output",type=Path,default=OUT,
                        help="fresh output namespace for reproduction; never reuse completed results")
    parser.add_argument("--teacher",type=Path,default=SOURCE,
                        help="archived learned_w.npy or preserved outputs/stage5a/teacher.npy")
    args=parser.parse_args()
    OUT,SOURCE=args.output.resolve(),args.teacher.resolve()
    torch.set_num_threads(1)
    with torch.no_grad() if args.phase=="evaluate" else torch.enable_grad():
        fit_campaign() if args.phase=="fit" else evaluate()


if __name__=="__main__":
    main()
