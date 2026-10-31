"""Independent aggregation and bounded physical replay of frozen Stage-5A results."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.channel import complex_awgn, make_generator, noise_variance
from src.metrics import constellation_encode, nearest_neighbor_decode, qam16_constellation, qpsk_constellation, random_bits
from src.stage5a import MultiscaleUnitary, role_seed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "outputs/stage5a"


def read(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def check():
    torch.set_num_threads(1)
    assert (OUT / "evaluation_complete.json").exists()
    checks={}
    teacher=torch.from_numpy(np.load(OUT / "teacher.npy").copy())
    selection=json.loads((OUT / "selection_frozen.json").read_text())
    selected=next(x for x in selection["candidates"] if x["label"]=="FIT_descending_p4")
    checkpoint=OUT / "checkpoints" / (selected["checkpoint"]+".pt")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selected["checkpoint_sha256"]
    payload=torch.load(checkpoint,weights_only=True)
    model=MultiscaleUnitary(64,4,"descending")
    model.load_state_dict(payload["state"])
    action=model.compile()
    candidate=model.matrix().detach()
    raw=read("communication_raw.csv")
    groups={}
    for r in raw:
        key=(r["name"],r["modulation"],r["snr_db"],r["detector"])
        groups.setdefault(key,[]).append(r)
    assert len(groups)==192 and len(raw)==192*1024
    for rows in groups.values():
        assert [int(r["channel_index"]) for r in rows]==list(range(1024))
        assert all(0<=int(r["bit_errors"])<=int(r["total_bits"]) for r in rows)
    for row in read("communication_summary.csv"):
        key=tuple(row[k] for k in ("name","modulation","snr_db","detector"))
        rows=groups[key]
        counts=sum(int(r["bit_errors"]) for r in rows)
        bits=sum(int(r["total_bits"]) for r in rows)
        assert counts==int(row["bit_errors"]) and bits==int(row["total_bits"])
        assert counts/bits==float(row["ber"])
    checks["raw_rows_verified"]=len(raw)
    checks["summary_groups_verified"]=len(groups)
    for mod,snr,det in itertools.product(("QPSK","16QAM"),("12","15","18","20"),("legacy","gain-aware")):
        left=groups[("LEARNED_0",mod,snr,det)]
        right=groups[("EXACT_GIVENS",mod,snr,det)]
        assert [r["bit_errors"] for r in left]==[r["bit_errors"] for r in right]
    checks["exact_givens_all_channel_error_counts_match_teacher"]=True
    for name,mod,snr in itertools.product({r["name"] for r in raw},("QPSK",),("12","15","18","20")):
        assert [r["bit_errors"] for r in groups[(name,mod,snr,"legacy")]]==[r["bit_errors"] for r in groups[(name,mod,snr,"gain-aware")]]
    checks["qpsk_positive_gain_slicing_identical"]=True
    h=torch.from_numpy(np.load(OUT / "communication_bank.npz")["H"][:64].copy())
    sigma2=float(noise_variance(18))
    identity=torch.eye(64,dtype=torch.complex128)
    replay_max=0
    equivalent_error=0.
    for constellation in (qpsk_constellation(),qam16_constellation()):
        bits=random_bits((64,16,64,constellation.bits_per_symbol),
                         make_generator(role_seed("communication-symbols",constellation.name,18)))
        symbols,_=constellation_encode(bits,constellation)
        noise=complex_awgn(symbols.shape,sigma2,make_generator(role_seed("communication-noise",constellation.name,18)))
        for name,w in (("LEARNED_0",teacher),("FIT_descending_p4",candidate)):
            # Independent effective-coordinate solve, not the production physical solve.
            a=w.mH@h@w
            g=torch.linalg.solve(a.mH@a+sigma2*identity,a.mH)
            tx=symbols@w.T
            y=torch.einsum("cij,cbj->cbi",h,tx)+noise
            analyzed=y@w.conj()
            estimate=torch.einsum("cij,cbj->cbi",g,analyzed)
            gains=(g@a).diagonal(dim1=-2,dim2=-1).real
            if name.startswith("FIT"):
                equivalent_error=max(equivalent_error,float((action.apply_W(symbols)-tx).abs().max()))
            for detector,z in (("legacy",estimate),("gain-aware",estimate/gains[:,None,:])):
                decoded,_=nearest_neighbor_decode(z,constellation)
                observed=(decoded!=bits).sum((1,2,3)).numpy()
                saved=np.array([int(r["bit_errors"]) for r in groups[(name,constellation.name,"18",detector)][:64]])
                replay_max=max(replay_max,int(np.abs(observed-saved).max()))
    assert replay_max==0
    checks["independent_effective_solve_replay_counts"]=512
    checks["max_replayed_count_difference"]=replay_max
    checks["native_vs_dense_forward_max_error"]=equivalent_error
    paired=read("communication_paired.csv")
    chosen=next(r for r in paired if r["name"]=="FIT_descending_p4" and r["modulation"]=="QPSK" and r["snr_db"]=="18" and r["detector"]=="legacy")
    c=np.array([int(r["bit_errors"])/int(r["total_bits"]) for r in groups[("FIT_descending_p4","QPSK","18","legacy")]])
    b=np.array([int(r["bit_errors"])/int(r["total_bits"]) for r in groups[("LEARNED_0","QPSK","18","legacy")]])
    rng=np.random.default_rng(role_seed("bootstrap","QPSK",18,"legacy"))
    means=np.empty(5000)
    for i in range(5000):
        indices=rng.integers(0,1024,size=1024)
        means[i]=np.mean(c[indices]-1.1*b[indices])
    assert abs(np.quantile(means,.95)-float(chosen["noninferiority_upper95"]))<1e-17
    checks["paired_noninferiority_reproduced"]=True
    geom=read("stream_geometry.csv")
    baseline={(r["snr_db"],r["channel_index"]):float(r["stream_mse_mean"]) for r in geom if r["name"]=="LEARNED_0"}
    mse_error=max(abs(float(r["stream_mse_mean"])-baseline[(r["snr_db"],r["channel_index"])]) for r in geom)
    assert mse_error<1e-12
    checks["full_lmmse_mean_mse_invariance_max_error"]=mse_error
    provenance=json.loads((OUT / "provenance_fit.json").read_text())
    protected={k:v for k,v in provenance["source_hashes"].items() if not k.startswith('.context/') and k!='src/stage5a.py' and k!='reports/stage5a_protocol.json'}
    for name,digest in protected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest()==digest,name
    checks["protected_historical_files_verified"]=len(protected)
    checks["status"]="PASS_BOUNDED_INDEPENDENT_REPLAY_AND_AGGREGATION"
    checks["limitations"]="No optimization success inferred; missing Learned repeats and Q-only controls remain missing; independent implementation shares repository channel/constellation definitions."
    (OUT / "independent_verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    with torch.no_grad():
        check()
