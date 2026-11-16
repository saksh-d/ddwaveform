"""Bounded exports and explicit paired BER checks; no training entry point."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .artifacts import load_reference, sha, write_json
from .core import COMPLEX, REAL, rng, seed
from .delay_doppler import assemble, bridge, classical, frequency_paths, sample
from .receivers import constellation, decode, encode, receiver, stream_mse, support
from .selection import identity


def numpy(value):
    return value.detach().resolve_conj().cpu().numpy()


def load_bases(root=None):
    result = classical()
    if root is not None:
        result = {"Learned": load_reference(root), **{k: result[k] for k in ["OTFS", "AFDM"]}}
    return result


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: numpy(v) if isinstance(v, torch.Tensor) else v for k, v in values.items()})


def output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dd(out, artifacts=None, count=2, namespace=("advisor", "export")):
    if not 1 <= count <= 16:
        raise ValueError("Compact export limited to 1–16 channels")
    out = output_dir(out)
    channels = sample(count, namespace)
    bases = load_bases(artifacts)
    values = dict(H=channels.h, Hraw=channels.raw, tau=channels.tau, nu=channels.nu,
                  gains=channels.gains, normalization=channels.scale)
    if artifacts:
        with np.load(Path(artifacts) / "focused/data/bases.npz", allow_pickle=False) as source:
            for key in ["labels", "permutation", "selected"]:
                values[key] = source[key]
        with np.load(Path(artifacts) / "focused/data/supports.npz", allow_pickle=False) as source:
            values.update({key: source[key] for key in source.files})
    for name, w in bases.items():
        values.update({name + "_W": w, name + "_A": w.mH @ channels.h @ w,
                       **{name + "_" + k: v for k, v in bridge(w, channels.h).items()}})
    save_npz(out / "matrices.npz", **values)
    write_json(out / "metadata.json", {"workflow": "delay_doppler", "count": count,
        "namespace": namespace, "seed": seed(*namespace), "N": 64, "P": 3,
        "dtype": "complex128", "device": "cpu", "normalization": "Hraw/sqrt(sum(abs(Hraw)^2)/64)",
        "ordering": "native; permutation is a separate frozen discovery view",
        "orientation": "W[sample,symbol]; A[output,input]=W^H H W",
        "sha256": sha(out / "matrices.npz"), "selected_learned": identity() if artifacts else None})
    return {"output": str(out), "channels": count, "bases": list(bases)}


def export_cfo(out, artifacts, eps=(0., .05, .1)):
    from .residual_cfo import effective, load_qualified, operator
    out = output_dir(out)
    w, r, checks = load_qualified(artifacts)
    eps = torch.as_tensor(eps, dtype=torch.float32)
    h = operator(eps, w.shape[0])
    a = effective(w, r, eps)
    save_npz(out / "matrices.npz", H=h, W=w, V=r.mH, R=r, A=a,
             historical_operator=a.transpose(-1, -2), eps=eps,
             receive_noise_covariance_unit_variance=r @ r.mH)
    write_json(out / "metadata.json", {**checks, "workflow": "residual_cfo", "device": "cpu",
        "physical": "H=diag(exp(+j*2*pi*eps*n/M)); M=64, no DD normalization",
        "orientation": "W[64,45], V[64,45]=R^H, R[45,64]; A=R H W; historical operator=A.T",
        "noise": "analysis covariance sigma2 R R^H; historical Eb/N0 sigma2=1/(4*10^(EbN0/10))",
        "scope": "Qualified stored W/R and deterministic operator, no CFO BER rerun or multipath claim",
        "sha256": sha(out / "matrices.npz")})
    return {"output": str(out), **checks}


def ber_point(ws, masks, mod, snr, count, blocks, namespace, *, channel_model=None, kinds=None):
    """Every arm shares H, bits and physical noise; counts retain channel axis."""
    n = next(iter(ws.values())).shape[0]
    channels = sample(count, (*namespace, "channel"), **{"n": n, **(channel_model or {})})
    bps = 2 if mod == "QPSK" else 4
    bits = torch.randint(0, 2, (count, blocks, n, bps), generator=rng(*namespace, "bits", mod, snr))
    symbols = encode(bits, mod)
    variance = 10 ** (-float(snr) / 10)
    shape = symbols.shape
    generator = rng(*namespace, "noise", mod, snr)
    noise = math.sqrt(variance / 2) * (torch.randn(shape, generator=generator, dtype=REAL) +
                                                1j * torch.randn(shape, generator=generator, dtype=REAL))
    kinds = kinds if kinds is not None else ["full", "realized"] + (["fixed"] if masks else [])
    errors, mses = [], []
    for name, w in ws.items():
        a = w.mH @ channels.h @ w
        # Direct physical pipeline, independent of support approximation.
        physical = (symbols @ w.T) @ channels.h.transpose(-1, -2) + noise
        z = physical @ w.conj()
        basis_errors, basis_mses = [], []
        for kind in kinds:
            g = receiver(a, variance, kind, masks.get(name), k=3*n//8)
            estimate = z @ g.transpose(-1, -2)
            basis_errors.append((decode(estimate, mod) != bits).sum((1, 2, 3)))
            basis_mses.append(stream_mse(a, g, variance).mean(-1))
        errors.append(torch.stack(basis_errors))
        mses.append(torch.stack(basis_mses))
    return torch.stack(errors), torch.stack(mses), kinds, channels


def ber(out, artifacts=None, count=8, blocks=2, snrs=(0., 18.), mods=("QPSK", "16QAM")):
    if not 1 <= count <= 64 or not 1 <= blocks <= 16 or len(snrs) > 8:
        raise ValueError("Bounded reevaluation: <=64 channels, <=16 blocks, <=8 SNRs")
    out = output_dir(out)
    ws = load_bases(artifacts)
    masks = {}
    if artifacts:
        with np.load(Path(artifacts) / "focused/data/supports.npz", allow_pickle=False) as data:
            masks = {name: torch.from_numpy(data[name + "_mask"].copy()) for name in ws}
    rows = []
    for mod in mods:
        for snr in snrs:
            namespace = ("advisor", "ber-smoke")
            errors, mses, kinds, channels = ber_point(ws, masks, mod, snr, count, blocks, namespace)
            bps = 2 if mod == "QPSK" else 4
            bits_per_channel = blocks * 64 * bps
            save_npz(out / f"{mod}_{snr:g}dB.npz", errors=errors, analytic_mse=mses,
                     tau=channels.tau, nu=channels.nu, gains=channels.gains, normalization=channels.scale,
                     bits_per_channel=np.array(bits_per_channel), waveforms=np.array(list(ws)), receivers=np.array(kinds),
                     channel_seed=np.array(seed(*namespace, "channel")),
                     bits_seed=np.array(seed(*namespace, "bits", mod, snr)),
                     noise_seed=np.array(seed(*namespace, "noise", mod, snr)))
            for wi, name in enumerate(ws):
                for ri, kind in enumerate(kinds):
                    e = int(errors[wi, ri].sum())
                    rates = errors[wi, ri].to(REAL) / bits_per_channel
                    rows.append({"modulation": mod, "snr_db": snr, "waveform": name, "receiver": kind,
                        "slicer": "raw", "channels": count, "bits": count * bits_per_channel,
                        "errors": e, "ber": float(rates.mean()),
                        "channel_se": float(rates.std(unbiased=True) / math.sqrt(count)) if count > 1 else None,
                        "zero_cluster_upper95": 1 - .05 ** (1 / count) if e == 0 else None,
                        "analytic_mse": float(mses[wi, ri].mean())})
    write_json(out / "summary.json", {"qualification_only": True, "axis": "Es/N0 dB", "rows": rows,
        "namespace": ["advisor", "ber-smoke"], "channel_seed": seed("advisor", "ber-smoke", "channel"),
        "count_dependence": "count is part of draw shapes; changing count changes the bank",
        "slicing": "nearest normalized Gray constellation, no gain correction",
        "uncertainty": "channel SE descriptive; zero-event bound is conservative for independent channels",
        "device": "cpu", "dtype": "float64/complex128", "threads": torch.get_num_threads(),
        "selected_learned": identity() if artifacts else None})
    return {"output": str(out), "points": len(rows), "channels_per_point": count, "qualification_only": True}


def check(artifacts=None):
    ws = load_bases(artifacts)
    channels = sample(3, ("advisor", "check"))
    values = {}
    for name, w in ws.items():
        a = w.mH @ channels.h @ w
        b = bridge(w, channels.h)
        values[name] = {"unitarity_frobenius": float(torch.linalg.matrix_norm(w.mH @ w - torch.eye(64, dtype=COMPLEX))),
                       "bridge_relative_error": float((torch.linalg.matrix_norm(b["A_spectral"] - a) / torch.linalg.matrix_norm(a)).max())}
    if max(x["unitarity_frobenius"] for x in values.values()) > 1e-10 or max(x["bridge_relative_error"] for x in values.values()) > 1e-10:
        raise ValueError("Unitary/bridge check failed")
    return {"device": str(channels.h.device), "dtype": str(channels.h.dtype), "bases": values}
