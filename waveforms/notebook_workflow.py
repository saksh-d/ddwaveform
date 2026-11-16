"""Sequential research notebook workflow; saved replay and bounded fresh checks."""
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import json
import math
import numpy as np
import pandas as pd
import torch
from .artifacts import provision, sha, verify, write_json
from .core import COMPLEX, configure, seed
from .delay_doppler import Channels, assemble, classical, dft, sample
from .experiments import ber_point, numpy, save_npz
from .figures import plot_basis, plot_ber, plot_channels, style
from .selection import WAVEFORMS, selected_frame
from .basis_source import SelectedBasis, require_available_mode, select_basis
from .scaling_evidence import ScalingEvidence, prepare_scaling, load_scaling, GRIDS, SIZES
from .coupling import analyze, plot_groups, plot_propagation, grouped_spectral_data, plot_grouped_spectra, example_indices, plot_basis_examples

from .overlays import OVERLAY_VARIANTS, validate_review, overlay_mapping, plot_overlay

NOMINAL = {"n": 64, "paths": 3, "delay_range": (0., 4.), "doppler_range": (-.75, .75)}
ESSENTIALS = ("channel_matrices_native", "basis_time", "basis_frequency", "ber_QPSK_raw", "ber_16QAM_raw", "coupling_groups", "coupling_propagation", "grouped_spectra", "basis_examples")
FRESH_KEYS = {"seed", "matrix_channels", "matrix_index", "ber_channels", "blocks_per_channel", "snrs_db"}


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def validate_config(mode, model, display, fresh=None, review=None):
    """Reject unsupported/unused settings instead of relabeling precomputed data."""
    if mode not in ("saved_replay", "fresh"):
        raise ValueError("EVIDENCE_MODE must be saved_replay or fresh; this setting does not select/train a basis")
    if set(model) != set(NOMINAL) or set(display) not in ({"basis_columns", "snr_max_db"}, {"basis_columns", "snr_max_db", "example_columns"}):
        raise ValueError("Unsupported or missing MODEL/DISPLAY keys")
    if type(model["n"]) is not int or model["n"] not in SIZES:
        raise ValueError("N=64/128/256 only: own verified learned basis required, no resizing")
    _integer(model["paths"], 1, 6, "paths")
    normalized = dict(model)
    for name, lower, upper in [("delay_range", 0., 8.), ("doppler_range", -2., 2.)]:
        if not isinstance(model[name], (tuple, list)):
            raise ValueError(f"{name} must contain two numeric limits")
        values = tuple(model[name])
        if len(values) != 2 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values) or not lower <= values[0] < values[1] <= upper:
            raise ValueError(f"{name} needs increasing finite limits within [{lower},{upper}]")
        normalized[name] = values
    if not isinstance(display["basis_columns"], (tuple, list)):
        raise ValueError("basis_columns must be a list/tuple of column indices")
    columns = tuple(display["basis_columns"])
    if not 1 <= len(columns) <= 8 or len(set(columns)) != len(columns):
        raise ValueError("Choose 1–8 unique basis_columns")
    for column in columns:
        _integer(column, 0, model["n"]-1, "basis column")
    examples = display.get("example_columns", columns)
    if not isinstance(examples,(tuple,list)) or not 1 <= len(examples) <= 8 or len(set(examples)) != len(examples):
        raise ValueError("Choose1–8 unique example_columns (native basis IDs)")
    for column in examples:
        _integer(column,0,model["n"]-1,"example column")
    limit = display["snr_max_db"]
    if limit is not None and (not isinstance(limit, (int, float)) or not math.isfinite(limit) or not -5 <= limit <= 40):
        raise ValueError("snr_max_db must be None or finite in [-5,40]")
    if mode == "saved_replay":
        if normalized != {**NOMINAL, "n":model["n"]}:
            raise ValueError("Saved BER evidence is fixed to selected N/P3, delay[0,4], Doppler[-.75,.75]; use fresh for model changes")
        if fresh is not None:
            raise ValueError("FRESH must be None in saved_replay; seeds/budgets/SNR grids cannot alter saved curves")
        if limit is not None and limit < 10:
            raise ValueError("Replay display needs snr_max_db>=10 or None to show multiple16QAM points")
    else:
        if fresh is None or set(fresh) != FRESH_KEYS:
            raise ValueError("Fresh mode requires exactly the documented FRESH seed/count/grid settings")
        _integer(fresh["seed"], 0, 2**31-1, "seed")
        _integer(fresh["matrix_channels"], 1, 16, "matrix_channels")
        _integer(fresh["matrix_index"], 0, fresh["matrix_channels"]-1, "matrix_index")
        _integer(fresh["ber_channels"], 2, 64, "ber_channels")
        _integer(fresh["blocks_per_channel"], 1, 16, "blocks_per_channel")
        snrs = tuple(fresh["snrs_db"])
        if not 2 <= len(snrs) <= 8 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not -5 <= x <= 40 for x in snrs) or any(a >= b for a,b in zip(snrs, snrs[1:])):
            raise ValueError("snrs_db needs 2–8 strictly increasing finite values in [-5,40]")
        if limit is not None and sum(x <= limit for x in snrs) < 2:
            raise ValueError("SNR display must include at least two simulated points")
        if fresh["ber_channels"] * fresh["blocks_per_channel"] * len(snrs) > 4096:
            raise ValueError("Bounded fresh cap: ber_channels × blocks_per_channel × SNR points <=4096")
        fresh = {**fresh, "snrs_db": snrs}
    return {"review": validate_review(review,model["n"]), "evidence_mode": mode, "model": normalized, "display": {"basis_columns": columns, "example_columns":tuple(examples), "snr_max_db": limit}, "fresh": fresh}


class NotebookWorkflow:
    """State retained in memory; figures are generated/displayed here, never loaded from a plot folder."""
    def __init__(self, config, artifacts, workspace, save_run, *, selected_basis, scaling=None):
        self.config, self.artifacts = config, Path(artifacts)
        self.environment = configure()
        if not isinstance(selected_basis, SelectedBasis):
            raise ValueError("A basis provider must return SelectedBasis; no implicit loading fallback")
        self.selected_basis = selected_basis.validate()
        self.n = config["model"]["n"]; self.k = SIZES[self.n]
        if selected_basis.matrix.shape != (self.n,self.n):
            raise ValueError("Selected basis size differs from configured N; no substitution")
        self.scaling = scaling
        self.saved_reference = scaling.matches(selected_basis) if scaling else selected_basis.matches_saved_reference(self.artifacts)
        if config["evidence_mode"] == "saved_replay" and not self.saved_reference:
            raise ValueError("Saved BER curves belong to the verified reference W/support; select fresh evidence for another basis")
        if self.n != 64 and scaling is None:
            raise FileNotFoundError("Size-specific evidence required; no N64 fallback")
        controls = classical(self.n,*GRIDS[self.n])
        self.ws = {"Learned": selected_basis.matrix.clone(), **{name: controls[name] for name in WAVEFORMS[1:]}}
        self._selected = self.ws["Learned"].clone()
        self.bases = {name: numpy(w) for name,w in self.ws.items()}
        self.bases["dft"] = numpy(dft(self.n))
        if scaling:
            self.masks={name:torch.from_numpy(data["mask_nominal"].copy()) for name,data in scaling.records.items()}
        if scaling and self.saved_reference:
            record=scaling.records["Learned"]
            self.reference_views={"labels":record["labels"],"permutation":record["permutation"],"selected":np.array(config["display"]["basis_columns"])}
            self.masks={name:torch.from_numpy(data["mask_nominal"].copy()) for name,data in scaling.records.items()}
        elif self.saved_reference:
            with np.load(self.artifacts / "focused/data/bases.npz", allow_pickle=False) as data:
                self.reference_views = {key: data[key].copy() for key in ("labels", "permutation", "selected")}
        else:
            self.reference_views = {"labels": np.full(self.n, -1), "permutation": np.arange(self.n),
                                    "selected": np.array([], dtype=np.int64)}
        if not scaling:
            with np.load(self.artifacts / "focused/data/supports.npz", allow_pickle=False) as data:
                self.masks = {name: torch.from_numpy(data[name+"_mask"].copy()) for name in WAVEFORMS}
        self.bases["labels"] = self.reference_views["labels"]
        self.example_columns()  # Fail invalid native/group selection before creating any local archive.
        self.masks["Learned"] = selected_basis.fixed_q_mask.clone()
        self.channels = self.effective_channels = self.ber_table = None
        self.counts, self.images = {}, {}
        self.matrix_index = 0
        self.run_dir = None
        if save_run:
            name = config["evidence_mode"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
            self.run_dir = Path(config.get('output_root',Path(workspace) / "local/notebook_runs")) / name
            self.run_dir.mkdir(parents=True)
        self.provenance = {"selected_learned": selected_basis.describe(), "environment": self.environment,
            "config": config, "fixed_support": "Provider-supplied Learned fixed K=3N/8; frozen nominal baseline masks; realized complex coefficients, no hidden recalibration",
            "reference_group_views_applicable": self.saved_reference,
            "BER_status": "Not yet evaluated", "evidence_mode": config["evidence_mode"]}
        self.provenance["inputs_sha256"] = {name: sha(self.artifacts / name) for name in
            ("reference/learned_w.npy", "focused/data/bases.npz", "focused/data/channels.npz", "focused/data/supports.npz", "focused/ber_summary.csv")} if not scaling else scaling.input_hashes()

    def summary(self):
        description = "Saved-evidence replay: reconstruct saved H/A; load completed BER counts. NO BER simulation." if self.config["evidence_mode"] == "saved_replay" else "Fresh bounded check: new H/A and paired uncoded BER simulation. NOT a production campaign."
        return {"operation": description, "active_config": self.config, "selected_learned": self.selected_basis.describe(),
                "device_dtype_threads": self.environment, "fixed_support": self.provenance["fixed_support"],
                "available_sizes": {"64":"K24 · historical export", "128":"K48 · one fresh nominal trial", "256":"K96 · one fresh nominal trial"},
                "statistics": "Frozen nominal groups, Q(2048 atoms) and mean |A|²(512 channels); fresh channel changes do not refit them"}

    def generate_channels(self):
        if self.channels is not None:
            return self.channels, self.effective_channels
        model = self.config["model"]
        if self.config["evidence_mode"] == "saved_replay" and self.scaling:
            record=self.scaling.records["Learned"]
            tau,nu,gains=(torch.from_numpy(record[k].copy())[None] for k in ("example_tau","example_nu","example_gains"))
            raw=assemble(tau,nu,gains,n=self.n); scale=(raw.abs().square().sum((-2,-1))/self.n).sqrt()
            self.channels=Channels(raw/scale[:,None,None],raw,tau,nu,gains,scale)
            self.provenance["matrix_generation"]={"method":"Reconstruct frozen structural channel index0; same H for all waveforms", "bank_count":512, "namespace":["scaling-dd-transfer-v1","structural-channel",0], "seed":seed("scaling-dd-transfer-v1","structural-channel",0), "source_batch_count":32, "source":"saved paths/gains; original index0, no outcome selection"}
            for name,w in self.ws.items():
                if "example" in self.scaling.records[name]:
                    torch.testing.assert_close((w.mH@self.channels.h@w)[0],torch.from_numpy(self.scaling.records[name]["example"]),atol=2e-12,rtol=1e-12)
        elif self.config["evidence_mode"] == "saved_replay":
            with np.load(self.artifacts / "focused/data/channels.npz", allow_pickle=False) as saved:
                tau, nu, gains = (torch.from_numpy(saved[k].copy())[None] for k in ("tau", "nu", "gains"))
                raw = assemble(tau, nu, gains)
                scale = (raw.abs().square().sum((-2,-1)) / 64).sqrt()
                h = raw / scale[:,None,None]
                torch.testing.assert_close(h[0], torch.from_numpy(saved["H"]), atol=1e-12, rtol=1e-12)
                self.provenance["matrix_generation"] = {"method": "Reconstructed from frozen paths/gains",
                    "original_structural_index": int(saved["channel_index"]), "bank_count": 2048,
                    "bank_namespace": ["global", "structural-channel"], "bank_seed": seed("global", "structural-channel")}
            self.channels = Channels(h, raw, tau, nu, gains, scale)
        else:
            fresh = self.config["fresh"]
            namespace = ("advisor-notebook-v1", fresh["seed"], "matrix")
            self.channels = sample(fresh["matrix_channels"], namespace, n=self.n, p=model["paths"],
                                   delay_range=model["delay_range"], doppler_range=model["doppler_range"])
            self.matrix_index = fresh["matrix_index"]
            self.provenance["matrix_generation"] = {"namespace": namespace, "seed": seed(*namespace), "index": self.matrix_index}
        assert self.channels.h.device.type == "cpu" and self.channels.h.dtype == COMPLEX
        self.effective_channels = {name: w.mH @ self.channels.h @ w for name,w in self.ws.items()}
        return self.channels, self.effective_channels

    def matrix_summary(self):
        self.generate_channels()
        values = {"H": self.channels.h, **{name+" A": a for name,a in self.effective_channels.items()}}
        return pd.DataFrame([{"matrix": name, "shape": tuple(a.shape), "dtype": str(a.dtype),
                              "example complex [0,0]": complex(a[self.matrix_index,0,0])} for name,a in values.items()])

    def path_table(self):
        self.generate_channels(); i=self.matrix_index
        return pd.DataFrame({'delay':self.channels.tau[i].numpy(), 'Doppler':self.channels.nu[i].numpy(),
                             'complex gain':self.channels.gains[i].numpy()})

    def show_basis_analysis(self):
        from IPython.display import display, JSON
        self.show('basis_time'); self.show('basis_frequency')
        with pd.option_context('display.max_rows',self.n):
            display(self.group_table())
            display(self.overlay_table('overlay8')); display(self.overlay_table('half'))
        data=self.coupling_data()
        print('max |WᴴW-I|:',data['unitarity_max'],'max |CᴴH_fC-A|:',data['bridge_max'])
        display(JSON(self.review_summary(),expanded=True))
        self.show('coupling_groups'); self.show_variations()
        display(self.example_table()); self.show('basis_examples'); self.show('coupling_propagation')

    def show_ber_figures(self):
        from IPython.display import display
        frame=self.evaluate_ber()
        print(self.provenance['BER_status'])
        display(frame)
        for mod in frame.modulation.unique(): self.show(f'ber_{mod}_raw')

    def evaluate_ber(self):
        if self.ber_table is not None:
            return self.ber_table
        if self.config["evidence_mode"] == "saved_replay":
            frame = self.scaling.ber() if self.scaling else selected_frame(pd.read_csv(self.artifacts / "focused/ber_summary.csv"))
            self.ber_table = frame[(frame.slicer == "raw") & frame.receiver.isin(["fixed", "full"])].copy()
            self.provenance["BER_status"] = "LOADED completed saved counts; no BER simulation/retraining. Channel-bootstrap intervals approximate under adaptive/budget stopping; flags retained."
            return self.ber_table
        fresh, model = self.config["fresh"], self.config["model"]
        namespace = ("advisor-notebook-v1", fresh["seed"], "ber")
        rows = []
        for mod in ("QPSK", "16QAM"):
            bits_per_channel = fresh["blocks_per_channel"] * self.n * (2 if mod == "QPSK" else 4)
            for snr in fresh["snrs_db"]:
                errors, mses, kinds, channel = ber_point(self.ws, self.masks, mod, snr,
                    fresh["ber_channels"], fresh["blocks_per_channel"], namespace,
                    channel_model={"p": model["paths"], "delay_range": model["delay_range"], "doppler_range": model["doppler_range"]},
                    kinds=["fixed", "full"])
                self.counts[f"{mod}_{snr:g}dB"] = {"errors": numpy(errors), "analytic_mse": numpy(mses),
                    "tau": numpy(channel.tau), "nu": numpy(channel.nu), "gains": numpy(channel.gains),
                    "normalization": numpy(channel.scale), "bits_per_channel": np.array(bits_per_channel),
                    "channel_seed": np.array(seed(*namespace,"channel")),
                    "bits_seed": np.array(seed(*namespace,"bits",mod,snr)), "noise_seed": np.array(seed(*namespace,"noise",mod,snr))}
                for wi,name in enumerate(WAVEFORMS):
                    for ri,kind in enumerate(kinds):
                        rate = errors[wi,ri].to(torch.float64) / bits_per_channel
                        total = int(errors[wi,ri].sum()); bits = bits_per_channel * fresh["ber_channels"]
                        mean = total / bits; se = float(rate.std(unbiased=True) / math.sqrt(len(rate)))
                        rows.append({"modulation": mod, "snr_db": snr, "waveform": name, "receiver": kind,
                            "slicer": "raw", "channels": len(rate), "bits": bits, "errors": total, "ber": mean,
                            "channel_se": se if total else None,
                            "ci_low": max(0.,mean-1.96*se) if total else np.nan,
                            "ci_high": min(1.,mean+1.96*se) if total else np.nan,
                            "zero_cluster_upper95": 1-.05**(1/len(rate)) if total == 0 else np.nan,
                            "detection_limit": 1/bits, "analytic_mse": float(mses[wi,ri].mean())})
        self.ber_table = pd.DataFrame(rows)
        self.provenance["BER_status"] = "NEW bounded simulation; descriptive +/-1.96 channel-SE bands, not campaign/bootstrap intervals. Zero errors unresolved; conservative channel bound in table."
        self.provenance["BER_namespace"] = namespace
        return self.ber_table

    def coupling_data(self):
        if not self.scaling or not self.saved_reference:
            raise ValueError("Qualified size-specific coupling statistics unavailable; provision SCALING_BUNDLE, no borrowed groups")
        self.generate_channels()
        return analyze(self.bases["Learned"],numpy(self.channels.h[self.matrix_index]),self.scaling.records["Learned"])

    def grouped_spectra_data(self):
        data=self.coupling_data()
        return grouped_spectral_data(self.bases["Learned"],data["labels"],data["permutation"])

    def example_columns(self):
        return example_indices(self.reference_views["labels"],self.reference_views["permutation"],
            self.config["display"]["example_columns"])

    def example_table(self):
        table=self.group_table()
        return table[table.selected_example].reset_index(drop=True)

    def group_table(self):
        data=self.coupling_data(); order=data["permutation"]
        c=data["C"]; labels=data["labels"]
        return pd.DataFrame({"permuted_rank":np.arange(self.n),"native_basis_index":order,
            "group":labels[order],"dominant_DFT_bin":np.abs(c[:,order]).argmax(axis=0),
            "selected_example":np.isin(order,self.example_columns())})

    def required_figures(self):
        return (set(ESSENTIALS)-{'grouped_spectra'}) | set(self.config['review']['variants'])

    def review_summary(self):
        return {'enabled_variants':self.config['review']['variants'],
                'overlay8_count':8,'half_count':self.n//2,
                'ordering':'Frozen coupling permutation; native indices in curve key/table',
                'scope':'Readability alternatives; same frozen Learned, no training or BER changes',
                'representations':'Complex real/imaginary components vs absolute energy / individually normalized dB'}

    def overlay_table(self,kind):
        if kind not in ('overlay8','half'):
            raise ValueError('Choose overlay8 or half mapping')
        return pd.DataFrame(overlay_mapping(self.reference_views['labels'],self.reference_views['permutation'],
                                           self.config['review'][kind+'_columns']))

    def show_variations(self):
        for name in self.config['review']['variants']:
            self.show(name)

    def show(self, name):
        from IPython.display import Image, display
        import matplotlib.pyplot as plt
        if name not in ESSENTIALS and name not in OVERLAY_VARIANTS:
            raise ValueError("Only essential matrix/basis/BER and requested spectral-coupling figures are supported")
        if not torch.equal(self.ws["Learned"], self._selected):
            raise ValueError("Selected Learned matrix changed in memory; restart from configuration")
        style()
        if name == "channel_matrices_native":
            self.generate_channels()
            data = {"H": numpy(self.channels.h[self.matrix_index]),
                    **{key+"_example": numpy(a[self.matrix_index]) for key,a in self.effective_channels.items()}}
            fig = plot_channels(data)
        elif name in ("basis_time", "basis_frequency"):
            fig = plot_basis(self.bases, self.config["display"]["basis_columns"], name == "basis_frequency")
        elif name in OVERLAY_VARIANTS:
            kind='overlay8' if name.startswith('overlay8_') else 'half'
            representation='complex' if name.endswith('_complex') else 'energy_magnitude'
            fig=plot_overlay(self.bases['Learned'],self.grouped_spectra_data(),
                             self.config['review'][kind+'_columns'],representation)
        elif name == "grouped_spectra":
            fig=plot_grouped_spectra(self.grouped_spectra_data(),self.config["display"]["example_columns"])
        elif name == "basis_examples":
            fig=plot_basis_examples(self.coupling_data(),self.grouped_spectra_data(),self.bases["Learned"],
                                    self.config["display"]["example_columns"])
        elif name in ("coupling_groups","coupling_propagation"):
            data=self.coupling_data()
            fig=plot_groups(data) if name=="coupling_groups" else plot_propagation(data,self.bases["Learned"],numpy(self.channels.h[self.matrix_index]),self.config["display"]["basis_columns"][0])
        else:
            table = self.evaluate_ber(); mod = "QPSK" if name == "ber_QPSK_raw" else "16QAM"
            label = "Saved BER replay" if self.config["evidence_mode"] == "saved_replay" else "Fresh BER smoke"
            fig = plot_ber(table, mod, self.config["display"]["snr_max_db"], provenance=label, k=self.k,
                           support_label=self.config.get('fixed_support_label','Fixed discovery Q'))
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300)
        plt.close(fig)
        data = buffer.getvalue(); self.images[name] = data
        if self.run_dir:
            (self.run_dir / (name+".png")).write_bytes(data)
        display(Image(data=data, width=1100))

    def finish(self):
        if set(self.images) != self.required_figures():
            raise ValueError("Run all essential and spectral/coupling figure cells before finishing")
        if self.run_dir:
            ch = self.channels
            save_npz(self.run_dir / "matrices.npz", H=ch.h, Hraw=ch.raw, tau=ch.tau, nu=ch.nu,
                gains=ch.gains, normalization=ch.scale,
                labels=self.reference_views["labels"], permutation=self.reference_views["permutation"],
                frozen_selected_columns=self.reference_views["selected"],
                displayed_columns=np.array(self.config["display"]["basis_columns"]),
                example_native_columns=self.example_columns(),
                **{name+"_W": w for name,w in self.ws.items()}, **{name+"_A": a for name,a in self.effective_channels.items()},
                **{name+"_mask": mask for name,mask in self.masks.items()})
            if self.scaling:
                values=self.coupling_data()
                save_npz(self.run_dir/"spectral_coupling.npz",**values)
                save_npz(self.run_dir/"grouped_spectra.npz",**self.grouped_spectra_data(),
                         example_native_columns=self.example_columns())
                self.example_table().to_csv(self.run_dir/"examples.csv",index=False)
            if self.scaling:
                for kind in ('overlay8','half'):
                    self.overlay_table(kind).to_csv(self.run_dir/(kind+'_mapping.csv'),index=False)
            self.ber_table.to_csv(self.run_dir / "ber.csv", index=False)
            for name,values in self.counts.items():
                save_npz(self.run_dir / (name+".npz"), **values)
            self.provenance["source_sha256"] = {name: sha(Path(__file__).parent / name) for name in
                ("basis_source.py", "notebook_workflow.py", "figures.py", "coupling.py", "overlays.py", "scaling_evidence.py", "delay_doppler.py", "experiments.py")}
            self.provenance["files_sha256"] = {p.name: sha(p) for p in self.run_dir.iterdir() if p.is_file()}
            write_json(self.run_dir / "run.json", self.provenance)
        return {"basis_mode": self.config.get("basis_mode", "provider_supplied"), "evidence_mode": self.config["evidence_mode"], "essential_inline_figures": len(self.images),
                "core_inline_figures":len(set(self.images)-set(self.config["review"]["variants"])),
                "review_inline_figures":len(set(self.images)&set(self.config["review"]["variants"])),
                "BER_status": self.provenance["BER_status"], "optional_local_archive": str(self.run_dir) if self.run_dir else None,
                "training": self.selected_basis.provenance.get("training_status", "No training performed by downstream evaluation cells")}


def start_workflow(*, mode, evidence_mode, model, display, fresh, artifacts=None, bundle=None, workspace, save_run=True, scaling_artifacts=None, scaling_bundle=None, review=None, output_root=None):
    # Unavailable training must fail before provisioning, loading or creating a run.
    require_available_mode(mode)
    config = {**validate_config(evidence_mode, model, display, fresh, review=review), "basis_mode": mode}
    if output_root is not None:
        config['output_root']=str(output_root)
    if type(save_run) is not bool:
        raise ValueError("SAVE_RUN must be True or False")
    workspace = Path(workspace).resolve()
    if scaling_artifacts is not None:
        scaling=load_scaling(workspace,scaling_artifacts,scaling_bundle,model["n"])
        return NotebookWorkflow(config, scaling.root, workspace, save_run,
                                selected_basis=scaling.selected,scaling=scaling)
    if model["n"] != 64:
        raise FileNotFoundError("Size-specific scaling artifacts required; no N64 substitution")
    if artifacts is None:
        raise FileNotFoundError("Provide qualified scaling evidence or legacy N64 artifacts")
    artifacts = Path(artifacts).resolve()
    manifest = json.loads((workspace / "support/configs/artifacts.json").read_text())
    if not artifacts.exists():
        if not artifacts.is_relative_to(workspace / "local"):
            raise ValueError("New artifact provisioning must stay under this checkout's local/ directory")
        if bundle is None or not Path(bundle).is_file():
            raise FileNotFoundError("Local artifact bundle missing: set ARTIFACT_BUNDLE in the configuration cell. No download or source-project fallback.")
        provision(bundle, artifacts, manifest)
    verify(artifacts, manifest)
    selected = select_basis(mode, artifacts)
    return NotebookWorkflow(config, artifacts, workspace, save_run, selected_basis=selected)
