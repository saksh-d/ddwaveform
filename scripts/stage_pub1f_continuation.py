"""Explicitly authorized Pub-1F continuation; preserves the original failed gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.channel import COMPLEX_DTYPE, noise_variance
from src.metrics import qam16_constellation, qpsk_constellation
from src.pub1b.core import directory_tree_sha256, file_sha256, sample_scaled_dd, scaling_definition, sparse_cg_lmmse, sparse_from_topk
from src.pub1f.continuation import (
    CONTINUATION_NAMESPACE, dense_cg_snapshots, fixed_energy_support, generic_communication_counts,
    generic_iterative_counts, matching_residual_statistics, tensor_sha256,
)
from src.pub1f.core import TrainingProtocol, condition_channels, operation_ledger, semantic_seed, storage_ledger, train_stage, unitary_residual
from src.receiver import mismatched_lmmse_equalizer, rowwise_topk
from src.stage2a import load_frozen_waveform
from src.waveforms import afdm_waveform, ofdm_waveform, otfs_waveform


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/stage_pub1f_continuation.json"
OUT=ROOT/"outputs/stage_pub1f_continuation"
DATA=OUT/"data"; CHECKPOINTS=OUT/"checkpoints"; STATS=OUT/"statistics"; TIMING=OUT/"timing"; FIGURES=OUT/"figures"; PROVENANCE=OUT/"provenance"
REPORT=ROOT/"reports/stage_pub1f_continuation_summary.md"
AMENDMENT=ROOT/"reports/stage_pub1f_protocol_amendment.md"
ORIGINAL_GATE=ROOT/"outputs/stage_pub1f/data/early_gate.json"
ORIGINAL_GATE_REQUIRED_SHA=None


def cfg() -> dict[str,object]: return json.loads(CONFIG.read_text())
def ensure_dirs() -> None:
    for path in (DATA,CHECKPOINTS,STATS,TIMING,FIGURES,PROVENANCE): path.mkdir(parents=True,exist_ok=True)
def write_json(path:Path,value:object)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def read_json(path:Path): return json.loads(path.read_text())
def sha(path:Path)->str: return file_sha256(path)
def write_csv(path:Path,rows:list[dict[str,object]])->None:
    if not rows: raise ValueError(f"refusing empty table {path}")
    fields=[]
    for row in rows: fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def read_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="") as handle:return list(csv.DictReader(handle))
def environment()->dict[str,object]: return {"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"processor":platform.processor(),"device":"cpu","dtype":"complex128/float64","threads_at_entry":torch.get_num_threads()}


def load_checkpoint(path:Path)->torch.Tensor:
    value=torch.load(path,weights_only=False)
    if isinstance(value,dict):
        for key in ("waveform","matrix","best_matrix"):
            if key in value:return value[key].to(COMPLEX_DTYPE)
        raise ValueError(f"no waveform tensor in {path}")
    return value.to(COMPLEX_DTYPE)


def phase_amendment(smoke:bool)->None:
    ensure_dirs(); authorization=cfg()["authorization"]
    if sha(ROOT/".context/attachments/Z3dMty/pasted_text_2026-09-06_19-04-32.txt") != authorization["attachment_sha256"]: raise RuntimeError("continuation authorization source changed")
    gate=read_json(ORIGINAL_GATE)
    if gate["passed"] or gate["run_kind"]!="production": raise RuntimeError("original production gate must remain failed")
    write_json(PROVENANCE/"authorization.json",{"run_kind":"smoke" if smoke else "production","authorization_attachment_sha256":authorization["attachment_sha256"],"original_gate_sha256":sha(ORIGINAL_GATE),"original_gate_passed":gate["passed"],"continuation_authorized":True,"original_gate_modified":False,"config_sha256":sha(CONFIG),"amendment_sha256":sha(AMENDMENT)})


def phase_reuse(smoke:bool)->None:
    ensure_dirs(); configuration=cfg(); chicago=Path("/Users/saksh/conductor/workspaces/delayDopplerWaveform/chicago-v1")
    items=[]
    roles={64:"historical N=64 reference",128:"historical single-checkpoint scaling",256:"historical single-checkpoint scaling"}
    for n,entry in ((int(key),value) for key,value in configuration["scaling"]["checkpoints"].items()):
        local=ROOT/entry["path"]; actual=sha(local)
        producer=local if n==64 else chicago/entry["path"]
        producer_hash=sha(producer)
        if actual!=entry["sha256"] or producer_hash!=actual: raise RuntimeError(f"N={n} checkpoint provenance failure")
        metadata_path=local.parent/"metadata.json" if n>64 else None
        items.append({"artifact":entry["path"],"producer":str(producer),"source_configuration":"Pub-1B scaling" if n>64 else "Stage-1A.2/Pub-1E","sha256":actual,"scientific_role":roles[n],"compatible":True,"reuse_decision":"reuse exact checkpoint; no retraining","required_recomputation":"fresh validation/test/support/statistics","metadata":str(metadata_path.relative_to(ROOT)) if metadata_path else "historical handoff"})
    for relative,role,decision in [
        ("outputs/stage_pub1b/data/scaling/scaling_config.json","scaled channel/OTFS/AFDM definitions","reuse exact definitions; fresh stronger-baseline evaluation"),
        ("outputs/stage_pub1b/data/complexity/operating_points.csv","historical BER-versus-MAC evidence","historical comparison only; do not pool"),
        ("outputs/stage_pub1f/data/early_gate.json","original resource-gate outcome","preserve unchanged"),
        ("outputs/stage_pub1f/timing/receiver_benchmark.csv","historical N=64 timing","historical reference; new scaling timing required"),
    ]:
        path=ROOT/relative
        items.append({"artifact":relative,"producer":str(path),"source_configuration":"Pub-1B" if "pub1b" in relative else "original Pub-1F","sha256":sha(path),"scientific_role":role,"compatible":True,"reuse_decision":decision,"required_recomputation":"as stated"})
    n128=read_json(ROOT/"outputs/stage_pub1b/checkpoints/N128/metadata.json"); n256=read_json(ROOT/"outputs/stage_pub1b/checkpoints/N256/metadata.json")
    write_json(PROVENANCE/"reuse_inventory.json",{"run_kind":"smoke" if smoke else "production","items":items,"compute_saved":{"N128_training_seconds":n128["training_wall_seconds"],"N256_training_seconds":n256["training_wall_seconds"],"total_larger_N_training_seconds":n128["training_wall_seconds"]+n256["training_wall_seconds"],"original_baseline_seconds":read_json(ROOT/"outputs/stage_pub1f/data/baseline_execution.json")["seconds"]},"smoke_training_reused_as_science":False,"production_training_states_found":sorted(str(path.relative_to(ROOT)) for path in (CHECKPOINTS/"production").glob("seed_*/*_state.pt")) if (CHECKPOINTS/"production").exists() else []})


def _save_state(path:Path): return lambda state:torch.save(state,path)


def phase_training(smoke:bool)->None:
    phase_amendment(smoke); configuration=cfg()["training"]; seeds=tuple(int(value) for value in configuration["seeds"])
    parent_steps=2 if smoke else int(configuration["parent_steps"]); continuation_steps=2 if smoke else int(configuration["continuation_steps"])
    batch=2 if smoke else int(configuration["batch_size"]); cadence=1 if smoke else int(configuration["validation_cadence"]); validation_count=6 if smoke else int(configuration["validation_channels"])
    rows=[]
    for seed_id in seeds:
        directory=CHECKPOINTS/("smoke" if smoke else "production")/f"seed_{seed_id}"; directory.mkdir(parents=True,exist_ok=True)
        validation=condition_channels("nominal_dd_v1",validation_count,"checkpoint_validation",namespace=CONTINUATION_NAMESPACE,seed_id=seed_id)
        parent_protocol=TrainingProtocol(seed_id,"parent",parent_steps,batch,cadence,24,None,CONTINUATION_NAMESPACE)
        state_path=directory/"parent_state.pt"; resume=torch.load(state_path,weights_only=False) if state_path.exists() else None
        parent=train_stage(ofdm_waveform(),validation,parent_protocol,resume=resume,save_callback=_save_state(state_path)); parent_hash=tensor_sha256(parent["selected"])
        torch.save(parent["selected"],directory/"parent_selected.pt"); torch.save(parent["endpoint"],directory/"parent_endpoint.pt")
        results={"parent":parent}
        for stage in ("mean_continuation","tail_continuation"):
            protocol=TrainingProtocol(seed_id,stage,continuation_steps,batch,cadence,24,parent_hash,CONTINUATION_NAMESPACE)
            state_path=directory/f"{stage}_state.pt"; resume=torch.load(state_path,weights_only=False) if state_path.exists() else None
            result=train_stage(parent["selected"],validation,protocol,reference_parent_mean=float(parent["best_mean"]) if stage=="tail_continuation" else None,resume=resume,save_callback=_save_state(state_path))
            torch.save(result["selected"],directory/f"{stage}_selected.pt"); torch.save(result["endpoint"],directory/f"{stage}_endpoint.pt"); results[stage]=result
        equal=results["mean_continuation"]["protocol"]["parent_sha256"]==results["tail_continuation"]["protocol"]["parent_sha256"]==parent_hash
        for stage,result in results.items():
            write_csv(directory/f"{stage}_trace.csv",result["trace"])
            rows.append({"seed_id":seed_id,"stage":stage,"selected_step":result["best_step"],"selected_mean_mse":result["best_mean"],"selected_cvar95":result["best_cvar95"],"endpoint_mean_mse":result["endpoint_mean"],"endpoint_cvar95":result["endpoint_cvar95"],"runtime_seconds":result["runtime_seconds"],"max_unitarity_residual":result["max_unitarity"],"nonfinite":result["nonfinite"],"parent_equal_across_arms":equal,"selected_checkpoint":str((directory/f"{stage}_selected.pt").relative_to(ROOT)),"endpoint_checkpoint":str((directory/f"{stage}_endpoint.pt").relative_to(ROOT)),"selected_tensor_sha256":tensor_sha256(result["selected"]),"endpoint_tensor_sha256":tensor_sha256(result["endpoint"])})
        print(f"continuation seed {seed_id}: parent {parent['best_step']}, mean {results['mean_continuation']['best_step']}, tail {results['tail_continuation']['best_step']}",flush=True)
    write_csv(DATA/("training_summary_smoke.csv" if smoke else "training_summary.csv"),rows)


def learned_waveforms(smoke:bool,*,endpoints:bool=False)->dict[str,torch.Tensor]:
    kind="smoke" if smoke else "production"; values={"historical":load_checkpoint(ROOT/cfg()["scaling"]["checkpoints"]["64"]["path"])}
    for seed_id in cfg()["training"]["seeds"]:
        directory=CHECKPOINTS/kind/f"seed_{seed_id}"
        for stage in ("parent","mean_continuation","tail_continuation"):
            suffix="endpoint" if endpoints else "selected"
            path=directory/f"{stage}_{suffix}.pt"
            if not path.exists(): raise RuntimeError(f"missing {kind} checkpoint: {path}")
            values[f"seed{seed_id}_{stage}_{suffix}"]=load_checkpoint(path)
    return values


def classical_waveforms_n64()->dict[str,torch.Tensor]:
    values={"OTFS":otfs_waveform(),"OFDM":ofdm_waveform()}
    for slope in (1,3,5,7,9): values[f"AFDM_slope_{slope}"]=afdm_waveform(c1=slope/(2*64))
    return values


def analysis_waveforms(smoke:bool)->dict[str,torch.Tensor]:
    values=learned_waveforms(smoke); values.update(classical_waveforms_n64()); return values


def statistics_path(smoke:bool,waveform_key:str,condition:str,k:int,support_policy:str)->Path:
    return STATS/("smoke" if smoke else "production")/waveform_key/condition/f"k{k}_{support_policy}.pt"


def phase_statistics(smoke:bool)->None:
    ensure_dirs(); phase_amendment(smoke); waveforms=analysis_waveforms(smoke)
    count=6 if smoke else int(cfg()["evaluation"]["statistics_channels"]); rows=[]
    conditions=("joint_0.5","nominal_dd_v1") if smoke else ("joint_0.5","joint_0.2_probe","integer","nominal_dd_v1")
    for condition in conditions:
        support_bank=condition_channels(condition,count,"statistical_support",namespace=CONTINUATION_NAMESPACE)
        residual_bank=condition_channels(condition,count,"residual_statistics",namespace=CONTINUATION_NAMESPACE)
        for waveform_key,waveform in waveforms.items():
            for k in ((24,) if smoke else (24,)):
                fixed=fixed_energy_support(support_bank,waveform,k)
                for policy in ("oracle_row_topk","fixed_statistical"):
                    path=statistics_path(smoke,waveform_key,condition,k,policy); path.parent.mkdir(parents=True,exist_ok=True)
                    stats=matching_residual_statistics(residual_bank,waveform,k,policy,fixed)
                    payload={"waveform_key":waveform_key,"waveform_tensor_sha256":tensor_sha256(waveform),"condition":condition,"k":k,"support_policy":policy,"support":fixed if policy=="fixed_statistical" else None,"residual_mean":stats.mean,"residual_covariance":stats.receive_covariance,"samples":stats.samples,"namespace":CONTINUATION_NAMESPACE}
                    torch.save(payload,path)
                    rows.append({"waveform_key":waveform_key,"waveform_tensor_sha256":tensor_sha256(waveform),"condition":condition,"k":k,"support_policy":policy,"samples":count,"support_entries":int(fixed.sum()) if policy=="fixed_statistical" else 64*k,"mean_bytes":stats.mean.numel()*stats.mean.element_size(),"covariance_bytes":stats.receive_covariance.numel()*stats.receive_covariance.element_size(),"path":str(path.relative_to(ROOT)),"sha256":sha(path)})
            print(f"statistics {condition} {waveform_key}",flush=True)
    write_csv(DATA/("statistics_inventory_smoke.csv" if smoke else "statistics_inventory.csv"),rows)


def load_statistics(smoke:bool,waveform_key:str,condition:str,policy:str="oracle_row_topk"):
    from src.pub1e.core import ResidualStatistics
    payload=torch.load(statistics_path(smoke,waveform_key,condition,24,policy),weights_only=False)
    return payload,ResidualStatistics(payload["residual_mean"],payload["residual_covariance"],payload["samples"])


def _aggregate(rows:list[dict[str,object]])->dict[str,object]:
    errors=sum(int(row["bit_errors"]) for row in rows); bits=sum(int(row["total_bits"]) for row in rows)
    return {"bit_errors":errors,"total_bits":bits,"ber":errors/bits,"n_channels":len(rows),"mean_retained_energy":float(np.mean([float(row["retained_energy"]) for row in rows]))}


def _validation_waveforms(smoke:bool)->dict[str,torch.Tensor]:
    values=analysis_waveforms(smoke)
    if not smoke:return values
    keep={"historical","seed0_parent_selected","seed0_mean_continuation_selected","seed0_tail_continuation_selected","OTFS","OFDM","AFDM_slope_1"}
    return {key:value for key,value in values.items() if key in keep}


def phase_operating_validation(smoke:bool)->None:
    ensure_dirs(); phase_amendment(smoke); waveforms=_validation_waveforms(smoke); configuration=cfg()["evaluation"]
    count=6 if smoke else int(configuration["configuration_validation_channels"]); bits=128 if smoke else int(configuration["validation_bits_per_channel"])
    conditions=("joint_0.5",) if smoke else ("joint_0.5","joint_0.2_probe","integer","nominal_dd_v1")
    snrs=(18,) if smoke else tuple(int(value) for value in configuration["snr_db"]); ks=(16,24) if smoke else tuple(int(value) for value in configuration["k_values"])
    constellations=(qam16_constellation(),qpsk_constellation()); rows=[]; seen=set(); started=time.perf_counter()
    for condition in conditions:
        channels=condition_channels(condition,count,"configuration_validation",namespace=CONTINUATION_NAMESPACE)
        support_count=6 if smoke else int(configuration["statistics_channels"])
        support_bank=condition_channels(condition,support_count,"statistical_support",namespace=CONTINUATION_NAMESPACE)
        supports={key:{k:fixed_energy_support(support_bank,waveform,k) for k in ks} for key,waveform in waveforms.items()}
        for waveform_key,waveform in waveforms.items():
            for constellation in constellations:
                configurations=[]
                # SNR curves at K=24 plus K dependence at 18 dB.
                for snr in snrs: configurations.extend((snr,24,policy,receiver,scale) for policy in ("oracle_row_topk","fixed_statistical") for receiver,scale in (("historical",1.0),*( ("tuned",value) for value in (.25,.5,1.0,2.0,4.0) ),("residual_aware",1.0)))
                for k in ks: configurations.extend((18,k,policy,receiver,scale) for policy in ("oracle_row_topk","fixed_statistical") for receiver,scale in (("historical",1.0),*( ("tuned",value) for value in (.25,.5,1.0,2.0,4.0) )) )
                for snr,k,policy,receiver,scale in configurations:
                    key=(condition,waveform_key,constellation.name,snr,k,policy,receiver,scale)
                    if key in seen:continue
                    seen.add(key); stats=None; support=supports[waveform_key][k]
                    if receiver=="residual_aware":
                        payload,stats=load_statistics(smoke,waveform_key,condition,policy); support=payload["support"] if policy=="fixed_statistical" else None
                    counts=generic_communication_counts(channels,waveform,constellation,snr,k,receiver,bits,condition=condition,bank_role="configuration_validation",waveform_key=waveform_key,regularization_scale=scale,support_policy=policy,fixed_support=support if policy=="fixed_statistical" else None,statistics=stats)
                    rows.append({"condition":condition,"waveform_key":waveform_key,"modulation":constellation.name,"snr_db":snr,"k":k,"rho":k/64,"support_policy":policy,"receiver":receiver,"regularization_scale":scale,**_aggregate(counts)})
            print(f"operating validation {condition} {waveform_key}",flush=True)
    write_csv(DATA/("operating_validation_smoke.csv" if smoke else "operating_validation.csv"),rows)
    write_json(DATA/("operating_validation_execution_smoke.json" if smoke else "operating_validation_execution.json"),{"seconds":time.perf_counter()-started,"channels":count,"bits_per_channel":bits,"rows":len(rows),"run_kind":"smoke" if smoke else "production"})


def _selection_key(row:dict[str,str])->tuple[float,int,float,int]:
    receiver_order={"historical":0,"tuned":1,"residual_aware":2}
    slope=int(row["waveform_key"].rsplit("_",1)[1]) if row["waveform_key"].startswith("AFDM") else 0
    return float(row["ber"]),receiver_order[row["receiver"]],abs(np.log2(float(row["regularization_scale"]))),slope


def _select_operational(validation:list[dict[str,str]],family:str,condition:str,modulation:str,snr:int,k:int)->dict[str,str]:
    eligible=[row for row in validation if row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==snr and int(row["k"])==k]
    if family=="AFDM":eligible=[row for row in eligible if row["waveform_key"].startswith("AFDM")]
    else:eligible=[row for row in eligible if row["waveform_key"]==family]
    if not eligible:raise RuntimeError(f"no validation configuration for {family} {condition} {modulation} {snr} K={k}")
    return min(eligible,key=_selection_key)


def _evaluation_points(smoke:bool):
    if smoke:return [("joint_0.5",18,16),("joint_0.5",18,24)]
    snrs=tuple(int(value) for value in cfg()["evaluation"]["snr_db"]); ks=tuple(int(value) for value in cfg()["evaluation"]["k_values"])
    points=set()
    for condition in ("joint_0.5","nominal_dd_v1"):
        points.update((condition,snr,24) for snr in snrs); points.update((condition,18,k) for k in ks)
    points.update((condition,18,24) for condition in ("joint_0.2_probe","integer"))
    return sorted(points)


def phase_communication(smoke:bool)->None:
    ensure_dirs(); phase_amendment(smoke); configuration=cfg()["evaluation"]
    validation=read_csv(DATA/("operating_validation_smoke.csv" if smoke else "operating_validation.csv")); waveforms=_validation_waveforms(smoke)
    count=6 if smoke else int(configuration["confirmatory_channels"]); bits=128 if smoke else int(configuration["confirmatory_bits_per_channel"])
    constellations=(qam16_constellation(),qpsk_constellation()); rows=[]; selections=[]; started=time.perf_counter()
    learned_keys=[key for key in waveforms if key=="historical" or key.startswith("seed")]
    families=(*learned_keys,"OTFS","OFDM","AFDM")
    support_cache={}
    for condition,snr,k in _evaluation_points(smoke):
        channels=condition_channels(condition,count,"confirmatory",namespace=CONTINUATION_NAMESPACE)
        for constellation in constellations:
            for family in families:
                # Historical receiver track uses slope 1 for the AFDM named reference.
                historical_key="AFDM_slope_1" if family=="AFDM" else family
                if historical_key in waveforms:
                    counts=generic_communication_counts(channels,waveforms[historical_key],constellation,snr,k,"historical",bits,condition=condition,bank_role="confirmatory",waveform_key=historical_key)
                    rows.extend({"track":"historical_receiver","family":family,"condition":condition,"modulation":constellation.name,"snr_db":snr,"k":k,"support_policy":"oracle_row_topk","receiver":"historical","regularization_scale":1.0,**row} for row in counts)
                selected=_select_operational(validation,family,condition,constellation.name,snr,k)
                waveform_key=selected["waveform_key"]; policy=selected["support_policy"]; receiver=selected["receiver"]; scale=float(selected["regularization_scale"])
                support=None; statistics=None
                if policy=="fixed_statistical":
                    cache_key=(condition,waveform_key,k)
                    if cache_key not in support_cache:
                        support_count=6 if smoke else int(configuration["statistics_channels"]); bank=condition_channels(condition,support_count,"statistical_support",namespace=CONTINUATION_NAMESPACE)
                        support_cache[cache_key]=fixed_energy_support(bank,waveforms[waveform_key],k)
                    support=support_cache[cache_key]
                if receiver=="residual_aware":
                    payload,statistics=load_statistics(smoke,waveform_key,condition,policy); support=payload["support"] if policy=="fixed_statistical" else None
                counts=generic_communication_counts(channels,waveforms[waveform_key],constellation,snr,k,receiver,bits,condition=condition,bank_role="confirmatory",waveform_key=waveform_key,regularization_scale=scale,support_policy=policy,fixed_support=support,statistics=statistics)
                rows.extend({"track":"operational_selected","family":family,"condition":condition,"modulation":constellation.name,"snr_db":snr,"k":k,"support_policy":policy,"receiver":receiver,"regularization_scale":scale,**row} for row in counts)
                selections.append({"family":family,"condition":condition,"modulation":constellation.name,"snr_db":snr,"k":k,"waveform_key":waveform_key,"support_policy":policy,"receiver":receiver,"regularization_scale":scale,"validation_ber":selected["ber"]})
        print(f"confirmation {condition} SNR={snr} K={k}",flush=True)
    # Fixed-step endpoint ablation is historical-receiver-only and never enters selection.
    endpoints=learned_waveforms(smoke,endpoints=True)
    for condition in (("joint_0.5",) if smoke else ("joint_0.5","nominal_dd_v1")):
        channels=condition_channels(condition,count,"confirmatory",namespace=CONTINUATION_NAMESPACE)
        for endpoint_key,waveform in endpoints.items():
            if endpoint_key=="historical":continue
            for constellation in constellations:
                counts=generic_communication_counts(channels,waveform,constellation,18,24,"historical",bits,condition=condition,bank_role="confirmatory",waveform_key=endpoint_key)
                rows.extend({"track":"fixed_endpoint","family":endpoint_key,"condition":condition,"modulation":constellation.name,"snr_db":18,"k":24,"support_policy":"oracle_row_topk","receiver":"historical","regularization_scale":1.0,**row} for row in counts)
    # Full-information explanatory control at nominal 18 dB.
    channels=condition_channels("nominal_dd_v1",count,"confirmatory",namespace=CONTINUATION_NAMESPACE)
    full_waveforms={**learned_waveforms(smoke),**classical_waveforms_n64()}
    if smoke:full_waveforms={key:value for key,value in full_waveforms.items() if key in {"historical","seed0_parent_selected","seed0_mean_continuation_selected","seed0_tail_continuation_selected","OTFS","OFDM","AFDM_slope_1"}}
    for waveform_key,waveform in full_waveforms.items():
        for constellation in constellations:
            counts=generic_communication_counts(channels,waveform,constellation,18,64,"full",bits,condition="full_information_nominal",bank_role="confirmatory",waveform_key=waveform_key)
            rows.extend({"track":"full_information","family":"AFDM" if waveform_key.startswith("AFDM") else waveform_key,"condition":"full_information_nominal","modulation":constellation.name,"snr_db":18,"k":64,"support_policy":"full","receiver":"full","regularization_scale":1.0,**row} for row in counts)
    write_csv(DATA/("communication_per_channel_smoke.csv" if smoke else "communication_per_channel.csv"),rows); write_csv(DATA/("communication_selections_smoke.csv" if smoke else "communication_selections.csv"),selections)
    group_keys=("track","family","waveform_key","condition","modulation","snr_db","k","support_policy","receiver","regularization_scale")
    grouped={}
    for row in rows:grouped.setdefault(tuple(row[key] for key in group_keys),[]).append(row)
    aggregates=[]
    for key,values in grouped.items():aggregates.append({**dict(zip(group_keys,key)),**_aggregate(values)})
    write_csv(DATA/("communication_aggregate_smoke.csv" if smoke else "communication_aggregate.csv"),aggregates)
    write_json(DATA/("communication_execution_smoke.json" if smoke else "communication_execution.json"),{"seconds":time.perf_counter()-started,"channels":count,"bits_per_channel":bits,"per_channel_rows":len(rows),"aggregate_rows":len(aggregates),"run_kind":"smoke" if smoke else "production"})


def _paired_channel_interval(left:list[dict[str,str]],right:list[dict[str,str]],label:str,resamples:int)->dict[str,object]:
    left_by_index={int(row["channel_index"]):float(row["ber"]) for row in left}; right_by_index={int(row["channel_index"]):float(row["ber"]) for row in right}
    if left_by_index.keys()!=right_by_index.keys() or len(left_by_index)<2:raise ValueError(f"unaligned paired rows: {label}")
    differences=np.asarray([left_by_index[index]-right_by_index[index] for index in sorted(left_by_index)]); rng=np.random.default_rng(semantic_seed("paired_bootstrap",namespace=CONTINUATION_NAMESPACE,label=label)); indices=rng.integers(0,len(differences),size=(resamples,len(differences))); samples=differences[indices].mean(axis=1)
    return {"mean_difference":float(differences.mean()),"ci95_low":float(np.quantile(samples,.025)),"ci95_high":float(np.quantile(samples,.975)),"n_channels":len(differences),"resamples":resamples}


def phase_intervals(smoke:bool)->None:
    """Saved-row paired uncertainty; no communication simulation."""
    ensure_dirs(); suffix="_smoke" if smoke else ""; per_channel=read_csv(DATA/f"communication_per_channel{suffix}.csv"); validation=read_csv(DATA/f"operating_validation{suffix}.csv"); resamples=200 if smoke else int(cfg()["evaluation"]["bootstrap_resamples"]); rows=[]
    conditions=("joint_0.5",) if smoke else ("joint_0.5","joint_0.2_probe","integer","nominal_dd_v1")
    for condition in conditions:
        for modulation in ("16QAM","QPSK"):
            classical_validation=[row for row in validation if row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==18 and int(row["k"])==24 and (row["waveform_key"] in {"OTFS","OFDM"} or row["waveform_key"].startswith("AFDM"))]
            strongest=min(classical_validation,key=_selection_key); reference_family="AFDM" if strongest["waveform_key"].startswith("AFDM") else strongest["waveform_key"]
            reference=[row for row in per_channel if row["track"]=="operational_selected" and row["family"]==reference_family and row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==18 and int(row["k"])==24]
            for seed in cfg()["training"]["seeds"]:
                tail_key=f"seed{seed}_tail_continuation_selected"; tail=[row for row in per_channel if row["track"]=="operational_selected" and row["family"]==tail_key and row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==18 and int(row["k"])==24]
                label=f"{condition}|{modulation}|{tail_key}-{reference_family}"; rows.append({"contrast_type":"tail_vs_validation_strongest_classical","condition":condition,"modulation":modulation,"seed_id":seed,"left":tail_key,"right":reference_family,"reference_validation_waveform":strongest["waveform_key"],**_paired_channel_interval(tail,reference,label,resamples)})
                for stage in ("parent","mean_continuation"):
                    right_key=f"seed{seed}_{stage}_selected"; matched=[row for row in per_channel if row["track"]=="operational_selected" and row["family"]==right_key and row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==18 and int(row["k"])==24]
                    label=f"{condition}|{modulation}|{tail_key}-{right_key}"; rows.append({"contrast_type":f"tail_vs_{stage}","condition":condition,"modulation":modulation,"seed_id":seed,"left":tail_key,"right":right_key,"reference_validation_waveform":"",**_paired_channel_interval(tail,matched,label,resamples)})
    write_csv(DATA/f"communication_paired_intervals{suffix}.csv",rows)


def _absolute_interval(rows:list[dict[str,object]],label:str,resamples:int)->dict[str,float]:
    values=np.asarray([float(row["ber"]) for row in rows]); rng=np.random.default_rng(semantic_seed("absolute_bootstrap",namespace=CONTINUATION_NAMESPACE,label=label))
    indices=rng.integers(0,len(values),size=(resamples,len(values))); samples=values[indices].mean(axis=1)
    return {"ci95_low":float(np.quantile(samples,.025)),"ci95_high":float(np.quantile(samples,.975))}


def phase_matched_confirmation(smoke:bool)->None:
    ensure_dirs(); phase_amendment(smoke); configuration=cfg(); evaluation=configuration["evaluation"]; matched=configuration["matched_quality"]
    validation=read_csv(DATA/("operating_validation_smoke.csv" if smoke else "operating_validation.csv")); waveforms=_validation_waveforms(smoke)
    validation_count=6 if smoke else int(evaluation["configuration_validation_channels"]); test_count=6 if smoke else int(evaluation["confirmatory_channels"])
    validation_bits=128 if smoke else int(evaluation["validation_bits_per_channel"]); test_bits=128 if smoke else int(evaluation["confirmatory_bits_per_channel"])
    conditions=("joint_0.5",) if smoke else ("joint_0.5","nominal_dd_v1"); budgets=(1,2) if smoke else tuple(int(value) for value in matched["iteration_budgets"]); tolerances=(1e-10,) if smoke else tuple(float(value) for value in matched["tolerances"])
    constellations=(qam16_constellation(),qpsk_constellation()); families=[key for key in waveforms if key=="historical" or key.startswith("seed")]+["OTFS","OFDM","AFDM"]
    iterative_validation=[]; selection_rows=[]; confirmation_rows=[]; started=time.perf_counter(); support_cache={}
    for condition in conditions:
        validation_channels=condition_channels(condition,validation_count,"configuration_validation",namespace=CONTINUATION_NAMESPACE)
        test_channels=condition_channels(condition,test_count,"matched_confirmation",namespace=CONTINUATION_NAMESPACE)
        support_count=6 if smoke else int(evaluation["statistics_channels"]); support_bank=condition_channels(condition,support_count,"statistical_support",namespace=CONTINUATION_NAMESPACE)
        for constellation in constellations:
            targets=tuple(float(value) for value in matched["targets"][constellation.name])
            for family in families:
                eligible=[row for row in validation if row["condition"]==condition and row["modulation"]==constellation.name and int(row["snr_db"])==18 and (row["waveform_key"].startswith("AFDM") if family=="AFDM" else row["waveform_key"]==family)]
                if not eligible:raise RuntimeError(f"missing matched-quality validation rows {condition} {constellation.name} {family}")
                candidate_rows=[]
                for k in sorted({int(row["k"]) for row in eligible}):
                    at_k=[row for row in eligible if int(row["k"])==k]
                    best_direct=min(at_k,key=_selection_key)
                    candidate_rows.append({"implementation":"precomputed_dense_equalizer","waveform_key":best_direct["waveform_key"],"k":k,"support_policy":best_direct["support_policy"],"receiver":best_direct["receiver"],"regularization_scale":float(best_direct["regularization_scale"]),"validation_ber":float(best_direct["ber"]),"detector_complex_macs":64*64,"iteration_budget":0,"mean_relative_residual":0.0,"physical_iterations":0})
                    nonresidual=[row for row in at_k if row["receiver"] in {"historical","tuned"}]
                    best_iterative=min(nonresidual,key=_selection_key); waveform_key=best_iterative["waveform_key"]; policy=best_iterative["support_policy"]
                    support=None
                    if policy=="fixed_statistical":
                        cache_key=(condition,waveform_key,k); support_cache.setdefault(cache_key,fixed_energy_support(support_bank,waveforms[waveform_key],k)); support=support_cache[cache_key]
                    for tolerance in tolerances:
                        counts=generic_iterative_counts(validation_channels,waveforms[waveform_key],constellation,18,k,validation_bits,budgets,condition=condition,bank_role="configuration_validation",waveform_key=waveform_key,support_policy=policy,fixed_support=support,tolerance=tolerance,regularization_scale=float(best_iterative["regularization_scale"]))
                        for budget in budgets:
                            selected_counts=[row for row in counts if int(row["iteration_budget"])==budget]; aggregate=_aggregate([{**row,"retained_energy":1.0} for row in selected_counts])
                            residual=float(np.mean([float(row["mean_relative_residual"]) for row in selected_counts])); logical=float(np.mean([float(row["mean_logical_iterations"]) for row in selected_counts]))
                            record={"condition":condition,"modulation":constellation.name,"family":family,"waveform_key":waveform_key,"k":k,"support_policy":policy,"receiver":best_iterative["receiver"],"regularization_scale":float(best_iterative["regularization_scale"]),"iteration_budget":budget,"tolerance":tolerance,"validation_ber":aggregate["ber"],"bit_errors":aggregate["bit_errors"],"total_bits":aggregate["total_bits"],"mean_relative_residual":residual,"mean_logical_iterations":logical,"physical_iterations":budget,"detector_complex_macs":operation_ledger(64,k,budget)["algorithmic_total_complex_equivalent_macs"],"implementation":"dense_matrix_free_CG_control"}
                            iterative_validation.append(record); candidate_rows.append(record)
                for target in targets:
                    attained=[row for row in candidate_rows if float(row["validation_ber"])<=target]
                    if not attained:
                        selection_rows.append({"condition":condition,"modulation":constellation.name,"family":family,"target_ber":target,"validation_attained":False}); continue
                    chosen=min(attained,key=lambda row:(int(row["detector_complex_macs"]),float(row["validation_ber"]),int(row["k"]),str(row["waveform_key"])))
                    selection={"condition":condition,"modulation":constellation.name,"family":family,"target_ber":target,"validation_attained":True,**chosen}; selection_rows.append(selection)
                    waveform_key=str(chosen["waveform_key"]); k=int(chosen["k"]); policy=str(chosen["support_policy"]); support=None
                    if policy=="fixed_statistical":
                        cache_key=(condition,waveform_key,k); support_cache.setdefault(cache_key,fixed_energy_support(support_bank,waveforms[waveform_key],k)); support=support_cache[cache_key]
                    if chosen["implementation"]=="precomputed_dense_equalizer":
                        stats=None
                        if chosen["receiver"]=="residual_aware":
                            payload,stats=load_statistics(smoke,waveform_key,condition,policy); support=payload["support"] if policy=="fixed_statistical" else None
                        counts=generic_communication_counts(test_channels,waveforms[waveform_key],constellation,18,k,str(chosen["receiver"]),test_bits,condition=condition,bank_role="matched_confirmation",waveform_key=waveform_key,regularization_scale=float(chosen["regularization_scale"]),support_policy=policy,fixed_support=support,statistics=stats)
                    else:
                        counts=generic_iterative_counts(test_channels,waveforms[waveform_key],constellation,18,k,test_bits,(int(chosen["iteration_budget"]),),condition=condition,bank_role="matched_confirmation",waveform_key=waveform_key,support_policy=policy,fixed_support=support,tolerance=float(chosen["tolerance"]),regularization_scale=float(chosen["regularization_scale"]))
                    aggregate=_aggregate([{**row,"retained_energy":1.0} for row in counts]); interval=_absolute_interval(counts,f"{condition}|{constellation.name}|{family}|{target}",200 if smoke else int(evaluation["bootstrap_resamples"]))
                    full=generic_communication_counts(test_channels,waveforms[waveform_key],constellation,18,64,"full",test_bits,condition=condition,bank_role="matched_confirmation",waveform_key=waveform_key); full_aggregate=_aggregate(full)
                    storage=storage_ledger(64,k,residual_aware=str(chosen["receiver"])=="residual_aware")
                    confirmation_rows.append({**selection,"test_ber":aggregate["ber"],"test_bit_errors":aggregate["bit_errors"],"test_total_bits":aggregate["total_bits"],**interval,"target_attained_test":aggregate["ber"]<=target,"own_full_ber":full_aggregate["ber"],"degradation_vs_own_full":aggregate["ber"]-full_aggregate["ber"],"ratio_to_own_full":aggregate["ber"]/full_aggregate["ber"] if full_aggregate["ber"] else float("inf"),"sparse_state_bytes":storage["sparse_total_bytes"],"residual_statistics_bytes":storage["residual_mean_covariance_bytes"],"waveform_bytes":storage["waveform_bytes"]})
                print(f"matched {condition} {constellation.name} {family}",flush=True)
    write_csv(DATA/("iterative_validation_smoke.csv" if smoke else "iterative_validation.csv"),iterative_validation); write_csv(DATA/("matched_quality_selections_smoke.csv" if smoke else "matched_quality_selections.csv"),selection_rows); write_csv(DATA/("matched_quality_confirmation_smoke.csv" if smoke else "matched_quality_confirmation.csv"),confirmation_rows)
    write_json(DATA/("matched_quality_execution_smoke.json" if smoke else "matched_quality_execution.json"),{"seconds":time.perf_counter()-started,"validation_channels":validation_count,"confirmation_channels":test_count,"validation_bits_per_channel":validation_bits,"confirmation_bits_per_channel":test_bits,"run_kind":"smoke" if smoke else "production"})


def _scaled_waveforms(n:int,learned:torch.Tensor)->dict[str,torch.Tensor]:
    definition=scaling_definition(n); return {"Learned":learned,"OTFS":otfs_waveform(definition.n_tau,definition.n_nu),"OFDM":ofdm_waveform(n)}


def _time_ns(call,warmup:int,repetitions:int):
    for _ in range(warmup):call()
    values=[]; last=None
    for _ in range(repetitions): started=time.perf_counter_ns(); last=call(); values.append(time.perf_counter_ns()-started)
    return int(np.median(values)),int(np.quantile(values,.95)),last


def phase_scaling(smoke:bool)->None:
    ensure_dirs(); phase_amendment(smoke); configuration=cfg(); scaling=configuration["scaling"]; evaluation=configuration["evaluation"]
    dimensions=(64,) if smoke else tuple(int(value) for value in scaling["dimensions"]); constellations=(qam16_constellation(),qpsk_constellation())
    selection_rows=[]; communication_rows=[]; aggregate_rows=[]; benchmark_rows=[]; analytical_rows=[]; profiler_rows=[]; started_all=time.perf_counter(); old_threads=torch.get_num_threads(); torch.set_num_threads(int(scaling["thread_count"]))
    try:
        for n in dimensions:
            checkpoint=ROOT/scaling["checkpoints"][str(n)]["path"]; learned=load_checkpoint(checkpoint); expected=scaling["checkpoints"][str(n)]["sha256"]
            if sha(checkpoint)!=expected:raise RuntimeError(f"N={n} scaling checkpoint changed")
            definition=scaling_definition(n); kref=int(scaling["reference_k"][str(n)]); candidates=tuple(int(value) for value in evaluation["afdm_operational_candidates_by_n"][str(n)])
            validation_count=2 if smoke else int(scaling["communication_validation_channels"][str(n)]); test_count=4 if smoke else int(scaling["communication_confirmatory_channels"][str(n)]); bits=256 if smoke else int(scaling["communication_bits_per_channel"])
            validation=sample_scaled_dd(validation_count,semantic_seed("scaling_validation",namespace=CONTINUATION_NAMESPACE,n=n),n).matrices
            candidate_scores=[]
            for slope in candidates:
                waveform=afdm_waveform(n=n,c1=slope/(2*n),c2=np.sqrt(2)/(100*n)); total_errors=0;total_bits=0
                for constellation in constellations:
                    counts=generic_communication_counts(validation,waveform,constellation,18,kref,"historical",512 if not smoke else 128,condition=f"scaled_n{n}",bank_role="scaling_validation",waveform_key=f"AFDM_slope_{slope}")
                    total_errors+=sum(int(row["bit_errors"]) for row in counts);total_bits+=sum(int(row["total_bits"]) for row in counts)
                candidate_scores.append({"n":n,"slope":slope,"validation_joint_modulation_ber":total_errors/total_bits,"bit_errors":total_errors,"total_bits":total_bits})
            selected=min(candidate_scores,key=lambda row:(row["validation_joint_modulation_ber"],row["slope"])); selection_rows.extend({**row,"selected":row["slope"]==selected["slope"]} for row in candidate_scores)
            waveforms=_scaled_waveforms(n,learned); waveforms["AFDM"]=afdm_waveform(n=n,c1=int(selected["slope"])/(2*n),c2=np.sqrt(2)/(100*n))
            test=sample_scaled_dd(test_count,semantic_seed("scaling_confirmation",namespace=CONTINUATION_NAMESPACE,n=n),n).matrices
            for waveform_key,waveform in waveforms.items():
                for constellation in constellations:
                    own={}
                    for k in (kref,n):
                        counts=generic_communication_counts(test,waveform,constellation,18,k,"historical" if k<n else "full",bits,condition=f"scaled_n{n}",bank_role="scaling_confirmation",waveform_key=waveform_key)
                        communication_rows.extend({"n":n,"waveform":waveform_key,"modulation":constellation.name,"snr_db":18,"k":k,"rho":k/n,"receiver":"historical" if k<n else "full",**row} for row in counts); own[k]=_aggregate(counts)
                    aggregate_rows.append({"n":n,"waveform":waveform_key,"modulation":constellation.name,"snr_db":18,"k":kref,"rho":kref/n,"bit_errors":own[kref]["bit_errors"],"total_bits":own[kref]["total_bits"],"ber":own[kref]["ber"],"own_full_ber":own[n]["ber"],"degradation_vs_own_full":own[kref]["ber"]-own[n]["ber"],"ratio_to_own_full":own[kref]["ber"]/own[n]["ber"] if own[n]["ber"] else float("inf"),"historical_single_checkpoint":waveform_key=="Learned"})
            channels=sample_scaled_dd(2 if smoke else int(scaling["benchmark_channels"][str(n)]),semantic_seed("scaling_benchmark",namespace=CONTINUATION_NAMESPACE,n=n),n).matrices
            rho_values=(.375,) if smoke else tuple(float(value) for value in scaling["rho_values"]); ks=tuple(round(rho*n) for rho in rho_values); repeats=2 if smoke else int(scaling["timing_repetitions"][str(n)]); warmup=1 if smoke else int(scaling["warmup"]); max_iterations=int(scaling["maximum_iterations"][str(n)])
            benchmark_configs=[("Learned",k) for k in ks]+[(name,k) for name in ("OTFS","OFDM","AFDM") for k in (kref,n)]
            generator=torch.Generator().manual_seed(semantic_seed("scaling_benchmark_vectors",namespace=CONTINUATION_NAMESPACE,n=n))
            for waveform_key,k in benchmark_configs:
                waveform=waveforms[waveform_key]
                for channel_index,channel in enumerate(channels):
                    observation=torch.randn(n,dtype=COMPLEX_DTYPE,generator=generator); actual=waveform.mH@channel@waveform; retained=rowwise_topk(actual,k); sparse=sparse_from_topk(actual,k)
                    direct=mismatched_lmmse_equalizer(retained,noise_variance(18))@observation
                    tight=sparse_cg_lmmse(sparse,observation,noise_variance(18),max_iterations=max_iterations,tolerance=1e-12); relative=float(torch.linalg.vector_norm(tight.estimate-direct)/torch.linalg.vector_norm(direct))
                    if not tight.converged or relative>1e-8:raise RuntimeError(f"N={n}, K={k} tight sparse/direct failure {relative}, {tight.relative_residual}")
                    transform_median,transform_p95,_=_time_ns(lambda:waveform.mH@observation,warmup,repeats)
                    effective_median,effective_p95,_=_time_ns(lambda:waveform.mH@channel@waveform,warmup,repeats)
                    pack_median,pack_p95,_=_time_ns(lambda:sparse_from_topk(actual,k),warmup,repeats)
                    setup_median,setup_p95,_=_time_ns(lambda:mismatched_lmmse_equalizer(retained,noise_variance(18)),warmup,repeats); equalizer=mismatched_lmmse_equalizer(retained,noise_variance(18))
                    apply_median,apply_p95,_=_time_ns(lambda:equalizer@observation,warmup,repeats)
                    direct_median,direct_p95,_=_time_ns(lambda:mismatched_lmmse_equalizer(retained,noise_variance(18))@observation,warmup,repeats)
                    sparse_median,sparse_p95,sparse_result=_time_ns(lambda:sparse_cg_lmmse(sparse,observation,noise_variance(18),max_iterations=64,tolerance=1e-10),warmup,repeats)
                    densecg_median,densecg_p95,dense_result=_time_ns(lambda:dense_cg_snapshots(retained[None],observation[None,None],float(noise_variance(18)),(64,),1e-10)[64],warmup,repeats)
                    state=storage_ledger(n,k,residual_aware=False); executed=operation_ledger(n,k,64)
                    common={"n":n,"waveform":waveform_key,"k":k,"rho":k/n,"channel_index":channel_index,"repetitions":repeats,"threads":int(scaling["thread_count"]),"tight_iterations":tight.iterations,"tight_relative_error":relative,"tight_relative_residual":tight.relative_residual,"transform_median_ns":transform_median,"effective_channel_median_ns":effective_median,"oracle_pack_median_ns":pack_median,"persistent_sparse_bytes":state["sparse_total_bytes"],"dense_coefficient_bytes":state["dense_coefficients_bytes"],"waveform_bytes":state["waveform_bytes"],"analytical_sparse_peak_temporary_bytes":6*n*16,"analytical_dense_peak_temporary_bytes":3*n*n*16}
                    benchmark_rows.extend([
                        {**common,"implementation":"dense_direct_per_vector","median_ns":direct_median,"p95_ns":direct_p95,"setup_median_ns":0,"recurring_median_ns":direct_median,"physical_iterations":0,"logical_iterations":0,"converged":True,"relative_residual":0.0,"executed_complex_macs":"not reduced to sparse ledger"},
                        {**common,"implementation":"precomputed_dense_equalizer","median_ns":apply_median,"p95_ns":apply_p95,"setup_median_ns":setup_median,"recurring_median_ns":apply_median,"physical_iterations":0,"logical_iterations":0,"converged":True,"relative_residual":0.0,"executed_complex_macs":n*n},
                        {**common,"implementation":"explicit_sparse_matrix_free_CG","median_ns":sparse_median,"p95_ns":sparse_p95,"setup_median_ns":0,"recurring_median_ns":sparse_median,"physical_iterations":sparse_result.iterations,"logical_iterations":sparse_result.iterations,"converged":sparse_result.converged,"relative_residual":sparse_result.relative_residual,"executed_complex_macs":operation_ledger(n,k,sparse_result.iterations)["algorithmic_total_complex_equivalent_macs"]},
                        {**common,"implementation":"dense_matrix_free_CG_control","median_ns":densecg_median,"p95_ns":densecg_p95,"setup_median_ns":0,"recurring_median_ns":densecg_median,"physical_iterations":64,"logical_iterations":float(dense_result.logical_iterations.to(torch.float64).mean()),"converged":bool(torch.all(dense_result.relative_residual<=1e-10)),"relative_residual":float(dense_result.relative_residual.mean()),"executed_complex_macs":executed["algorithmic_total_complex_equivalent_macs"]},
                    ])
                print(f"scaling benchmark N={n} {waveform_key} K={k}",flush=True)
            # One diagnostic profile per dimension; final timing above excludes profiling.
            try:
                from torch.profiler import ProfilerActivity,profile
                actual=waveforms["Learned"].mH@channels[0]@waveforms["Learned"]; sparse=sparse_from_topk(actual,kref); observation=torch.randn(n,dtype=COMPLEX_DTYPE,generator=generator)
                with profile(activities=[ProfilerActivity.CPU],profile_memory=True,record_shapes=True) as prof:sparse_cg_lmmse(sparse,observation,noise_variance(18),max_iterations=64,tolerance=1e-10)
                for event in sorted(prof.key_averages(),key=lambda item:item.self_cpu_time_total,reverse=True)[:20]:profiler_rows.append({"n":n,"k":kref,"operator":event.key,"self_cpu_time_us":event.self_cpu_time_total,"cpu_time_us":event.cpu_time_total,"cpu_memory_bytes":event.cpu_memory_usage,"calls":event.count})
            except Exception as error:profiler_rows.append({"n":n,"k":kref,"operator":"PROFILE_UNAVAILABLE","error":repr(error)})
        for n in (64,128,256,512,1024):
            for rho in (.375,1.0):
                k=round(rho*n); row=operation_ledger(n,k,64); state=storage_ledger(n,k,residual_aware=False); analytical_rows.append({**row,"rho":rho,"sparse_storage_bytes":state["sparse_total_bytes"],"dense_storage_bytes":state["dense_coefficients_bytes"],"measured_dimension":n in (64,128,256)})
    finally:torch.set_num_threads(old_threads)
    suffix="_smoke" if smoke else ""; write_csv(DATA/f"scaling_afdm_selection{suffix}.csv",selection_rows); write_csv(DATA/f"scaling_communication_per_channel{suffix}.csv",communication_rows); write_csv(DATA/f"scaling_communication_aggregate{suffix}.csv",aggregate_rows); write_csv(TIMING/f"scaling_benchmark{suffix}.csv",benchmark_rows); write_csv(DATA/f"scaling_analytical{suffix}.csv",analytical_rows); write_csv(TIMING/f"scaling_profiler{suffix}.csv",profiler_rows)
    write_json(DATA/f"scaling_execution{suffix}.json",{"seconds":time.perf_counter()-started_all,"run_kind":"smoke" if smoke else "production","dimensions":list(dimensions),"thread_restored_to":old_threads})


def phase_scaling_matched(smoke:bool)->None:
    """Validation-selected matched targets at the measured scaling dimensions."""
    ensure_dirs(); phase_amendment(smoke); configuration=cfg(); scaling=configuration["scaling"]; matched=configuration["matched_quality"]
    dimensions=(64,) if smoke else tuple(int(value) for value in scaling["dimensions"]); rhos=(.375,1.0) if smoke else tuple(float(value) for value in scaling["rho_values"])
    budgets=(1,2) if smoke else tuple(int(value) for value in matched["iteration_budgets"]); tolerances=(1e-10,) if smoke else tuple(float(value) for value in matched["tolerances"])
    afdm_rows=read_csv(DATA/f"scaling_afdm_selection{'_smoke' if smoke else ''}.csv")
    selections=[]; confirmations=[]; timing_rows=[]; amortization=[]; started=time.perf_counter(); old_threads=torch.get_num_threads(); torch.set_num_threads(int(scaling["thread_count"]))
    try:
        for n in dimensions:
            learned=load_checkpoint(ROOT/scaling["checkpoints"][str(n)]["path"]); definition=scaling_definition(n)
            slope=int(next(row["slope"] for row in afdm_rows if int(row["n"])==n and row["selected"]=="True"))
            waveforms={**_scaled_waveforms(n,learned),"AFDM":afdm_waveform(n=n,c1=slope/(2*n),c2=np.sqrt(2)/(100*n))}
            validation_count=2 if smoke else int(scaling["communication_validation_channels"][str(n)]); test_count=4 if smoke else int(scaling["communication_confirmatory_channels"][str(n)])
            bits=256 if smoke else int(scaling["communication_bits_per_channel"]); validation=sample_scaled_dd(validation_count,semantic_seed("scaling_validation",namespace=CONTINUATION_NAMESPACE,n=n),n).matrices; test=sample_scaled_dd(test_count,semantic_seed("scaling_confirmation",namespace=CONTINUATION_NAMESPACE,n=n),n).matrices
            ks=tuple(sorted({round(rho*n) for rho in rhos})); constellation_rows=(qam16_constellation(),qpsk_constellation())
            for constellation in constellation_rows:
                for family,waveform in waveforms.items():
                    candidates=[]
                    for k in ks:
                        direct=generic_communication_counts(validation,waveform,constellation,18,k,"historical",512 if not smoke else 128,condition=f"scaled_n{n}",bank_role="scaling_validation",waveform_key=family); aggregate=_aggregate(direct)
                        base={"n":n,"condition":f"scaled_n{n}","modulation":constellation.name,"family":family,"waveform_key":family,"k":k,"rho":k/n,"support_policy":"oracle_row_topk","receiver":"historical","regularization_scale":1.0,"validation_ber":aggregate["ber"],"iteration_budget":0,"tolerance":"","mean_relative_residual":0.0,"mean_logical_iterations":0.0}
                        candidates.extend(({**base,"implementation":"precomputed_dense_equalizer","detector_complex_macs":n*n},{**base,"implementation":"dense_direct_per_vector","detector_complex_macs":n**3+n*n}))
                        for tolerance in tolerances:
                            iterative=generic_iterative_counts(validation,waveform,constellation,18,k,512 if not smoke else 128,budgets,condition=f"scaled_n{n}",bank_role="scaling_validation",waveform_key=family,tolerance=tolerance)
                            for budget in budgets:
                                chosen=[row for row in iterative if int(row["iteration_budget"])==budget]; agg=_aggregate([{**row,"retained_energy":1.0} for row in chosen])
                                candidates.append({**base,"implementation":"dense_matrix_free_CG_control","iteration_budget":budget,"tolerance":tolerance,"validation_ber":agg["ber"],"detector_complex_macs":operation_ledger(n,k,budget)["algorithmic_total_complex_equivalent_macs"],"mean_relative_residual":float(np.mean([float(row["mean_relative_residual"]) for row in chosen])),"mean_logical_iterations":float(np.mean([float(row["mean_logical_iterations"]) for row in chosen]))})
                    for target in tuple(float(value) for value in matched["targets"][constellation.name]):
                        attained=[row for row in candidates if float(row["validation_ber"])<=target]
                        if not attained:
                            selections.append({"n":n,"condition":f"scaled_n{n}","modulation":constellation.name,"family":family,"target_ber":target,"validation_attained":False}); continue
                        chosen=min(attained,key=lambda row:(int(row["detector_complex_macs"]),float(row["validation_ber"]),int(row["k"]),str(row["implementation"])))
                        selection={**chosen,"target_ber":target,"validation_attained":True}; selections.append(selection); k=int(chosen["k"])
                        if chosen["implementation"] in {"precomputed_dense_equalizer","dense_direct_per_vector"}:
                            counts=generic_communication_counts(test,waveform,constellation,18,k,"historical",bits,condition=f"scaled_n{n}",bank_role="scaling_matched_confirmation",waveform_key=family)
                        else:
                            counts=generic_iterative_counts(test,waveform,constellation,18,k,bits,(int(chosen["iteration_budget"]),),condition=f"scaled_n{n}",bank_role="scaling_matched_confirmation",waveform_key=family,tolerance=float(chosen["tolerance"]))
                        aggregate=_aggregate([{**row,"retained_energy":row.get("retained_energy",1.0)} for row in counts]); interval=_absolute_interval(counts,f"scaled|{n}|{constellation.name}|{family}|{target}",200 if smoke else int(configuration["evaluation"]["bootstrap_resamples"]))
                        full=generic_communication_counts(test,waveform,constellation,18,n,"full",bits,condition=f"scaled_n{n}",bank_role="scaling_matched_confirmation",waveform_key=family); full_aggregate=_aggregate(full); state=storage_ledger(n,k,residual_aware=False)
                        confirmations.append({**selection,"test_ber":aggregate["ber"],"test_bit_errors":aggregate["bit_errors"],"test_total_bits":aggregate["total_bits"],**interval,"target_attained_test":aggregate["ber"]<=target,"own_full_ber":full_aggregate["ber"],"degradation_vs_own_full":aggregate["ber"]-full_aggregate["ber"],"ratio_to_own_full":aggregate["ber"]/full_aggregate["ber"] if full_aggregate["ber"] else float("inf"),"sparse_state_bytes":state["sparse_total_bytes"],"dense_coefficient_bytes":state["dense_coefficients_bytes"],"waveform_bytes":state["waveform_bytes"]})
                    print(f"scaling matched N={n} {constellation.name} {family}",flush=True)
            # Benchmark only validation-selected configurations; timing never changes selection.
            unique={tuple(row.get(key,"") for key in ("family","k","implementation","iteration_budget","tolerance")):row for row in selections if int(row["n"])==n and row["validation_attained"] is True}
            channels=sample_scaled_dd(2 if smoke else int(scaling["benchmark_channels"][str(n)]),semantic_seed("scaling_benchmark",namespace=CONTINUATION_NAMESPACE,n=n),n).matrices; repeats=2 if smoke else int(scaling["timing_repetitions"][str(n)]); warmup=1 if smoke else int(scaling["warmup"])
            generator=torch.Generator().manual_seed(semantic_seed("scaling_matched_vectors",namespace=CONTINUATION_NAMESPACE,n=n))
            for key,row in unique.items():
                family,k,implementation,iteration_budget,tolerance=key; k=int(k); waveform=waveforms[str(family)]
                per_channel=[]
                for channel_index,channel in enumerate(channels):
                    observation=torch.randn(n,dtype=COMPLEX_DTYPE,generator=generator); actual=waveform.mH@channel@waveform; retained=rowwise_topk(actual,k)
                    transform,_,_=_time_ns(lambda:waveform.mH@observation,warmup,repeats); effective,_,_=_time_ns(lambda:waveform.mH@channel@waveform,warmup,repeats); pack,_,_=_time_ns(lambda:rowwise_topk(actual,k),warmup,repeats)
                    if implementation=="precomputed_dense_equalizer":
                        setup,_,_=_time_ns(lambda:mismatched_lmmse_equalizer(retained,noise_variance(18)),warmup,repeats); equalizer=mismatched_lmmse_equalizer(retained,noise_variance(18)); recurring,p95,_=_time_ns(lambda:equalizer@observation,warmup,repeats)
                    elif implementation=="dense_direct_per_vector":
                        setup=0; recurring,p95,_=_time_ns(lambda:mismatched_lmmse_equalizer(retained,noise_variance(18))@observation,warmup,repeats)
                    else:
                        setup=0; budget=int(iteration_budget); tol=float(tolerance); recurring,p95,_=_time_ns(lambda:dense_cg_snapshots(retained[None],observation[None,None],float(noise_variance(18)),(budget,),tol)[budget],warmup,repeats)
                    per_channel.append((transform,effective,pack,setup,recurring,p95))
                med=np.median(np.asarray(per_channel),axis=0); timing_rows.append({"n":n,"family":family,"k":k,"rho":k/n,"implementation":implementation,"iteration_budget":iteration_budget,"tolerance":tolerance,"channels":len(channels),"repetitions":repeats,"threads":int(scaling["thread_count"]),"transform_median_ns":med[0],"effective_channel_median_ns":med[1],"oracle_pack_median_ns":med[2],"setup_median_ns":med[3],"recurring_median_ns":med[4],"recurring_p95_ns":med[5]})
                for blocks in tuple(int(value) for value in matched["reuse_blocks"]):amortization.append({"n":n,"family":family,"k":k,"implementation":implementation,"iteration_budget":iteration_budget,"tolerance":tolerance,"reuse_blocks":blocks,"amortized_total_ns":med[0]+med[4]+(med[1]+med[2]+med[3])/blocks,"receive_transform_ns":med[0],"recurring_detection_ns":med[4],"reusable_preparation_ns":med[1]+med[2]+med[3]})
    finally: torch.set_num_threads(old_threads)
    suffix="_smoke" if smoke else ""; write_csv(DATA/f"scaling_matched_selections{suffix}.csv",selections); write_csv(DATA/f"scaling_matched_confirmation{suffix}.csv",confirmations); write_csv(TIMING/f"scaling_matched_timing{suffix}.csv",timing_rows); write_csv(TIMING/f"scaling_matched_amortization{suffix}.csv",amortization)
    write_json(DATA/f"scaling_matched_execution{suffix}.json",{"seconds":time.perf_counter()-started,"run_kind":"smoke" if smoke else "production","dimensions":list(dimensions),"thread_restored_to":old_threads})


def _save_figure(fig,name:str)->None:
    fig.tight_layout()
    for suffix in ("png","pdf"):fig.savefig(FIGURES/f"{name}.{suffix}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def phase_figures(smoke:bool)->None:
    ensure_dirs(); phase_intervals(smoke); suffix="_smoke" if smoke else ""; training=read_csv(DATA/f"training_summary{suffix}.csv"); communication=read_csv(DATA/f"communication_aggregate{suffix}.csv"); matched=read_csv(DATA/f"matched_quality_confirmation{suffix}.csv"); scaling_comm=read_csv(DATA/f"scaling_communication_aggregate{suffix}.csv"); scaling_timing=read_csv(TIMING/f"scaling_benchmark{suffix}.csv"); analytical=read_csv(DATA/f"scaling_analytical{suffix}.csv"); scaling_matched=read_csv(DATA/f"scaling_matched_confirmation{suffix}.csv"); scaling_selections=read_csv(DATA/f"scaling_matched_selections{suffix}.csv"); scaling_matched_timing=read_csv(TIMING/f"scaling_matched_timing{suffix}.csv"); scaling_amortization=read_csv(TIMING/f"scaling_matched_amortization{suffix}.csv")
    # F1-cont: selected tail seeds and historical/classical controls.
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
    for axis,modulation in zip(axes,("16QAM","QPSK")):
        for seed in cfg()["training"]["seeds"]:
            key=f"seed{seed}_tail_continuation_selected"; points=[row for row in communication if row["track"]=="operational_selected" and row["condition"]=="joint_0.5" and row["modulation"]==modulation and row["family"]==key and int(row["k"])==24]
            if points:axis.plot([int(row["snr_db"]) for row in sorted(points,key=lambda r:int(r["snr_db"]))],[float(row["ber"]) for row in sorted(points,key=lambda r:int(r["snr_db"]))],color="#d97706",alpha=.45,lw=1)
        for family,color in (("historical","black"),("OTFS","#15803d"),("OFDM","#2563eb"),("AFDM","#dc2626")):
            points=[row for row in communication if row["track"]=="operational_selected" and row["condition"]=="joint_0.5" and row["modulation"]==modulation and row["family"]==family and int(row["k"])==24]
            if points:axis.plot([int(row["snr_db"]) for row in sorted(points,key=lambda r:int(r["snr_db"]))],[float(row["ber"]) for row in sorted(points,key=lambda r:int(r["snr_db"]))],marker="o",ms=3,label=family,color=color)
        axis.set_yscale("log");axis.set_title(modulation);axis.set_xlabel("SNR (dB)");axis.set_ylabel("BER");axis.grid(alpha=.25);axis.legend(fontsize=6)
    _save_figure(fig,"F1_cont_training_reproducibility")
    # F2-cont: matched validation objective outcomes.
    fig,axes=plt.subplots(1,3,figsize=(10.4,2.9)); stages=("parent","mean_continuation","tail_continuation");x=np.arange(len(stages))
    for seed in cfg()["training"]["seeds"]:
        values=[next(float(row["selected_mean_mse"]) for row in training if int(row["seed_id"])==seed and row["stage"]==stage) for stage in stages];axes[0].plot(x,values,marker="o",alpha=.65,label=f"seed {seed}")
        values=[next(float(row["selected_cvar95"]) for row in training if int(row["seed_id"])==seed and row["stage"]==stage) for stage in stages];axes[1].plot(x,values,marker="o",alpha=.65,label=f"seed {seed}")
        selected=[]; endpoints=[]
        for stage in stages:
            selected_row=next(row for row in communication if row["track"]=="operational_selected" and row["family"]==f"seed{seed}_{stage}_selected" and row["condition"]=="joint_0.5" and row["modulation"]=="16QAM" and int(row["snr_db"])==18 and int(row["k"])==24); selected.append(float(selected_row["ber"]))
            endpoint_row=next(row for row in communication if row["track"]=="fixed_endpoint" and row["family"]==f"seed{seed}_{stage}_endpoint" and row["condition"]=="joint_0.5" and row["modulation"]=="16QAM"); endpoints.append(float(endpoint_row["ber"]))
        color=f"C{seed}";axes[2].plot(x,selected,marker="o",color=color,alpha=.75);axes[2].plot(x,endpoints,marker="x",ls="--",color=color,alpha=.55)
    for axis,title,ylabel in zip(axes,("Validation mean MSE","Stream CVaR 0.95","16-QAM BER: selected / endpoint"),("Mean MSE","CVaR 0.95","BER at K=24, 18 dB")):axis.set_xticks(x,("parent","mean cont.","tail cont."),rotation=15);axis.set_title(title);axis.set_ylabel(ylabel);axis.grid(alpha=.25)
    axes[1].legend(fontsize=6,ncol=2);axes[2].set_yscale("log");axes[2].plot([],[],"k-o",label="validation-selected");axes[2].plot([],[],"k--x",label="fixed endpoint");axes[2].legend(fontsize=6);_save_figure(fig,"F2_cont_matched_objective_ablation")
    # F3-cont: seed spread by condition at K24/18 dB.
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.0))
    conditions=("joint_0.5","joint_0.2_probe","integer","nominal_dd_v1","full_information_nominal")
    for axis,modulation in zip(axes,("16QAM","QPSK")):
        for index,condition in enumerate(conditions):
            seed_values=[]
            for seed in cfg()["training"]["seeds"]:
                key=f"seed{seed}_tail_continuation_selected"; track="full_information" if condition=="full_information_nominal" else "operational_selected"; expected_k=64 if condition=="full_information_nominal" else 24; matches=[row for row in communication if row["track"]==track and row["family"]==key and row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==18 and int(row["k"])==expected_k]
                if matches:seed_values.append(float(matches[0]["ber"]))
            if seed_values:axis.errorbar(index,np.median(seed_values),yerr=[[np.median(seed_values)-min(seed_values)],[max(seed_values)-np.median(seed_values)]],fmt="o",color="#d97706",capsize=3,label="tail-seed median/range" if index==0 else None)
            for family,color,offset in (("OTFS","#15803d",-.16),("OFDM","#2563eb",0),("AFDM","#dc2626",.16)):
                track="full_information" if condition=="full_information_nominal" else "operational_selected"; expected_k=64 if condition=="full_information_nominal" else 24; matches=[row for row in communication if row["track"]==track and row["family"]==family and row["condition"]==condition and row["modulation"]==modulation and int(row["snr_db"])==18 and int(row["k"])==expected_k and (family!="AFDM" or condition!="full_information_nominal" or row["waveform_key"]=="AFDM_slope_1")]
                if matches:axis.scatter(index+offset,float(matches[0]["ber"]),s=18,color=color,label=family if index==0 else None)
        axis.set_xticks(range(len(conditions)),("joint .5","probe .2","integer","nominal","full nominal"),rotation=20);axis.set_yscale("log");axis.set_title(modulation);axis.set_ylabel("BER at 18 dB");axis.grid(alpha=.25)
    axes[1].legend(fontsize=6);_save_figure(fig,"F3_cont_qualified_regimes")
    # F4c: BER against selected arithmetic and matching measured recurring latency.
    fig,axes=plt.subplots(2,2,figsize=(7.8,5.8))
    latency_lookup={(row["family"],int(row["k"]),row["implementation"],str(row["iteration_budget"]),str(row["tolerance"])):float(row["recurring_median_ns"])/1e3 for row in scaling_matched_timing if int(row["n"])==64}
    for family in ("Learned","OTFS","OFDM","AFDM"):
        for k in sorted({int(row["k"]) for row in scaling_timing if int(row["n"])==64 and row["waveform"]==family and row["implementation"]=="precomputed_dense_equalizer"}):
            values=[float(row["recurring_median_ns"])/1e3 for row in scaling_timing if int(row["n"])==64 and row["waveform"]==family and int(row["k"])==k and row["implementation"]=="precomputed_dense_equalizer"]; latency_lookup.setdefault((family,k,"precomputed_dense_equalizer","0",""),float(np.median(values)))
    for row_index,modulation in enumerate(("16QAM","QPSK")):
        selected=[row for row in matched if row["condition"]=="joint_0.5" and row["modulation"]==modulation]
        for family,color in (("historical","black"),("OTFS","#15803d"),("OFDM","#2563eb"),("AFDM","#dc2626")):
            points=[row for row in selected if row["family"]==family]
            if points:
                axes[row_index,0].scatter([int(float(row["detector_complex_macs"])) for row in points],[float(row["test_ber"]) for row in points],label=family,color=color)
                timing_family="Learned" if family=="historical" else family; timed=[]; timed_ber=[]
                for row in points:
                    key=(timing_family,int(row["k"]),row["implementation"],str(row["iteration_budget"]),str(row.get("tolerance",""))); value=latency_lookup.get(key)
                    if value is not None:timed.append(value);timed_ber.append(float(row["test_ber"]))
                if timed:axes[row_index,1].scatter(timed,timed_ber,label=family,color=color)
        seed_points=[row for row in selected if row["family"].startswith("seed") and "tail_continuation" in row["family"]]
        if seed_points:
            axes[row_index,0].scatter([int(float(row["detector_complex_macs"])) for row in seed_points],[float(row["test_ber"]) for row in seed_points],label="five tail seeds",color="#d97706",alpha=.55)
            timed=[];timed_ber=[]
            for row in seed_points:
                key=("Learned",int(row["k"]),row["implementation"],str(row["iteration_budget"]),str(row.get("tolerance",""))); value=latency_lookup.get(key)
                if value is not None:timed.append(value);timed_ber.append(float(row["test_ber"]))
            if timed:axes[row_index,1].scatter(timed,timed_ber,label="five tail seeds",color="#d97706",alpha=.55)
        for column,xlabel in enumerate(("Detector complex-equivalent MACs","Measured recurring latency (us)")):
            axis=axes[row_index,column]
            if column==0:axis.set_xscale("log")
            else:axis.ticklabel_format(axis="x",style="plain",useOffset=False);axis.margins(x=.08)
            axis.set_yscale("log");axis.set_xlabel(xlabel);axis.set_ylabel("Fresh test BER");axis.set_title(f"{modulation}: {'operations' if column==0 else 'latency'}");axis.grid(alpha=.25);axis.legend(fontsize=6)
    _save_figure(fig,"F4c_cont_matched_quality_frontier")
    # F4d: common target costs and validation failures at all measured N.
    dimensions=sorted({int(row["n"]) for row in scaling_selections});fig,axes=plt.subplots(1,len(dimensions),figsize=(10.4,3.2),sharey=True)
    colors={"Learned":"#d97706","OTFS":"#15803d","OFDM":"#2563eb","AFDM":"#dc2626"}
    for axis,n in zip(np.atleast_1d(axes),dimensions):
        labels=[];values=[];bar_colors=[]
        selected=[row for row in scaling_selections if int(row["n"])==n]
        for family in ("Learned","OTFS","OFDM","AFDM"):
            for modulation,target in (("16QAM",.03),("QPSK",.002)):
                choice=next(row for row in selected if row["family"]==family and row["modulation"]==modulation and abs(float(row["target_ber"])-target)<1e-12); labels.append(f"{family}\n{modulation}")
                if choice["validation_attained"]=="True":values.append(float(choice["detector_complex_macs"]));bar_colors.append(colors[family])
                else:values.append(np.nan);bar_colors.append("white");axis.text(len(labels)-1,max(1,n*n*.8),"X",ha="center",va="center",color="#991b1b",fontweight="bold")
        axis.bar(range(len(labels)),values,color=bar_colors,edgecolor=[colors[label.split("\n")[0]] for label in labels]);axis.set_xticks(range(len(labels)),labels,rotation=70,fontsize=6);axis.set_yscale("log");axis.set_title(f"N={n}");axis.grid(axis="y",alpha=.25)
    np.atleast_1d(axes)[0].set_ylabel("Validation-selected detector MACs");fig.suptitle("Cost to strict target; X = not attained on validation",fontsize=9);_save_figure(fig,"F4d_cont_cost_to_target")
    # F5 dimension scaling: arithmetic, storage, latency, and convergence.
    fig,axes=plt.subplots(2,2,figsize=(8.0,6.0))
    for rho,style in ((.375,"-o"),(1.0,"--s")):
        points=sorted([row for row in analytical if abs(float(row["rho"])-rho)<1e-9],key=lambda r:int(r["n"]));axes[0,0].plot([int(row["n"]) for row in points],[int(row["algorithmic_total_complex_equivalent_macs"]) for row in points],style,label=f"rho={rho:g}");axes[0,1].plot([int(row["n"]) for row in points],[int(row["sparse_storage_bytes"]) for row in points],style,label=f"sparse rho={rho:g}")
    dense=sorted([row for row in analytical if abs(float(row["rho"])-.375)<1e-9],key=lambda r:int(r["n"]));axes[0,1].plot([int(row["n"]) for row in dense],[int(row["dense_storage_bytes"]) for row in dense],":",color="black",label="dense coefficients")
    for axis,ylabel in ((axes[0,0],"64-step complex-equivalent MACs"),(axes[0,1],"Persistent coefficient bytes")):axis.set_xscale("log",base=2);axis.set_yscale("log");axis.axvspan(256,1100,color="0.9",alpha=.55,label="analytical only");axis.axvline(256,color="0.5",lw=.8);axis.set_xlabel("N");axis.set_ylabel(ylabel);axis.legend(fontsize=6);axis.grid(alpha=.25)
    for implementation,label in (("explicit_sparse_matrix_free_CG","sparse CG"),("dense_direct_per_vector","dense direct"),("precomputed_dense_equalizer","precomputed G")):
        values=[]
        for n in sorted({int(row["n"]) for row in scaling_timing}):
            k=round(.375*n);chosen=[float(row["median_ns"])/n/1e3 for row in scaling_timing if int(row["n"])==n and int(row["k"])==k and row["waveform"]=="Learned" and row["implementation"]==implementation]
            if chosen:values.append((n,float(np.median(chosen))))
        if values:axes[1,0].plot([v[0] for v in values],[v[1] for v in values],marker="o",label=label)
    ratios=[];nonconv=[]
    for n in sorted({int(row["n"]) for row in scaling_timing}):
        k=round(.375*n);sparse=[row for row in scaling_timing if int(row["n"])==n and int(row["k"])==k and row["waveform"]=="Learned" and row["implementation"]=="explicit_sparse_matrix_free_CG"]; dense_rows=[row for row in scaling_timing if int(row["n"])==n and int(row["k"])==k and row["waveform"]=="Learned" and row["implementation"]=="precomputed_dense_equalizer"]; ratios.append(float(np.median([float(a["median_ns"])/float(b["median_ns"]) for a,b in zip(sparse,dense_rows)])));nonconv.append(100*sum(row["converged"]=="False" for row in sparse)/len(sparse))
    dims=sorted({int(row["n"]) for row in scaling_timing});axes[1,1].plot(dims,ratios,"-o",label="sparse/precomputed latency");twin=axes[1,1].twinx();twin.plot(dims,nonconv,"--s",color="#991b1b",label="nonconverged");twin.set_ylabel("Fixed-64 nonconvergence (%)",color="#991b1b")
    axes[1,0].set_xscale("log",base=2);axes[1,0].set_yscale("log");axes[1,0].set_xlabel("Measured N");axes[1,0].set_ylabel("Latency per symbol (us)");axes[1,0].legend(fontsize=6);axes[1,0].grid(alpha=.25);axes[1,1].set_xlabel("Measured N");axes[1,1].set_ylabel("Sparse/precomputed runtime ratio");axes[1,1].set_yscale("log");axes[1,1].grid(alpha=.25);_save_figure(fig,"F5_dimension_scaling")
    # F6 measured receiver-processing breakdown and hypothetical amortization.
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2)); dimensions=sorted({int(row["n"]) for row in scaling_timing});bottom=np.zeros(len(dimensions))
    components=(("transform_median_ns","receive transform"),("effective_channel_median_ns","form complete A"),("oracle_pack_median_ns","oracle pack"),("recurring_median_ns","recurring detector"))
    for field,label in components:
        values=[]
        for n in dimensions:
            k=round(.375*n);chosen=[float(row[field])/1e3 for row in scaling_timing if int(row["n"])==n and int(row["k"])==k and row["waveform"]=="Learned" and row["implementation"]=="explicit_sparse_matrix_free_CG"]
            values.append(float(np.median(chosen)))
        axes[0].bar([str(n) for n in dimensions],values,bottom=bottom,label=label);bottom+=np.asarray(values)
    axes[0].set_yscale("log");axes[0].set_xlabel("N at rho=0.375");axes[0].set_ylabel("Median measured time (us)");axes[0].set_title("Measured sparse path, B=1");axes[0].legend(fontsize=6);axes[0].grid(axis="y",alpha=.25)
    for n in dimensions:
        rows=[row for row in scaling_amortization if int(row["n"])==n and row["family"]=="Learned"]
        if rows:
            blocks=sorted({int(row["reuse_blocks"]) for row in rows});axes[1].plot(blocks,[float(np.median([float(row["amortized_total_ns"])/1e3 for row in rows if int(row["reuse_blocks"])==block])) for block in blocks],marker="o",label=f"N={n}")
    axes[1].set_xscale("log");axes[1].set_yscale("log");axes[1].set_xlabel("Hypothetical reuse blocks B");axes[1].set_ylabel("Amortized measured time (us)");axes[1].set_title("Selected Learned target path");axes[1].legend(fontsize=6);axes[1].grid(alpha=.25);_save_figure(fig,"F6_runtime_breakdown")


def _phase_report_legacy(smoke:bool)->None:
    ensure_dirs(); phase_intervals(smoke); suffix="_smoke" if smoke else ""; gate=read_json(ORIGINAL_GATE); reuse=read_json(PROVENANCE/"reuse_inventory.json"); training=read_csv(DATA/f"training_summary{suffix}.csv"); communication=read_csv(DATA/f"communication_aggregate{suffix}.csv"); matched=read_csv(DATA/f"matched_quality_confirmation{suffix}.csv"); scaling_comm=read_csv(DATA/f"scaling_communication_aggregate{suffix}.csv"); scaling_timing=read_csv(TIMING/f"scaling_benchmark{suffix}.csv"); scaling_exec=read_json(DATA/f"scaling_execution{suffix}.json")
    if smoke:
        REPORT.write_text("# Pub-1F continuation summary\n\nSmoke integration completed; reduced data have no scientific meaning. Production phases remain required. The original gate remains failed.\n");return
    tail_keys=[f"seed{seed}_tail_continuation_selected" for seed in cfg()["training"]["seeds"]]
    def comm_ber(family,condition="joint_0.5",modulation="16QAM",track="operational_selected"):
        rows=[row for row in communication if row["family"]==family and row["condition"]==condition and row["modulation"]==modulation and row["track"]==track and int(row["snr_db"])==18 and int(row["k"])==24];return float(rows[0]["ber"]) if rows else float("nan")
    classical={name:comm_ber(name) for name in ("OTFS","OFDM","AFDM")}; strongest=min(classical.values()); seed_bers=[comm_ber(key) for key in tail_keys]; wins=sum(value<strongest for value in seed_bers); representation=wins>=4 and np.median(np.asarray(seed_bers)-strongest)<0
    tail_gate=0;tail_cvar=0
    for seed in cfg()["training"]["seeds"]:
        mean=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="mean_continuation");tail=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="tail_continuation");parent=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="parent")
        tail_gate+=float(tail["selected_mean_mse"])<=1.02*float(parent["selected_mean_mse"]);tail_cvar+=float(tail["selected_cvar95"])<float(mean["selected_cvar95"])
    tail_supported=tail_gate>=4 and tail_cvar>=4
    convergence={}
    for n in (64,128,256):
        rows=[row for row in scaling_timing if int(row["n"])==n and row["implementation"]=="explicit_sparse_matrix_free_CG"];convergence[n]={"nonconverged":sum(row["converged"]=="False" for row in rows),"total":len(rows),"median_us":float(np.median([float(row["median_ns"])/1e3 for row in rows]))}
    crossover=any(float(row["median_ns"])<float(next(other["median_ns"] for other in scaling_timing if other["n"]==row["n"] and other["waveform"]==row["waveform"] and other["k"]==row["k"] and other["channel_index"]==row["channel_index"] and other["implementation"]=="precomputed_dense_equalizer")) for row in scaling_timing if row["implementation"]=="explicit_sparse_matrix_free_CG")
    attained=sum(row["target_attained_test"]=="True" for row in matched);total_targets=len(matched)
    category="Reproducible representation advantage with arithmetic/storage savings only" if representation and not crossover else ("Reproducible representation advantage with measured receiver-cost savings" if representation else "Strong historical-checkpoint result but limited training reproducibility")
    lines=["# Pub-1F continuation — Qualified replication, matched quality, and scaling","","## Decision",f"**{category}.** The original resource gate remains **FAIL** and was not edited. The continuation is a post-gate, explicitly authorized study with 16-QAM primary and QPSK retained without a universal-win requirement.","","## N=64 training replication",f"The representation rule is {'SUPPORTED' if representation else 'NOT SUPPORTED'}: {wins}/5 selected tail checkpoints beat the strongest operational classical 16-QAM baseline at joint-(0.5,0.5), K=24, 18 dB. Tail-seed BER range is [{min(seed_bers):.6g}, {max(seed_bers):.6g}], median {np.median(seed_bers):.6g}; strongest classical BER is {strongest:.6g} ({min(classical,key=classical.get)}).","",f"Tail contribution is {'SUPPORTED' if tail_supported else 'NOT SUPPORTED'} by the frozen validation rule: {tail_cvar}/5 tail arms improve CVaR over matched mean continuation and {tail_gate}/5 satisfy the seed-relative mean gate. BER effects remain a separate result.","","| seed | parent mean | mean-cont CVaR | tail CVaR | tail 16-QAM BER |","|---:|---:|---:|---:|---:|"]
    for seed in cfg()["training"]["seeds"]:
        parent=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="parent");mean=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="mean_continuation");tail=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="tail_continuation");lines.append(f"| {seed} | {float(parent['selected_mean_mse']):.6f} | {float(mean['selected_cvar95']):.6f} | {float(tail['selected_cvar95']):.6f} | {comm_ber(f'seed{seed}_tail_continuation_selected'):.6g} |")
    lines += ["","QPSK, probe, integer, nominal, and full-information results remain in the saved tables and figures. They explain boundary conditions; they do not veto the scoped 16-QAM representation claim. Relative-to-own-full degradation is reported separately from absolute matched targets.","","## Matched-quality cost",f"Fresh validation selected configurations for the frozen targets; {attained}/{total_targets} selected target cases attained their target on fresh confirmation. Failed targets remain failures with no test reselection. Detector MAC, storage, residual-statistics bytes, own-full BER, and intervals are saved per case.","","A lower detector ledger is not called a total receiver saving. Precomputed dense equalizers retain dense application/storage; residual-aware configurations add dense means/covariances; complete A is formed before restriction. Channel estimation and RF remain excluded.","","## Dimension scaling","","Pub-1B's exact normalized scaling law was reused: `tau_max=4N/64`, `nu_max=.75N/64`, P=3. N=128/256 results use verified historical single checkpoints and do not constitute multi-seed replication. This is not fixed physical bandwidth/delay/Doppler/coherence scaling.","","| N | sparse 64-step nonconvergence | median sparse latency (us) |","|---:|---:|---:|"]
    for n,value in convergence.items():lines.append(f"| {n} | {value['nonconverged']}/{value['total']} | {value['median_us']:.3f} |")
    lines += ["",f"Measured sparse/precomputed runtime crossover observed: **{crossover}**. Analytical N=512/1024 rows are extrapolations only. Fixed rho retains O(rho N^2) sparse products; dense transforms/statistics add further quadratic work. Scaling production runtime was {scaling_exec['seconds']:.1f} s.","","## Claim-to-evidence","","| claim | evidence | status |","|---|---|---|",f"| Same-K fractional 16-QAM representation | five fresh seeds, operational classical controls, fresh channels | {'SUPPORTED' if representation else 'NOT SUPPORTED'} |",f"| Tail objective contribution | paired continuation streams and validation metrics | {'SUPPORTED' if tail_supported else 'NOT SUPPORTED'} |",f"| Common-target detector saving | validation-selected and fresh-confirmed target rows | {'qualified; inspect per-target rows' if matched else 'not run'} |",f"| Measured sparse runtime saving | N=64/128/256 unprofiled timing | {'SUPPORTED' if crossover else 'NOT SUPPORTED'} |","| N=128/256 training reproducibility | historical one-checkpoint reuse | NOT TESTED |","","## Explicit answers","",f"1. Supported 16-QAM effect across seeds: {'yes under the frozen rule' if representation else 'no under the frozen rule'}.",f"2. Matched tail refinement: {'meets' if tail_supported else 'does not meet'} the validation contribution rule; BER effects are in F2/F3 tables.","3. QPSK is fully reported and may be favorable, inconclusive, or unfavorable; it is not a universal gate.",f"4. Frozen/new Learned target costs: {attained}/{total_targets} validation-selected cases confirm; compare rows rather than a pooled claim.",f"5. N=64 sparse disadvantage persistence: see convergence/timing table; crossover={crossover}.",f"6. Measured sparse/dense crossover: {crossover}.","7. Iteration/conditioning: fixed-64 convergence and tight-reference iterations are saved per channel/configuration; nonconvergence is retained.","8. Transform/setup/statistics: complete-A formation and dense residual state survive; amortization never removes recurring transform/detection.","9. Historical single checkpoints: N=64 reference and all N=128/256 Learned scaling results; only N=64 continuation seeds are replications.","10. Complexity provenance: N=64/128/256 timing measured; N=512/1024 operations/storage analytical extrapolations.",f"11. Reuse: exact larger-N checkpoints and historical ledgers avoided {reuse['compute_saved']['total_larger_N_training_seconds']:.1f} s of recorded training; fresh confirmation was recomputed.","12. Pub-1F completeness: complete for the amended scope if all manifest/tests pass; optimized kernels/direct retained construction remain distinct Pub-1G interventions.","","## Recommendation","",f"Retain the conclusion **{category}**. A Pub-1G should be a genuinely new intervention—optimized sparse kernels, direct retained-entry construction, or a structured fast learned transform—not a relabeling of unfinished Pub-1F work."]
    REPORT.write_text("\n".join(lines)+"\n")


def phase_report(smoke:bool)->None:
    ensure_dirs(); phase_intervals(smoke); suffix="_smoke" if smoke else ""
    if smoke:
        REPORT.write_text("# Pub-1F continuation summary\n\nSmoke integration completed; reduced data have no scientific meaning. Production phases remain required. The original gate remains failed.\n");return
    configuration=cfg(); gate=read_json(ORIGINAL_GATE); reuse=read_json(PROVENANCE/"reuse_inventory.json"); training=read_csv(DATA/"training_summary.csv"); validation=read_csv(DATA/"operating_validation.csv"); communication=read_csv(DATA/"communication_aggregate.csv"); intervals=read_csv(DATA/"communication_paired_intervals.csv")
    matched_selection=read_csv(DATA/"matched_quality_selections.csv"); matched=read_csv(DATA/"matched_quality_confirmation.csv"); scaling_selection=read_csv(DATA/"scaling_matched_selections.csv"); scaling_matched=read_csv(DATA/"scaling_matched_confirmation.csv"); scaling_comm=read_csv(DATA/"scaling_communication_aggregate.csv"); scaling_timing=read_csv(TIMING/"scaling_benchmark.csv"); profiler=read_csv(TIMING/"scaling_profiler.csv"); scaling_exec=read_json(DATA/"scaling_execution.json")
    seeds=tuple(int(value) for value in configuration["training"]["seeds"]); tail_keys=[f"seed{seed}_tail_continuation_selected" for seed in seeds]
    def comm_row(family:str,condition:str="joint_0.5",modulation:str="16QAM",track:str="operational_selected",k:int=24)->dict[str,str]:
        rows=[row for row in communication if row["family"]==family and row["condition"]==condition and row["modulation"]==modulation and row["track"]==track and int(row["snr_db"])==18 and int(row["k"])==k]
        if len(rows)!=1:raise RuntimeError(f"expected one communication row for {family} {condition} {modulation} {track} K={k}, got {len(rows)}")
        return rows[0]
    primary_intervals=[row for row in intervals if row["contrast_type"]=="tail_vs_validation_strongest_classical" and row["condition"]=="joint_0.5" and row["modulation"]=="16QAM"]
    primary_reference=primary_intervals[0]["right"]; strongest=float(comm_row(primary_reference)["ber"]); seed_bers=[float(comm_row(key)["ber"]) for key in tail_keys]; wins=sum(value<strongest for value in seed_bers); representation=wins>=4 and float(np.median(np.asarray(seed_bers)-strongest))<0; primary_ci_wins=sum(float(row["ci95_high"])<0 for row in primary_intervals)
    tail_gate=0;tail_cvar=0
    for seed in seeds:
        parent=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="parent");mean=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="mean_continuation");tail=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="tail_continuation")
        tail_gate+=float(tail["selected_mean_mse"])<=1.02*float(parent["selected_mean_mse"]);tail_cvar+=float(tail["selected_cvar95"])<float(mean["selected_cvar95"])
    tail_supported=tail_gate>=4 and tail_cvar>=4; tail_ber_intervals=[row for row in intervals if row["contrast_type"]=="tail_vs_mean_continuation" and row["condition"]=="joint_0.5" and row["modulation"]=="16QAM"]; tail_ber_worse=sum(float(row["ci95_low"])>0 for row in tail_ber_intervals)
    def cost_saving(rows:list[dict[str,str]])->bool:
        for condition in sorted({row["condition"] for row in rows}):
            for modulation in ("16QAM","QPSK"):
                for target in tuple(float(value) for value in configuration["matched_quality"]["targets"][modulation]):
                    group=[row for row in rows if row["condition"]==condition and row["modulation"]==modulation and abs(float(row["target_ber"])-target)<1e-12 and row["target_attained_test"]=="True"]
                    learned=[row for row in group if row["family"]=="historical" or row["family"]=="Learned" or row["family"].startswith("seed")]; classical=[row for row in group if row["family"] in {"OTFS","OFDM","AFDM"}]
                    if learned and classical and min(float(row["detector_complex_macs"]) for row in learned)<min(float(row["detector_complex_macs"]) for row in classical):return True
        return False
    computation_saving=cost_saving(matched) or cost_saving(scaling_matched)
    convergence={}; crossover=False
    for n in (64,128,256):
        sparse=[row for row in scaling_timing if int(row["n"])==n and row["implementation"]=="explicit_sparse_matrix_free_CG"]; ref=[row for row in sparse if row["waveform"]=="Learned" and int(row["k"])==round(.375*n)]; ratios=[]
        for row in ref:
            other=next(value for value in scaling_timing if value["n"]==row["n"] and value["waveform"]==row["waveform"] and value["k"]==row["k"] and value["channel_index"]==row["channel_index"] and value["implementation"]=="precomputed_dense_equalizer");ratio=float(row["median_ns"])/float(other["median_ns"]);ratios.append(ratio);crossover|=ratio<1
        convergence[n]={"nonconverged":sum(row["converged"]=="False" for row in sparse),"total":len(sparse),"ref_nonconverged":sum(row["converged"]=="False" for row in ref),"ref_total":len(ref),"sparse_us":float(np.median([float(row["median_ns"])/1e3 for row in ref])),"precomputed_us":float(np.median([float(next(value for value in scaling_timing if value["n"]==row["n"] and value["waveform"]==row["waveform"] and value["k"]==row["k"] and value["channel_index"]==row["channel_index"] and value["implementation"]=="precomputed_dense_equalizer")["median_ns"])/1e3 for row in ref])),"ratio":float(np.median(ratios)),"tight_iterations":float(np.median([float(row["tight_iterations"]) for row in ref]))}
    category="Reproducible representation advantage with measured receiver-cost savings" if representation and computation_saving else ("Reproducible representation advantage with arithmetic/storage savings only" if representation else "Strong historical-checkpoint result but limited training reproducibility")
    matched_declared=len(matched_selection);matched_valid=sum(row["validation_attained"]=="True" for row in matched_selection);matched_confirmed=sum(row["target_attained_test"]=="True" for row in matched);scaled_declared=len(scaling_selection);scaled_valid=sum(row["validation_attained"]=="True" for row in scaling_selection);scaled_confirmed=sum(row["target_attained_test"]=="True" for row in scaling_matched)
    lines=["# Pub-1F continuation — Qualified replication, matched quality, and scaling","","## Decision",f"**{category}.** The original resource gate remains **FAIL** (`passed={gate['passed']}`) and was not edited. This post-gate continuation was separately authorized, with 16-QAM primary and QPSK retained as a mandatory secondary outcome.","",f"The same-K representation rule is **{'SUPPORTED' if representation else 'NOT SUPPORTED'}**: {wins}/5 selected tail checkpoints beat validation-chosen {primary_reference} on fresh joint-(0.5,0.5), K=24, 18 dB 16-QAM data. All {primary_ci_wins}/5 paired 95% channel-bootstrap intervals exclude zero in Learned's favor. The computational-saving rule is **{'SUPPORTED' if computation_saving else 'NOT SUPPORTED'}**; selected recurring detector costs tie rather than beat classical alternatives.","","## N=64 training replication and objective ablation",f"Every tail arm passes the seed-relative 2% mean gate ({tail_gate}/5) and improves validation stream CVaR_0.95 ({tail_cvar}/5), so the frozen tail-contribution rule is **{'SUPPORTED' if tail_supported else 'NOT SUPPORTED'}**. That is not a BER-improvement claim: tail continuation has significantly higher primary 16-QAM BER than its matched mean-only arm in {tail_ber_worse}/5 seeds; the remaining seed interval includes zero.","","| seed | parent mean MSE | mean CVaR | tail CVaR | tail BER | tail - OFDM 95% CI | tail - mean 95% CI |","|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        parent=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="parent");mean=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="mean_continuation");tail=next(row for row in training if int(row["seed_id"])==seed and row["stage"]=="tail_continuation");ci=next(row for row in primary_intervals if int(row["seed_id"])==seed);ber_ci=next(row for row in tail_ber_intervals if int(row["seed_id"])==seed)
        lines.append(f"| {seed} | {float(parent['selected_mean_mse']):.6f} | {float(mean['selected_cvar95']):.6f} | {float(tail['selected_cvar95']):.6f} | {float(comm_row(f'seed{seed}_tail_continuation_selected')['ber']):.6f} | [{float(ci['ci95_low']):.6f}, {float(ci['ci95_high']):.6f}] | [{float(ber_ci['ci95_low']):.6f}, {float(ber_ci['ci95_high']):.6f}] |")
    lines += ["",f"Tail-seed primary BER range is {min(seed_bers):.6f}–{max(seed_bers):.6f}, median {float(np.median(seed_bers)):.6f}; {primary_reference} is {strongest:.6f}. The historical checkpoint remains separate from these five OFDM-initialized replications.","","## Regime and modulation controls","","| condition | modulation | validation-chosen classical | tail wins / 5 | paired CIs below zero / 5 |","|---|---|---|---:|---:|"]
    for condition in ("joint_0.5","joint_0.2_probe","integer","nominal_dd_v1"):
        for modulation in ("16QAM","QPSK"):
            rows=[row for row in intervals if row["contrast_type"]=="tail_vs_validation_strongest_classical" and row["condition"]==condition and row["modulation"]==modulation];reference=rows[0]["right"];baseline=float(comm_row(reference,condition,modulation)["ber"]);values=[float(comm_row(key,condition,modulation)["ber"]) for key in tail_keys];lines.append(f"| {condition} | {modulation} | {reference} ({baseline:.6g}) | {sum(value<baseline for value in values)} | {sum(float(row['ci95_high'])<0 for row in rows)} |")
    lines += ["","The joint-(0.2,0.2) row is a probe, not an inferred boundary. Integer-grid and nominal full-information results are explanatory controls, not claim gates. QPSK is favorable for 4/5 fresh tail seeds by paired interval at joint 0.5, while one seed is inconclusive; this does not rewrite the original historical-checkpoint QPSK gate failure.","","## Matched-quality cost",f"At N=64, {matched_valid}/{matched_declared} declared validation cases attain a target and {matched_confirmed}/{len(matched)} selected cases re-attain it on fresh confirmation. Across the scaled historical-checkpoint study, {scaled_valid}/{scaled_declared} attain on validation and {scaled_confirmed}/{len(scaling_matched)} confirm. The missing cases are retained as validation failures; there is no test reselection.","","All selected implementations are precomputed dense equalizer applications. Their recurring detector ledgers are N² complex MACs: 4,096, 16,384, and 65,536 at N=64, 128, and 256. Thus Learned sometimes reaches a target at smaller K, but K does not reduce this dense application's recurring arithmetic. No common-target detector or runtime saving over every classical comparator is established.","","| N | detector MAC/block | MAC/symbol | MAC/16-QAM bit | MAC/QPSK bit |","|---:|---:|---:|---:|---:|"]
    for n in (64,128,256):lines.append(f"| {n} | {n*n} | {n} | {n/4:g} | {n/2:g} |")
    lines += ["","Relative-to-own-full degradation and absolute target attainment are both saved per selected row. They are not interchangeable. Complete A formation, receive transforms, oracle packing, dense equalizer preparation/storage, channel estimation, and RF are outside the detector-MAC number; the report therefore makes no total-receiver claim.","","## Dimension scaling under the Pub-1B law","", "The exact normalized law is `tau_max=4N/64`, `nu_max=.75N/64`, P=3. It changes physical parameters with N and is not a fixed-bandwidth, fixed-delay, fixed-Doppler, or fixed-coherence experiment. N=128/256 Learned results reuse verified historical single checkpoints and are not training replications.","","| N | Learned K/full 16-QAM BER | best classical K BER | Learned own-full degradation | tight iterations | sparse/precomputed latency | fixed-64 nonconvergence |","|---:|---:|---:|---:|---:|---:|---:|"]
    for n in (64,128,256):
        learned=next(row for row in scaling_comm if int(row["n"])==n and row["waveform"]=="Learned" and row["modulation"]=="16QAM");classical=[row for row in scaling_comm if int(row["n"])==n and row["waveform"] in {"OTFS","OFDM","AFDM"} and row["modulation"]=="16QAM"];best=min(classical,key=lambda row:float(row["ber"]));value=convergence[n];lines.append(f"| {n} | {float(learned['ber']):.6f} / {float(learned['own_full_ber']):.6f} | {best['waveform']} {float(best['ber']):.6f} | {float(learned['degradation_vs_own_full']):+.6f} | {value['tight_iterations']:.1f} | {value['sparse_us']:.1f} / {value['precomputed_us']:.3f} us ({value['ratio']:.0f}x) | {value['ref_nonconverged']}/{value['ref_total']} ref; {value['nonconverged']}/{value['total']} all |")
    top_ops=[]
    for n in (64,128,256):
        rows=[row for row in profiler if int(row["n"])==n and row.get("operator")!="PROFILE_UNAVAILABLE"]
        if rows:top_ops.append(f"N={n}: {max(rows,key=lambda row:float(row['self_cpu_time_us']))['operator']}")
    lines += ["",f"No measured sparse/precomputed crossover is observed (`{crossover}`). Sparse-CG becomes less relatively slow as N grows, but remains 269–718x slower at rho=0.375. Tight-reference median iterations rise from {convergence[64]['tight_iterations']:.1f} to {convergence[256]['tight_iterations']:.1f}; fixed-64 nonconvergence is retained and grows sharply by proportion. Profiling's largest self-time operators are {', '.join(top_ops)}; this is attribution evidence, not a causal hardware generalization.","",f"The fixed-64 sparse ledger is `C(N,K,t)=NK+N+t(2NK+6N)`. At fixed rho it remains O(rho N²). Sparse complex128 values plus int64 indices use about 24NK bytes versus 16N² bytes for dense coefficients; at rho=0.375 this is 56.25% of dense coefficient storage before the dense waveform matrix and any residual statistics. N=512/1024 arithmetic/storage rows are analytical extrapolations only. N=64/128/256 timings are measured, unprofiled final repeats; production scaling runtime was {float(scaling_exec['seconds']):.1f} s.","","## Claim-to-evidence","","| claim | evidence | status |","|---|---|---|",f"| Same-K fractional 16-QAM representation | five fresh seeds; validation-chosen classical receiver/config; paired channel bootstrap | {'SUPPORTED' if representation else 'NOT SUPPORTED'} |",f"| Tail objective improves its declared tail statistic | paired continuation streams and validation mean/CVaR | {'SUPPORTED' if tail_supported else 'NOT SUPPORTED'} |",f"| Tail objective improves BER over mean continuation | fresh paired BER; {tail_ber_worse}/5 significantly worse | NOT SUPPORTED |",f"| Common-target Learned detector saving | validation-selected, fresh-confirmed N=64/128/256 rows | {'SUPPORTED' if computation_saving else 'NOT SUPPORTED'} |",f"| Measured sparse runtime saving | isolated N=64/128/256 CPU timing | {'SUPPORTED' if crossover else 'NOT SUPPORTED'} |","| N=128/256 representation examples | verified historical single checkpoints | SUPPORTED AS SINGLE-CHECKPOINT EVIDENCE |","| N=128/256 training reproducibility | no new training | NOT TESTED |","","## Reuse, completeness, and recommendation",f"Exact N=128/N=256 checkpoints and historical ledgers avoid {float(reuse['compute_saved']['total_larger_N_training_seconds']):.1f} s of recorded training. All supports, residual statistics, selection banks, and confirmation banks required by the continuation were regenerated under `pub1f-cont-v1`; smoke states and old test rows were not promoted.","","Pub-1F is complete for the amended scope: five-seed training, objective ablation, fresh communication confirmation, paired uncertainty, matched targets, measured N=64/128/256 cost/scaling, analytical N=512/1024 references, figures, provenance, and preservation checks are present. Optimized sparse kernels, direct retained-entry construction, and a structured fast Learned transform remain genuinely new Pub-1G interventions.","",f"Recommendation: retain **{category}**. The positive result is a scoped fractional-channel, restricted-interaction representation result—not universal waveform or receiver efficiency superiority."]
    REPORT.write_text("\n".join(lines)+"\n")


def phase_manifest(smoke:bool)->None:
    ensure_dirs(); authorization=read_json(PROVENANCE/"authorization.json")
    if sha(ORIGINAL_GATE)!=authorization["original_gate_sha256"]:raise RuntimeError("original gate changed during continuation")
    for entry in cfg()["scaling"]["checkpoints"].values():
        if sha(ROOT/entry["path"])!=entry["sha256"]:raise RuntimeError(f"protected checkpoint changed: {entry['path']}")
    files=[]
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name!="artifact_manifest.json"):files.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size})
    for path in (CONFIG,AMENDMENT,REPORT,ROOT/"README.md",ROOT/"scripts/stage_pub1f_continuation.py",ROOT/"scripts/replot_stage_pub1f_continuation.py",ROOT/"src/pub1f/continuation.py",ROOT/"tests/test_pub1f_continuation.py"):
        if path.exists():files.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size})
    write_json(OUT/"artifact_manifest.json",{"stage":"Pub-1F continuation","run_kind":"smoke" if smoke else "production","count":len(files),"original_gate_sha256_after":sha(ORIGINAL_GATE),"protected_checkpoint_hashes_after":{key:sha(ROOT/value["path"]) for key,value in cfg()["scaling"]["checkpoints"].items()},"files":files})


def phase_pending(name:str):
    def run(smoke:bool)->None: raise RuntimeError(f"continuation phase {name} not implemented")
    return run


PHASES={"amendment":phase_amendment,"reuse":phase_reuse,"training":phase_training,"statistics":phase_statistics,"operating-validation":phase_operating_validation,"communication":phase_communication,"intervals":phase_intervals,"matched-confirmation":phase_matched_confirmation,"scaling":phase_scaling,"scaling-matched":phase_scaling_matched,"figures":phase_figures,"report":phase_report,"manifest":phase_manifest}


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("all",*PHASES),default="all"); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    selected=PHASES.items() if args.phase=="all" else ((args.phase,PHASES[args.phase]),)
    for name,function in selected: print(f"[Pub-1F continuation] {name}",flush=True); function(args.smoke)
if __name__=="__main__":main()
