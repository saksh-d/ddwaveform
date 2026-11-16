"""Read-only bounded snapshot of a completed scaling release; never runs source code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(source, destination, bundle, manifest_path):
    if destination.exists() or bundle.exists():
        raise FileExistsError('Choose fresh local snapshot/bundle paths')
    release = json.loads((source/'DELIVERY_MANIFEST.json').read_text())
    names = ['DELIVERY_MANIFEST.json', 'RUN_STATUS.md', 'REPORT.md', 'PROVENANCE.md',
             'config.json', 'SOFTWARE.json', 'release_checks.json', 'source_manifest.json',
             'source_amendment_svd.json', 'communication_summary.csv', 'example_channel_paths.csv',
             'group_frequency_regions.csv', 'basis_frequency_membership.csv', 'matrices/MANIFEST.json']
    for n in (64,128,256):
        names += [f'matrices/learned_N{n}.npy', f'communication/N{n}_nominal/complete.json',
                  f'communication/N{n}_nominal/per_channel.npz']
        for waveform in ('Learned','OTFS','AFDM'):
            names += [f'structure/N{n}_nominal_{waveform}.{suffix}' for suffix in ('npz','json')]
            names += [f'structure/N{n}_{waveform}_grouping.json']
        if n != 64:
            names += [f'training/N{n}/{phase}/selected.json' for phase in ('mean','tail')]
    names += [str(p.relative_to(source)) for p in (source/'executed_source_svd_v3').rglob('*.py')]
    before = {name: (digest(source/name), (source/name).stat().st_size, (source/name).stat().st_mtime_ns) for name in names}
    for name, expected in release['files'].items():
        # Verify complete declared delivery, without copying its unrelated plots.
        if digest(source/name) != expected:
            raise ValueError(f'Release checksum mismatch: {name}')
    for n in (64,128,256):
        complete=json.loads((source/f'communication/N{n}_nominal/complete.json').read_text())
        if complete['channels'] < 2048:
            raise ValueError('Incomplete minimum coverage')
        for waveform in ('Learned','OTFS','AFDM'):
            name=f'structure/N{n}_nominal_{waveform}.npz'
            meta=json.loads((source/name).with_suffix('.json').read_text())
            if digest(source/name) != meta['npz_sha256']:
                raise ValueError(f'Structural checksum mismatch: {name}')
    destination.mkdir(parents=True)
    for name in names:
        target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    after={name: (digest(source/name),(source/name).stat().st_size,(source/name).stat().st_mtime_ns) for name in names}
    if before != after or any(digest(destination/name)!=before[name][0] for name in names):
        raise ValueError('Source changed during snapshot; do not use copy')
    manifest={'schema':1,'source':'Completed waveform-mechanism-study/results/scaling_dd_transfer release, 2026-11-16',
              'source_thread':'b969399d-3ecb-42ff-b8fb-f0edae03151e',
              'snapshot_policy':'Two-pass hash/size/mtime agreement; nominal evidence only; no source writes',
              'files':{name:{'sha256':before[name][0],'bytes':before[name][1]} for name in sorted(names)}}
    with tarfile.open(bundle,'w:gz') as archive:
        for name in sorted(names): archive.add(destination/name,arcname=name,recursive=False)
    manifest['bundle_sha256']=digest(bundle)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    for path in destination.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    print(json.dumps({'files':len(names),'bundle_bytes':bundle.stat().st_size,'sha256':manifest['bundle_sha256']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','destination','bundle','manifest'): parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args(); snapshot(args.source,args.destination,args.bundle,args.manifest)
