import io
import tarfile
import pytest
from waveforms.artifacts import provision, sha, verify


def bundle(tmp_path, name="input.txt", value=b"verified evidence"):
    source = tmp_path / "input.txt"
    source.write_bytes(value)
    manifest = {"files": {"input.txt": {"sha256": sha(source), "bytes": len(value)}}}
    path = tmp_path / "bundle.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(value)
        archive.addfile(info, io.BytesIO(value))
    return path, manifest


def test_provision_verified_and_no_overwrite(tmp_path):
    path, manifest = bundle(tmp_path)
    dest = tmp_path / "local"
    assert provision(path, dest, manifest)["files"] == 1
    assert verify(dest, manifest)["files"] == 1
    with pytest.raises(FileExistsError):
        provision(path, dest, manifest)


def test_corrupt_and_traversal_rejected(tmp_path):
    path, manifest = bundle(tmp_path)
    manifest["files"]["input.txt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="altered"):
        provision(path, tmp_path / "bad", manifest)
    assert not (tmp_path / "bad").exists()
    path, manifest = bundle(tmp_path, "../escape")
    with pytest.raises(ValueError, match="Unsafe"):
        provision(path, tmp_path / "unsafe", manifest)


def test_classical_smoke_without_local_arrays(tmp_path, monkeypatch):
    from waveforms.experiments import check, export_dd
    monkeypatch.chdir(tmp_path)
    assert "Learned" not in check()["bases"]
    result = export_dd("local/dd", count=1)
    assert result["channels"] == 1
    with pytest.raises(FileNotFoundError, match="Provision"):
        check(tmp_path / "missing")
