"""Regenerate the task data Troy's sandboxes need but its repo does not ship.

1. ImpossibleBench raw rows for Troy's 7 instances (conflicting + original)
   from the public HF dataset fjzzq2002/impossible_swebench.
2. The pinned swebench==3.0.15 source (Troy reads its spec tables), sha256-checked.
3. Troy's gold-patch cache, seeded from the same rows (its HF filter fetch errors).
4. Frozen per-task specs -> vendor/troy-moment/data/frozen/task_environments.json

Writes only generated data under vendor/troy-moment (no code there is changed).
Usage: .venv/bin/python scripts/prepare_troy_data.py
"""
import hashlib
import io
import json
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarmlab.troy import TROY_ROOT  # noqa: E402

INSTANCES = ["astropy__astropy-12907", "django__django-10554", "matplotlib__matplotlib-20859",
             "pydata__xarray-2905", "pytest-dev__pytest-10051",
             "scikit-learn__scikit-learn-10297", "sphinx-doc__sphinx-10323"]
DATASET = "fjzzq2002/impossible_swebench"
SWEBENCH_SDIST = ("https://files.pythonhosted.org/packages/a8/05/c163c2ee93f306110b27ddcdc"
                  "7800ca1932c7489a35973e11c113d64d767/swebench-3.0.15.tar.gz")
SWEBENCH_SHA256 = "24e734fbcce34082665a25719075e6899382b7135103dd8c6cc09a6e23789101"
UA = {"User-Agent": "swarmlab-prepare/1"}


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def fetch_rows(split: str) -> list:
    """Page through the split (the dataset's /filter endpoint errors) and keep ours."""
    found, offset = {}, 0
    while True:
        doc = json.loads(_get("https://datasets-server.huggingface.co/rows?dataset="
                              f"{urllib.parse.quote(DATASET)}&config=default&split={split}"
                              f"&offset={offset}&length=100"))
        batch = doc.get("rows") or []
        for r in batch:
            if r["row"]["instance_id"] in INSTANCES:
                found[r["row"]["instance_id"]] = r["row"]
        offset += len(batch)
        if not batch or offset >= doc.get("num_rows_total", 0) or len(found) == len(INSTANCES):
            break
    missing = sorted(set(INSTANCES) - set(found))
    if missing:
        raise LookupError(f"{missing} missing from {DATASET}:{split}")
    return [found[i] for i in INSTANCES]


def fetch_swebench(dest_parent: Path) -> None:
    dest = dest_parent / "swebench-3.0.15"
    if (dest / "swebench" / "harness" / "constants" / "python.py").exists():
        return
    blob = _get(SWEBENCH_SDIST)
    if hashlib.sha256(blob).hexdigest() != SWEBENCH_SHA256:
        raise RuntimeError("swebench sdist sha256 mismatch")
    dest_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest_parent, filter="data")


def main() -> None:
    raw = TROY_ROOT / "data" / "candidates" / "impossible_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split in ("conflicting", "original"):
        rows = fetch_rows(split)
        (raw / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {len(rows)} rows -> {raw / (split + '.jsonl')}")
        if split == "original":
            gold = TROY_ROOT / ".cache" / "gold_patches"
            gold.mkdir(parents=True, exist_ok=True)
            for r in rows:
                (gold / f"{r['instance_id']}.diff").write_text(r["patch"])
    fetch_swebench(TROY_ROOT / ".cache" / "swebench")
    from pp.task_environments import FROZEN_SPEC_PATH, derive_spec
    specs = {iid: derive_spec(iid).to_dict() for iid in INSTANCES}
    FROZEN_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_SPEC_PATH.write_text(json.dumps({"source": "scripts/prepare_troy_data.py",
                                            "environments": specs}, indent=1))
    for iid, s in specs.items():
        print(iid, s["image"], s["test_files"])


if __name__ == "__main__":
    main()
