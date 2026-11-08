"""Host-owned protected-file snapshot scripts, executed inside the existing sandbox."""
import json


def snapshot_script(repo, paths, snapshot=None):
    # Only host-selected protected paths enter this script, never model arguments.
    payload = json.dumps(dict(repo=str(repo), paths=list(paths), snapshot=snapshot))
    return "python3 - <<'SWARM_SNAPSHOT_PY'\n" + "import json, base64, pathlib\n" + f"p = json.loads({payload!r})\n" + '''root = pathlib.Path(p['repo']).resolve()
result = {}
for name in p['paths']:
    path = root / name
    if pathlib.Path(name).is_absolute() or '..' in pathlib.Path(name).parts:
        raise RuntimeError('unsafe protected path')
    path.parent.resolve().relative_to(root)
    parent = path.parent
    while parent != root:
        if parent.is_symlink():
            raise RuntimeError('protected parent is a symlink')
        parent = parent.parent
    if path.is_symlink():
        raise RuntimeError('protected file is a symlink')
    if p['snapshot'] is None:
        result[name] = base64.b64encode(path.read_bytes()).decode() if path.exists() else None
    else:
        value = p['snapshot'][name]
        if value is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(value))
print('SWARM_SNAPSHOT=' + json.dumps(result))
SWARM_SNAPSHOT_PY
'''
