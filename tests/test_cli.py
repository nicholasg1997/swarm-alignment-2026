import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    return subprocess.run([sys.executable, "-m", "swarmlab", *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_cli_fake_run_creates_report(tmp_path):
    p = _run("run", "--condition", "C1", "--block", "0", "--seed", "0",
             "--model", "configs/models/scripted_demo.yaml", "--env", "fake",
             "--tasks", "f1,f2,f3,f4,f5,f6", "--focal-slot", "0", "--runs-dir", str(tmp_path))
    assert p.returncode == 0, p.stderr
    run_dir = Path(p.stdout.strip().splitlines()[-1])
    assert (run_dir / "events.jsonl").exists() and (run_dir / "index.html").exists()
    assert (tmp_path / "index.html").exists()
    ev = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "violation" and e["actor"] == "a2" for e in ev)


def test_cli_panel_matched_blocks(tmp_path):
    p = _run("panel", "--conditions", "C0,C1", "--blocks", "0-1",
             "--model", "configs/models/scripted_demo.yaml", "--env", "fake",
             "--tasks", "f1,f2,f3,f4,f5,f6", "--runs-dir", str(tmp_path))
    assert p.returncode == 0, p.stderr
    manifests = [json.loads(m.read_text()) for m in tmp_path.glob("*/run.json")]
    assert sorted((m["condition"], m["block"]) for m in manifests) == [
        ("C0", 0), ("C0", 1), ("C1", 0), ("C1", 1)]
    for b in (0, 1):
        assert {m["focal_slot"] for m in manifests if m["block"] == b} == {b % 6}


def test_cli_report_only(tmp_path):
    _run("run", "--condition", "C0", "--model", "configs/models/scripted_demo.yaml",
         "--env", "fake", "--tasks", "f1,f2,f3,f4,f5,f6", "--runs-dir", str(tmp_path))
    (tmp_path / "index.html").unlink()
    assert _run("report", str(tmp_path)).returncode == 0
    assert (tmp_path / "index.html").exists()


def test_cli_board_mode_and_team_framing_flags(tmp_path):
    p = _run("run", "--condition", "C1", "--model", "configs/models/scripted_demo.yaml",
             "--env", "fake", "--tasks", "f1,f2,f3,f4,f5,f6", "--runs-dir", str(tmp_path),
             "--board-mode", "passive", "--no-team-framing")
    assert p.returncode == 0, p.stderr
    m = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert m["board_mode"] == "passive" and m["team_framing"] is False


def test_cli_cap_flags(tmp_path):
    p = _run("run", "--condition", "C0", "--model", "configs/models/scripted_demo.yaml",
             "--env", "fake", "--tasks", "f1,f2,f3,f4,f5,f6", "--runs-dir", str(tmp_path),
             "--max-turns", "7", "--max-tokens", "123456")
    assert p.returncode == 0, p.stderr
    m = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert m["max_turns_per_agent"] == 7 and m["max_world_tokens"] == 123456


def test_cli_spec_file_custom_world(tmp_path):
    p = _run("run", "--spec", "configs/worlds/trio_with_devil.yaml",
             "--model", "configs/models/scripted_demo.yaml", "--env", "fake",
             "--runs-dir", str(tmp_path))
    assert p.returncode == 0, p.stderr
    m = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert m["condition"] == "trio_devil"
    assert [m["teams"][a] for a in ("a1", "a2", "a3")] == ["trio"] * 3
    assert m["roles"]["a4"] == "devil" and m["env_owner"]["a3"] == "a1"


def test_cli_devil_preset(tmp_path):
    p = _run("run", "--condition", "DV", "--model", "configs/models/scripted_demo.yaml",
             "--env", "fake", "--tasks", "f1,f2,f3,f4,f5,f6", "--runs-dir", str(tmp_path))
    assert p.returncode == 0, p.stderr
    m = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert list(m["roles"].values()).count("focal") == 2 and "devil" in m["roles"].values()


def test_cli_controlled_spec_resolves_options_and_scope(tmp_path):
    import yaml
    spec = yaml.safe_load((ROOT / "configs/worlds/sp_authority_xarray.yaml").read_text())
    spec["options"]["max_turns_per_agent"] = 2
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    p = _run("run", "--spec", str(path), "--scope-rule", "implicit",
             "--model", "configs/models/scripted_demo.yaml", "--env", "fake",
             "--max-turns", "4", "--runs-dir", str(tmp_path / "runs"))
    assert p.returncode == 0, p.stderr
    m = json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text())
    assert m["scope_rule"] == "implicit" and m["max_turns_per_agent"] == 2
    assert m["controlled_stimulus"]["variant"] == "authority"
    assert m["agent_models"]["a5"] == "scripted-confederate-v1"
    assert len(m["resolved_models"]) == 5
