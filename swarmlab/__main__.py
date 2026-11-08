"""CLI: python -m swarmlab {run,panel,report} (build plan Task 6)."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .conditions import CONDITIONS, Slot, WorldConfig
from .models import load_dotenv, load_model_config, make_backend_factory
from .report import render_dashboard, render_run_page
from .world import run_world


def _sandbox_timeout(wall_clock_s: int) -> int:
    """Sandboxes must outlive the world (setup and scoring happen outside the cap)."""
    return wall_clock_s + 1800


def _env_factory(kind: str, wall_clock_s: int = 3600, scope_rule: str = "explicit", answer_key=None, seed: int = 0):
    if kind in ('fake', 'modal'):
        def assessment_or_task(task_id, variant):
            if task_id == 'assessment-release-v1':
                from .assessment import AssessmentEnv, ModalAssessmentEnv
                cls = AssessmentEnv if kind == 'fake' else ModalAssessmentEnv
                extra = {} if kind == 'fake' else {'timeout_s': _sandbox_timeout(wall_clock_s)}
                return cls(task_id, variant, seed=seed, answer_key=answer_key, **extra)
            if kind == 'fake':
                from .fake_env import FakeTaskEnv
                return FakeTaskEnv(task_id, variant)
            from .troy_env import TroyTaskEnv
            return TroyTaskEnv(task_id, variant, timeout_s=_sandbox_timeout(wall_clock_s), scope_rule=scope_rule)
        return assessment_or_task
    raise SystemExit(f"unknown --env {kind!r}")


def _blocks(spec: str):
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def _load_spec(path):
    """Custom world YAML: condition label, optional board flag, and 6 slots."""
    doc = yaml.safe_load(Path(path).read_text())
    slots = tuple(Slot(s["task"], s.get("variant", "original"), s.get("role", "peer"),
                       s.get("team")) for s in doc["slots"])
    return doc["condition"], slots, doc.get("board"), dict(doc.get("options") or {})


def _one(args, condition: str, block: int, focal_slot: int, angel_slot,
         slots=None, board=None, options=None) -> Path:
    """`options` (from a world spec) set WorldConfig fields and override the CLI, so a
    spec file fully pins its cell (e.g. scope_rule, confederate_message)."""
    model_cfg = load_model_config(args.model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    settings = dict(run_id=f"{condition}-b{block}-s{args.seed}-{stamp}", condition=condition,
                      block=block, seed=args.seed, model=model_cfg.get("name", args.model),
                      task_ids=([s.task_id for s in slots] if slots else args.tasks.split(",")),
                      focal_slot=focal_slot, slots=slots, board=board,
                      angel_slot=(angel_slot if condition in CONDITIONS
                                  and CONDITIONS[condition].angel else None),
                      board_mode=args.board_mode, team_framing=not args.no_team_framing,
                      offer_style=args.offer_style,
                      autonomy_framing=not args.no_autonomy_framing, stakes=args.stakes,
                      helper_framing=not args.no_helper_framing,
                      wake_on_mention=not args.no_wake_on_mention,
                      check_every=args.check_every,
                      carry_offer_reasoning=not args.no_carry_offer_reasoning,
                      private_messages=args.private_messages,
                      memory_from=args.memory_from, memory_enabled=not args.no_memory,
                      memory_form=args.memory_form, memory_max_chars=args.memory_max_chars,
                      change_notices=args.change_notices,
                      max_turns_per_agent=args.max_turns, max_world_tokens=args.max_tokens,
                      max_wall_clock_s=args.max_wall_clock, scope_rule=args.scope_rule)
    options = options or {}
    # Specs pin experimental settings, but cannot replace run identity or layout.
    reserved = {"run_id", "condition", "block", "seed", "model", "task_ids",
                "focal_slot", "angel_slot", "slots", "board"}
    if reserved.intersection(options):
        raise ValueError(f"reserved world options: {sorted(reserved.intersection(options))}")
    cfg = WorldConfig(**{**settings, **options})
    res = run_world(cfg, make_backend_factory(model_cfg),
                    _env_factory(args.env, cfg.max_wall_clock_s, cfg.scope_rule, cfg.answer_key, cfg.seed),
                    Path(args.runs_dir))
    render_run_page(res.run_dir)
    render_dashboard(Path(args.runs_dir))
    return res.run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="swarmlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "panel"):
        p = sub.add_parser(name)
        p.add_argument("--memory-from", help="episode-1 run directory")
        p.add_argument("--no-memory", action="store_true", help="matched continuation control")
        p.add_argument("--memory-form", choices=("summary", "transcript"), default="summary")
        p.add_argument("--memory-max-chars", type=int, default=24000)
        p.add_argument("--change-notices", action="store_true")
        p.add_argument("--model", required=True)
        p.add_argument("--env", choices=("fake", "modal"), default="fake")
        p.add_argument("--tasks", help="six comma-separated task ids (not needed with --spec)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--board-mode", choices=("offer", "passive"), default="offer")
        p.add_argument("--no-team-framing", action="store_true")
        p.add_argument("--offer-style", choices=("notification", "pause"),
                       default="notification")
        p.add_argument("--no-autonomy-framing", action="store_true")
        p.add_argument("--no-helper-framing", action="store_true",
                       help="drop the 'hand work to teammates like sub-agents' line")
        p.add_argument("--stakes", choices=("normal", "high"), default="normal")
        p.add_argument("--scope-rule", choices=("explicit", "soft", "implicit"),
                       default="explicit",
                       help="soft = pressure + a vague norm; implicit = no test rule (grey zone)")
        p.add_argument("--no-wake-on-mention", action="store_true")
        p.add_argument("--check-every", type=int, default=5,
                       help="finished agents check the chat every N rounds (0 = never)")
        p.add_argument("--no-carry-offer-reasoning", action="store_true")
        p.add_argument("--private-messages", action="store_true",
                       help="let agents send private messages to named teammates")
        p.add_argument("--max-turns", type=int, default=30, help="turns per agent")
        p.add_argument("--max-tokens", type=int, default=15_000_000, help="per world")
        p.add_argument("--max-wall-clock", type=int, default=3600,
                       help="seconds per world; sandboxes live 30 min longer")
    run = sub.choices["run"]
    run.add_argument("--condition", choices=sorted(CONDITIONS))
    run.add_argument("--spec", help="custom world YAML (configs/worlds/*.yaml)")
    run.add_argument("--block", type=int, default=0)
    run.add_argument("--focal-slot", type=int, default=0)
    run.add_argument("--angel-slot", type=int, default=None)
    panel = sub.choices["panel"]
    panel.add_argument("--conditions", required=True, help="e.g. C0,C1,B0")
    panel.add_argument("--blocks", required=True, help="e.g. 0-7")
    rep = sub.add_parser("report")
    rep.add_argument("runs_dir")
    args = ap.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    if args.cmd == "report":
        runs = Path(args.runs_dir)
        for d in sorted(p for p in runs.iterdir() if (p / "run.json").exists()):
            render_run_page(d)
        print(render_dashboard(runs))
        return 0
    if args.cmd == "run":
        if args.spec:
            cond, slots, board, options = _load_spec(args.spec)
            print(_one(args, cond, args.block, args.focal_slot, None, slots, board, options))
            return 0
        if not args.condition or not args.tasks:
            ap.error("run needs --condition and --tasks, or --spec")
        angel = args.angel_slot if args.angel_slot is not None else (args.focal_slot + 1) % 6
        print(_one(args, args.condition, args.block, args.focal_slot, angel))
        return 0
    if not args.tasks:
        ap.error("panel needs --tasks")
    for block in _blocks(args.blocks):  # matched blocks: same tasks/focal/seed per block
        focal = block % 6
        for condition in args.conditions.split(","):
            print(_one(args, condition, block, focal, (focal + 1) % 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
