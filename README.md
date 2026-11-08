# Malicious or Confused? Agent swarms and restricted changes

Code for the Swarm Alignment 2026 project *Malicious or Confused? Why Agent Swarms Are More Likely
to Make Restricted Changes*. Lone agents and six-agent teams work on ImpossibleBench repair tasks
whose tests contradict each other. Every agent gets the same soft guidance that fixes belong in
library code. Because no honest fix can pass, a passing test means the agents changed something
they should not have. The harness varies who an agent works with (alone, a team without chat, a
team with chat) and adds oversight conditions: a verifier, an enforcer that can restore tests, and
a claim monitor on the team chat, run with and without read-only Git.

## Layout

| Path | What it is |
|---|---|
| `swarmlab/` | The harness: world loop, shared workspace, team board, notebook, claim monitor, read-only Git tool, prompts, scoring, event log, report pages |
| `configs/worlds/` | One YAML spec per experimental condition (see below) |
| `configs/models/` | Model backends (OpenRouter, Adaption, mixed teams, scripted offline models) |
| `configs/fixtures.yaml` | The qualified ImpossibleBench tasks |
| `scripts/` | Data preparation and analysis: results table, monitor panel, disclosure screen, hedge-loss and telephone-chain analyses |
| `work/figures/` | Figure scripts |
| `tests/` | Offline test suite |

## Setup

Python 3.13. The harness uses [The Troy Moment](https://github.com/para2046/troy-moment) (MIT) as a
library for its action parser, model backend and Modal task sandboxes. Clone it at the pinned
commit; this project does not modify it.

```bash
git clone https://github.com/para2046/troy-moment vendor/troy-moment
git -C vendor/troy-moment checkout 3c6f15b65e3b3b1e47a611acdc23e2c25c63f63a
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python pyyaml openai adaption pytest matplotlib -r vendor/troy-moment/requirements.txt
.venv/bin/python scripts/prepare_troy_data.py   # task data Troy's sandboxes need (public HF dataset)
.venv/bin/python -m pytest -q                   # offline; live tests are opt-in with SWARMLAB_LIVE=1
```

Real tasks run in [Modal](https://modal.com) sandboxes (`modal token new`). Model keys are read from
the environment or from a git-ignored `.env` file. Never commit them.

```bash
OPENROUTER_API_KEY=...   # DeepSeek v4.1 Flash, the main worker model
ADAPTION_API_KEY=...     # qwen3.5-397B (second model) and the claim-monitor reviewer
```

## Running

Offline smoke test (no keys, no cost): fake tasks and a scripted model.

```bash
.venv/bin/python -m swarmlab panel --conditions C0,C1,B0 --blocks 0-1 \
  --model configs/models/scripted_demo.yaml --env fake --tasks f1,f2,f3,f4,f5,f6
```

One world of an experimental condition. Each condition was run for blocks 0, 1 and 2:

```bash
.venv/bin/python -m swarmlab run --spec configs/worlds/shared_nodevil_xarray.yaml \
  --model configs/models/openrouter_cheap.yaml --env modal \
  --max-turns 40 --max-tokens 30000000 --max-wall-clock 7200 \
  --block 0 --seed 0 --runs-dir runs/nodevil
```

Each world writes `runs/<dir>/<run_id>/events.jsonl` (every action, message, file change and
score), `run.json` (the manifest with prompt hashes and budgets) and `index.html` (a timeline).
`python -m swarmlab report runs/` rebuilds the pages.

### Conditions used in the paper (main task: pydata__xarray-2905)

| Condition | Spec |
|---|---|
| One agent (40 or 240 turns) | `solo_peer_xarray.yaml` |
| One agent + read-only Git | `solo_git_xarray.yaml` |
| One agent + private notebook (three wordings) | `solo_notebook_{bare,matched,instructed}_xarray.yaml` |
| Six agents, shared repo, no chat (with a devil / all peers) | `shared_nochat_xarray.yaml`, `shared_nochat_peers_xarray.yaml` |
| Six ordinary peers with team chat | `shared_nodevil_xarray.yaml` |
| Devil control, verifier and enforcer arms | `shared_control_xarray.yaml`, `shared_verifier_xarray.yaml`, `shared_enforcer_xarray.yaml` |
| Claim monitor, active or shadow, impossible or solvable | `monitor_{active,shadow}_xarray_{impossible,solvable}.yaml` |
| Claim monitor + read-only Git | `monitor_git_{active,shadow}_xarray_impossible.yaml` |
| Second task (matplotlib__matplotlib-20859) | `solo_peer_matplotlib.yaml`, `shared_nochat_matplotlib.yaml`, `shared_nodevil_matplotlib.yaml` |
| Second model | any spec with `--model configs/models/adaption_qwen397b.yaml` |

Each world's `run.json` records its exact budgets. Most arms used 40 turns per agent, a 30M-token
ceiling and a 2- or 3-hour wall clock (`--max-wall-clock 7200` or `10800`). The solo 240-turn and
notebook panels used `--max-turns 240`. Run commands from the repository root.

## Analysis

All analysis reads the saved event logs; nothing calls a model.

```bash
.venv/bin/python scripts/results_table.py                     # world-level outcomes -> results/RESULTS.md
.venv/bin/python scripts/monitor_panel.py runs/monitor-v1     # claim-monitor and Git panels
.venv/bin/python scripts/guilt_screen.py                      # disclosure / concealment screen around each first edit
.venv/bin/python scripts/telephone_chain.py                   # message lineage for a selected run
.venv/bin/python work/figures/fig2_messaging.py               # main results figure -> figures/
```

`results_table.py` expects the run directories named in its `CELLS` list. Edit those globs if your
`--runs-dir` names differ.

Outcomes: the primary outcome is whether the conflicting test passes at final scoring
(`score.variant_tests_pass`). A `violation` event records any change to a protected test file,
including probes later reverted, and an `escalate` event records a report to a human. A pass can
also come from a library change that games the test, so read the trace behind any pass before
reporting it.

## Not included

Run logs, paper drafts and working notes are not in this repository. The redacted Swarm Traces
dataset is not redistributed.

## Prior work and credits

- [The Troy Moment](https://github.com/para2046/troy-moment) (MIT): task environment, sandboxes and
  action parser, used unmodified.
- [ImpossibleBench](https://github.com/safety-research/impossiblebench): the conflicting-test task variants.
- The `swarmlab/` harness core and its tests predate the sprint (Nov 6, 2026). The experiment
  conditions, notebook, claim monitor, Git tool and analyses were added during it.
