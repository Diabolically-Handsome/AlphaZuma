# AlphaZuma · Zuma RL

**A fidelity-first reinforcement-learning research environment for the Steam PC release of *Zuma's Revenge*, and the AlphaZuma-55 learned-speedrun program built on it.**

> 中文完整版(保真度门槛、证据链、常量与验收细节):[README.zh-CN.md](README.zh-CN.md)

Everything here is organised around one rule: **the environment is validated against the original game before any result is allowed to count, and every result carries a receipt.** Published evidence lives at [aispeedrun.ai](https://aispeedrun.ai) (source: [Diabolically-Handsome/aispeedrun](https://github.com/Diabolically-Handsome/aispeedrun)).

## Highlights

- **Tick-level simulator, validated tick-for-tick on the frozen Jungle2 evidence set.** `ZumaRevenge-v0` advances the game at the retail 100 Hz logic tick over the *original* level data, loaded read-only from the user's own Steam install (CURV v12–15 parser; 226 curve files audited, all 79 listed levels / 89 referenced curves load). Physics accumulate in `float32` in the original operation order; the retail global MT19937 (positive 31-bit output), MSVC CRT `rand` state transition and shooter QRand are reproduced. Three original trajectories without shooter synchronisation match tick for tick; a 501-tick Jungle2 memory trajectory satisfies the per-update contract on 501/501 samples and reconstructs 1,025 global MT advances exactly.
- **Two gates, both `OPEN`.** Fidelity Gate v4 recomputes 97 frozen evidence items live (four lanes PASS, 31 fixed requirements, no rejected evidence). Training Gate v3 authorises exactly one bounded stage (98,304 effective steps, Jungle2, fixed recipe) and binds the simulator and policy source closures by hash. Verdicts are `PASS` / `FAIL` / `INCOMPARABLE` and fail closed. The current authorisation is scoped to Jungle2 state-policy calibration (policy `original-transfer-jungle2-v4`); it does not authorise visual training or original-client deployment, and dual-curve and moving-frog levels are implemented with unit tests but not gate-certified.
- **Original-client evidence pipeline.** PC Golden Manifest v4 with content-addressed traces, read-only PopCap DMO input-replay decoding, lossless DXGI capture bound to process/window/display, memory-state decoding from raw bytes, double-replay determinism and save-transaction proofs. See [docs/PC_GOLDEN.md](docs/PC_GOLDEN.md) and [docs/PC_EVIDENCE_V4_PLAN.md](docs/PC_EVIDENCE_V4_PLAN.md).
- **AlphaZuma-55.** A learned-speedrun program over a 55-level Adventure panel (five moving-frog levels excluded) under an explicit elite-human actuator contract (120 ms reaction delay, 1080°/s aim slew, 18,000°/s² aim acceleration, 50 ms minimum button interval). A scripted, curve-aware teacher wins 55/55 under that contract only because it parks its aim and withholds the fire edge until the cursor has settled (18/55 without the settle gate); that motor program was promoted into a park-and-settle macro interface for students, which are trained by behaviour cloning from the teacher (DAgger where cloning alone failed) and then PPO under a win-first speedrun reward, with champions selected by expected time `E[t] = median winning ticks / win rate` at a win-rate floor.
- **Published records.** *AlphaZuma V1* (2026-08-13): 14 blind-seed individual-level runs in the deterministic simulator, each with checkpoint hash, trajectory hash and accepted replay; 4 are numerically below the currently displayed human leaderboard times; **0 original-client world-record claims** — the timing domains are not comparable and the release says so. Four further golden-seed (seed-enumerated) tool-assisted runs — Jungle1 5.30 s, Jungle2 7.07 s, Jungle6 15.46 s, Jungle10 16.29 s against displayed human RTA times of 0:06 / 0:12 / 0:18 / 0:25 — are labelled AI-TAS (`seed_list_tas`): simulator timing, RTA boundary audit pending, no original-client record claim. Receipts: `outputs/aispeedrun-alpha-v1/` and `outputs/alphazuma-v1-*.json` (release manifest in the aispeedrun repo), `diagnostics/alphazuma-55-jungle*-speedrun-record-*.json`.
- **Preregistered, receipted, honest by construction.** Experiments are preregistered: a frozen preregistration JSON and an outcome JSON with SHA-256s in `diagnostics/` (349 preregistration files at the time of writing). Teacher-assisted or recovery-mode wins count for nothing toward autonomy tiers; only student-only outcomes are scored. The 51 levels not yet won autonomously (46 of the 55-level panel plus 5 moving-frog levels the simulator currently refuses) have a per-level cause localisation; a validated remedy ladder covers the two dominant cause classes (weak initialisation, long-horizon drift) but not the interface-ceiling or dual-chain classes: [`diagnostics/alphazuma-55-unconquered-cause-localization-20260901-VERDICT.md`](diagnostics/alphazuma-55-unconquered-cause-localization-20260901-VERDICT.md).

## Repository layout

| Path | What it holds |
| --- | --- |
| `src/zuma_rl/` | The package (76 modules): simulator core, original-data parsers, Gymnasium environments, DMO decoder, `pc_*` evidence and audit modules, fidelity/training gates, elite-human actuator and park-and-settle wrappers, AlphaZuma-55 constants, training entry point |
| `tools/` | ~490 campaign scripts (454 Python, 33 shell, 5 PowerShell): build / audit / run / watch / distill / materialise AlphaZuma stages; collect / verify / compare PC evidence |
| `tests/` | 221 test files, ~1,750 test functions |
| `docs/` | Fidelity Gate, Fidelity Suite, Training Gate, PC Golden, PC Evidence v4, PC Capture Campaign, PC Mechanism Audit, Multi-Curve, Source Guidance, Windows capture resume |
| `diagnostics/` | Preregistrations, outcomes, verdicts, record-claim receipts and addenda (SHA-256-linked) |
| `outputs/` | AlphaZuma V1 release receipts and posters; the V1-vs-human Adv 17 analysis (replay videos are hosted in the aispeedrun repo) |

Not in this repository: training artefacts (`D:\ZumaTraining`), the content-addressed fidelity evidence store (`ZumaGolden`), retail-game captures and save backups, virtual environments, and third-party reference code.

## Two environments

| Environment | Purpose | Usable as evidence for original-transfer training? |
| --- | --- | --- |
| `ZumaRevenge-v0` | High-fidelity mainline: reads original CURV / level data, 100 Hz tick simulation | Only under the recipe fixed by Training Gate v3 |
| `ZumaSimple-v0` | Early simplified prototype for interface, algorithm and toolchain reference | No |

Win rates or step counts on `ZumaSimple-v0` are never read as pre-training results for the original game.

## Quick start

Python 3.11 or 3.12 in a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'        # add [pc-video] for golden-case video verification, [train] for PPO
```

The environment needs a **legally owned copy of *Zuma's Revenge***. The Steam library is auto-detected; otherwise point `ZUMA_REVENGE_ROOT` at the directory containing `ZumasRevenge.exe`, `main.pak` and `levels/`. No original art, audio or level files are copied or redistributed by this project.

```bash
export ZUMA_REVENGE_ROOT="/path/to/SteamLibrary/steamapps/common/Zuma's Revenge"
zuma-audit-original --level Jungle1        # confirm the parser sees the expected install and level
python -m pytest -q                        # audit tests that need the install are skipped when it is absent
```

Gymnasium smoke test (API and data flow only, not formal training):

```python
import gymnasium as gym
import zuma_rl  # registers ZumaRevenge-v0

env = gym.make("ZumaRevenge-v0", level_id="Jungle1")
obs, info = env.reset(seed=42)
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The action space is `MultiDiscrete([3, 180])`: verb (0 wait/aim, 1 fire, 2 swap) × aim bin. By default one action advances exactly **one** native tick, so a policy can re-aim at every 10 ms input boundary; `frame_skip > 1` is a throughput/diagnostic setting, not an equivalent of original input. The default observation is the *actor view*: balls inside tunnels are hidden, internal timers are exposed only with an explicit `privileged_debug` flag, and debug rendering is a separate visualisation with no original assets.

## Evidence and gates

| Command | Role |
| --- | --- |
| `zuma-audit-dmo <file.dmo>` | Read-only decode of a PopCap demo file: framework seed, update timing, ordered mouse/keyboard input (keyboard values redacted by default) |
| `zuma-verify-pc-golden <manifest.json>` | Single-case acceptance; exit code 0/1/2 = `PASS`/`FAIL`/`INCOMPARABLE` |
| `zuma-verify-fidelity-suite <suite.json>` | Multi-case Fidelity Gate over frozen evidence; derives mechanisms from report bodies, not hand-written labels |
| `zuma-verify-training-suite <suite.json>` | Training Gate: recipe, source-closure hashes and determinism KATs |
| `zuma-audit-actor-visual`, `zuma-audit-pc-mechanisms`, `zuma-verify-pc-capture-campaign`, `zuma-verify-pc-memory-followup` | Visual-recoverability audit (28 of 72 actor features proven so far — the rest return `INCOMPARABLE`), machine mechanism audit, capture-campaign and memory follow-up verifiers |

A single-case `PASS` certifies only that case (for example the 78-tick firing clip `jungle2_dmo_e43c645a18d7_u7693_7770`, 62 artefacts). It does not certify the simulator and it does not open a gate. Formats and hard-coded conditions: [docs/FIDELITY_SUITE.md](docs/FIDELITY_SUITE.md), [docs/TRAINING_GATE.md](docs/TRAINING_GATE.md), [docs/PC_MECHANISM_AUDIT.md](docs/PC_MECHANISM_AUDIT.md).

## Training

`zuma-train` refuses to run unless a mode is chosen explicitly. `--acknowledge-fidelity-gate` enters a diagnostic, non-transferable mode hard-capped at 100,000 timesteps (the cap cannot be waived); `--transfer-training` proceeds only when both suites actually return `OPEN` and the command matches the frozen environment, model, seed, PPO recipe and 98,304-step stage field for field. Every `config.json` records the gate state, suite SHA-256s, recipe hash, library versions and the actual sampled budget.

Measured on the development box (WSL2, Threadripper 7970X): 32 asynchronous `ZumaRevenge-v0` environments at `frame_skip=1` with random actions ≈ 3,244 native ticks/s; a 98,304-step PPO diagnostic segment ≈ 60 s (≈ 1,600 steps/s). Game logic runs on the CPU; the GPU only accelerates the policy network. Benchmark with `zuma-benchmark-revenge`.

## Conventions this project keeps

1. Discrete events (hit, merge start, insertion, explosion, ball removal, suck-back contact, win/loss trigger) must match the original tick for tick; "close on average" is not a pass.
2. All tick-critical arithmetic uses the original `float32` sequence and comparison order — no closed-form shortcuts, no silent `float64`.
3. Agent observations may only contain what a player can see on the original screen; privileged state stays isolated.
4. Steam installation and save files are read-only; nothing from the original game is copied into this repository.
5. Every replay declares its save/tutorial state (`profile_mode="tutorials_completed"` is the only supported contract today).
6. Results are reported wins first, winning ticks second; score is only a liveness diagnostic. Speedrun timing in the simulator is never presented as original-client RTA timing.

## Third-party material and references

- Human leaderboard times quoted in receipts come from [speedrun.com](https://www.speedrun.com/zumas_revenge/levels) (via its REST API where the receipt records it) and are context only; the displayed seconds are not a shared timing domain with simulator ticks.
- Readable reference sources for mechanism hypotheses (a WP7/XNA port of the game and the [alula/CircleShootApp](https://github.com/alula/CircleShootApp) *Zuma Deluxe* decompilation) are kept under `external/` locally and are **not** committed here; original-source evidence outranks them (see the C/S/U evidence grades in [docs/FIDELITY.md](docs/FIDELITY.md)).
- *Zuma's Revenge* is © PopCap Games / Electronic Arts. This is an independent research project with no affiliation.

## License

MIT — see [LICENSE](LICENSE). The license covers this project's code and documents only; *Zuma's Revenge* itself, its data files and any original-client material remain the property of their owners and are not distributed here.

## Author

Diabolically-Handsome — [github.com/Diabolically-Handsome](https://github.com/Diabolically-Handsome). Companion project: [AlphaDiablo / DiabloGym](https://github.com/Diabolically-Handsome/AlphaDiablo), a deterministic Diablo I RL environment.
