# CircleShootApp source-guidance policy

## Purpose

CircleShootApp is a reconstruction of **Zuma Deluxe 1.0.0**, not the source
code of **Zuma's Revenge**.  It is nevertheless a high-value ancestor source:
class boundaries, update ordering, list ownership, and many mechanism shapes
are close enough to turn blind reverse engineering into targeted questions.

The project therefore uses this authority order:

1. Pinned retail Zuma's Revenge runtime and natural retail trajectories.
2. Installed Revenge assets and decoded CURV/level data.
3. CircleShootApp as an ancestor hypothesis generator.
4. Simulator behavior only after it is checked against the target evidence.

CircleShootApp may suggest where and what to inspect.  It cannot overrule a
confirmed difference in the Revenge runtime, and source guidance by itself
does not earn Fidelity Gate credit.

## Pinned source snapshot

- Repository: `https://github.com/alula/CircleShootApp.git`
- Commit: `165d0fd30d977da7ad5ee6efe128d8af2178713b`
- Tree: `22f9cb1dac19cc491bb1ec582cff7d51348e3a11`
- Local state: detached HEAD, clean worktree, stored under ignored
  `external/CircleShootApp/`
- Assets: absent by upstream design; no original assets are copied into this
  project.

The audit pins the README, licenses, and the core `Board`, `CurveMgr`, `Ball`,
`Bullet`, `Gun`, `LevelParser`, `DataSync`, `CurveData`, and `WayPoint` files by
size and SHA-256.  The game-source licensing statement, framework license, and
third-party licenses remain distinct.  The checkout is used as an internal
reference and is not redistributed by this project.

## Confirmed target-runtime decisions

### Gap-shot proximity

CircleShootApp's `CurveMgr::CheckGapShot` at `CurveMgr.cpp:460` uses:

- waypoint sample stride: `2r`;
- curve proximity threshold: `(2r)^2`.

The pinned Zuma's Revenge runtime differs.  Function `0x0045C500`:

- loads the projectile radius from offset `+0x38`;
- converts it to an integer;
- stores `2r` as the waypoint-loop step;
- independently squares `r` and uses `r^2` for both the latched-point and
  candidate-sample distance comparisons;
- has one direct call site at `0x004180CE` in the Board projectile loop.

Consequently, the existing simulator and mechanism auditor were correct to
use a `2r` sample stride with an `r^2` threshold.  Copying the Deluxe source
verbatim would have introduced a 2x radial / 4x area false-positive region.
The added regression checks a projectile 27 px from the curve with radius 18:
Deluxe would accept it, while Revenge correctly rejects it.

### Pending-color rejection and RNG order

CircleShootApp's `CurveMgr::AddPendingBall` at `CurveMgr.cpp:1281` performs the
ball visual-frame draw before choosing a color, and its percentage-repeat
branch has no maximum-clump check.

The pinned Revenge function at `0x00458B00` instead:

- draws the repeat roll from the global MT stream and repeats only when
  `roll <= repeat_chance` **and** `current_run < max_clump`;
- otherwise calls CurveManager vtable slot `+0xA4`, statically bound to
  `0x004B4B70`, which returns global-MT output modulo the active color count;
- loops at `0x00458D40` while the candidate equals the previous color;
- commits the accepted color and only then calls `0x004020D0` to consume the
  rendering-only visual-frame draw, before inserting the ball into the pending
  list.

The simulator already follows this Revenge-specific order.  Regressions now
distinguish both deltas: a 100% repeat roll cannot exceed `max_clump`, and the
visual draw must occur after all rejected and accepted color candidates.

### Rollback direction, speed scale, and update order

CircleShootApp supplies the correct broad topology: a removed matching set
starts a suck-back at count 10, contact can seed a 30-tick rollback at
`max(combo * 1.5, 0.5)`, and entrance-side deletion enforces a 40-tick stop.
It does not have Revenge's direction flag, uses `suck_count / 8` without a
curve scale, and hardcodes the global reverse seed to speed 1.0.

The pinned Revenge runtime splits the two directions explicitly:

- `0x0045A8F0` dispatches on Ball offset `+0xC2`;
- ordinary gap rollback uses direction 1 and moves by
  `(suck_count >> 3) * curve_reverse_speed`;
- loss suction uses direction 0 through helper `0x0045A570`;
- `0x0045A230` seeds the skull-side ball with the same curve reverse-speed
  field, propagates movement through contacted balls, snaps newly contacted
  balls using both radii, and latches at least 20 stop ticks when movement
  reaches the entrance side;
- `0x0045AE30` explicitly writes direction 1 when set removal starts a new
  suck-back.

The retail caller also fixes the relevant order as suck update, normal advance,
backward advance, front/end cleanup, set update, then power-up update.  The
simulator already followed that order, but it had omitted the reverse-speed
multiplier during ordinary suck-back and relied on a default direction value
after set removal.  Both are now explicit, with a non-1.0 speed regression so
the former cannot hide behind the default again.

### Natural terminal predicates

The target `Curve::IsLosing` at `0x0045C7F0` and its Board call at
`0x0041B010` require all of the following: no active exploding set, a nonempty
chain, a lethal endpoint curve, the skull-side ball at or beyond the endpoint,
no curve bullet, no positive global reverse counter, and no suck counter on
the skull-connected front segment.  The Board quantifier is “any curve”.

The target `Curve::IsWinning` at `0x0045C8A0` requires empty active and pending
lists plus absence of the special/boss actor at manager offset `+0x1B0`.  The
Board inlines that predicate at `0x0041A1F1` and requires it for every curve.
The simulator's normal non-boss predicates align with these conditions.  Boss
actor completion is intentionally still unmodelled, and the observed two-tick
Board victory transition remains trajectory-bound rather than being inferred
from this static predicate alone.

### Ball power-up effect dispatch

CircleShootApp exposes the ancestor effect topology in
`CurveMgr::ActivatePower`, `CheckSet`, `StartClearCount`, and `ActivateBomb`.
Each ball in a matched span enters `StartClearCount` separately; that method
updates the last-cleared waypoint before dispatching the ball's power-up. Bomb
recursion follows the same path. Deluxe uses physical-collision pad 45.

The pinned Revenge runtime confirms the topology but changes and extends the
contract:

- `CurveMgr::ExplodeBall` at `0x00457680` rejects an already-exploding ball, then
  converts Ball `+0x1C` and writes Curve `+0x1B4` on every newly exploding
  ball, before resolving or dispatching its power-up;
- effective type priority is primary, then a still-live previous type, then
  the secondary/destination type;
- `Board::ActivatePower` at `0x00418720` visits every curve and calls
  `CurveMgr::ActivatePower` at `0x00457150`;
- bomb type 0 walks each active-ball list, skips balls already exploding, uses
  physical-collision pad 56, and recursively calls `ExplodeBall`;
- reverse type 3 writes 300 ticks only when the active list is nonempty;
- slow type 1 replaces counters below 1000 with 800 ticks;
- each trigger increments the per-type counter, records the Board game time as
  its cooldown timestamp, and sets the Curve trigger latch.

The simulator already matched the target constants and dispatch effects, but
it updated `last_powerup_waypoint` only once from the original match seed. A
middle-seed match or a recursive bomb could therefore leave the wrong final
waypoint. The assignment now lives in `_begin_ball_explosion`, after the
already-exploding guard and before effect resolution, exactly once for every
new explosion. Regressions cover both a middle-seed match and ordered bomb
recursion.

This proof deliberately does not claim the Revenge spawn scheduler or fruit
collision path. Those remain trajectory-bound Fidelity requirements.

The current immutable machine report is
`diagnostics/circleshoot-revenge-source-guidance-v3.json`.  It binds the source
commit, source-file hashes, runtime SHA-256, exact instruction blocks, direct
call xrefs/call routes, and normalized Python AST contracts for gap shot,
pending-color RNG, rollback, ball power-up effects, and normal non-boss
terminal predicates. The older v1 and v2 reports remain immutable historical
receipts.

Recreate it from an absent output path with:

```powershell
python tools/audit_circleshoot_revenge_alignment.py `
  --circleshoot-root external/CircleShootApp `
  --runtime-executable D:/ZumaGolden/tools/direct-runtime/popcapgame1.exe `
  --simulator-source src/zuma_rl/revenge_core.py `
  --mechanism-audit-source src/zuma_rl/pc_mechanism_audit.py `
  --output diagnostics/circleshoot-revenge-source-guidance-v3.json
```

## Mechanism routing

| Missing Fidelity v4 mechanism | Ancestor starting point | Revenge implementation | Route |
| --- | --- | --- | --- |
| `gap_shot` | `CurveMgr.cpp:460` | `revenge_core.py:_check_gap_shot` | Static target delta resolved; natural retail event and simulator diff still required. |
| `natural_loss` | `CurveMgr.cpp:546`, Board state machine | loss-suction methods | Static target predicate and Board quantifier recovered; natural loss trajectory and simulator diff still required. |
| `natural_win` | `CurveMgr.cpp:569`, Board state machine | terminal/Zuma state | Static non-boss predicate recovered; boss actor gate remains unmodelled and natural transition timing still requires a source-bound clear. |
| `powerup_proximity_bomb` | `CurveMgr.cpp:619,1935` | `_trigger_ball_powerup` | Static target active-chain traversal, pad 56, recursive explosion, and waypoint order resolved; natural event/diff and fruit interaction still required. |
| `powerup_reverse` | `CurveMgr.cpp:619` | `_trigger_ball_powerup` | Static target nonempty-list guard and 300-tick assignment resolved; natural activation and per-tick chain diff still required. |
| `powerup_spawn` | pending-ball/power-up paths | `_maybe_spawn_powerup` | Verify target eligibility, cooldown, and RNG call order. |
| `rng_rejection` | `GetRandomPendingBallColor` and `AddPendingBall` | `_append_pending_color` | Static target deltas and RNG order resolved; a natural rejected-candidate event and simulator diff still remain. |
| `rollback_chain` | `CurveMgr.cpp:1517,1670` | backward/set update methods | Static target core recovered; speed scale and direction fixed, while a natural scored rollback and simulator diff still remain. |
| `tunnel_collision` | `CurveMgr.cpp:387` | `_try_projectile_collision` | Verify target front/back tunnel sample and collision ordering. |
| `zuma_transition` | `SetStopAddingBalls` and Board state | `_update_zuma_bar` | Verify target threshold tick, pending deletion, backward/slow counters, and event order. |

The six fruit mechanisms have no Deluxe equivalent and remain Revenge-only
work.  CircleShootApp must not be used to fabricate fruit scheduling,
collision, score, expiry, or animation behavior.

## Build status

The source snapshot is usable for static guidance without compiling it.  The
current Windows host has no modern C++ compiler installed, and the upstream
convenience script downloads an old MSVC 7.0 bundle from an author-operated
server.  That script was not run.  A modern-toolchain build is deferred until
it contributes a specific oracle test; it is not a prerequisite for examining
the source or for validating Revenge against its own pinned runtime.

## Gate boundary

This source-guidance layer prevents known semantic drift and makes future
reverse engineering much faster.  It does **not** change the current Gate:
static machine-code proof is supporting evidence, while each missing Fidelity
mechanism still requires a natural, non-injected retail source window and its
corresponding simulator diff.

The implementation-fingerprint rebind and the current Fidelity/Training Gate
receipts are recorded in
`diagnostics/SOURCE_GUIDANCE_GATE_REBIND_2026-08-09.md`.  The rebind preserves
policy generation 4 and leaves the current Fidelity result at 18 requirements
passed and 13 missing; training remains closed on that Fidelity result alone.
