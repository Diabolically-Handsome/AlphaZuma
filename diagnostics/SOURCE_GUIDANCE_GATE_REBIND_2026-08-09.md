# Source-guidance and Gate rebind receipt - 2026-08-09

## Outcome

The pinned Revenge runtime and the pinned CircleShootApp ancestor now have a
fail-closed static alignment report for gap-shot, pending-color RNG, rollback,
ball power-up effects, and normal non-boss terminal predicates. Three
simulator mismatches exposed by the retail implementation were corrected:

- ordinary suck-back speed is `(suck_count >> 3) * curve_reverse_speed`;
- set removal explicitly selects the backward suck direction.
- every newly exploding ball updates the Curve's last power-up waypoint before
  effect dispatch, including middle-seed matches and recursive bomb clears.

No Gate criterion was relaxed. Static reverse-engineering evidence remains
supporting evidence only and does not replace the required natural retail
source window plus simulator diff.

## Immutable source evidence

- `diagnostics/circleshoot-revenge-source-guidance-v3.json`
  - status: `PASS`
  - SHA-256: `4272e5a1f26ba7efda7511d01718bc82267a380acc923e697c633595c31f1967`
  - scope boundary: spawn scheduling and fruit collision remain unresolved
- `D:/ZumaGolden/diagnostics/actor-audit-jungle2-current-v10.json`
  - status: `PASS` (8/8 checks)
  - SHA-256: `972f375ea0339c5aa877ac581dd714fe068db2c08d09db2071773825d077dc54`
  - implementation fingerprint:
    `sha256:c043faa39381a23624bb4b0b0230c21edf1e28fbef9929ee4f7c49483f2f5660`
  - audit fingerprint:
    `sha256:ebe9cba8076bbc472bfb47f1065fb5ae752eb30ac596522f645ff12e03e0f38f`
  - environment fingerprint:
    `sha256:d97fcb84be32248c3ce3c9ad2584b7b47a8de29ca0bebc45f5100ecbe81d7719`

## Fidelity Gate rebind

- suite: `D:/ZumaGolden/diagnostics/fidelity-suite-c126-candidate-v14.json`
  - 73 content-addressed evidence entries
  - SHA-256: `e79f3d56b2c37c206a2cf40a96c98e9fe07f8830f4bff4504b4da6958f26516a`
- recomputed report:
  `D:/ZumaGolden/diagnostics/fidelity-gate-c126-candidate-v17-report.json`
  - SHA-256: `9ab14104826d821461f5fb473b847da8a254082ff6da630786b9183f70ffeda4`
  - status: `CLOSED`
  - requirements: 18 `PASS`, 13 `MISSING`
  - source authenticity: `PASS`
  - distribution: `PASS`
  - interface: `PASS`
  - dynamics: `MISSING`

The 13 remaining mechanisms are:

`gap_shot`, `natural_loss`, `powerup_proximity_bomb`, `powerup_reverse`,
`powerup_spawn`, `rng_rejection`, `rollback_chain`,
`fruit_scheduler_spawn`, `fruit_expiry`, `fruit_projectile_collision`,
`fruit_powerup_collision`, `fruit_collection_score`, and
`fruit_collection_animation`.

The rebind removed the stale actor-fingerprint failure without changing the
requirement count or policy generation.

## Training Gate rebind

- environment contract v12:
  `D:/ZumaGolden/training/contracts/jungle2-training-environment-v12.json`
  - SHA-256: `36b700eee9ae6c0d8f6dbf2da2fbb1cc08e9443690243e8c8fca49eb0fe6b604`
  - contract fingerprint:
    `sha256:cb9398e7b73fa97791e7dc98db273cb885078e34afbd062272031538bcb2ccaf`
- model migration audit v6:
  `D:/ZumaGolden/training/reports/fruit-actor-model-migration-v6.json`
  - status: `PASS`
  - SHA-256: `0074d3bb5f4ef9b4db29e53482c6a749eb1a76d9a3cc145dc3306d995fe1c6dd`
  - migrated model identity unchanged:
    `sha256:32e1e841962833b3c74ce743598df1c87e21c9c9c931092152e8bc2700081f64`
- Training suite v8:
  `D:/ZumaGolden/training/suites/training-gate-v3-candidate-v8.json`
  - SHA-256: `b37a1d1b1e2493809470da1819f74c902819c5e643f8ee4b7795b29c662beb64`
- recomputed Training report v8:
  `D:/ZumaGolden/training/reports/training-gate-v3-candidate-v8-report.json`
  - SHA-256: `161842a65025fa39bc102602a208dae827fa96babb97eaa1279eb25d49026862`
  - 6/6 evidence files passed content-address verification
  - status: `CLOSED`
  - sole reason: `bound Fidelity Gate is not OPEN on all v4 evidence lanes`

Training, visual-policy training, and original-game deployment therefore remain
unauthorized. No PPO run was started.

## Verification

- Windows full regression: `PASS`; 1,415 tests collected with 13 conditional
  skips and one existing Gymnasium warning.
- WSL focused environment-contract, model-migration, Training Gate, actor-audit,
  and Fidelity tests: `PASS` with one native-platform conditional skip.
- A full WSL collection is not portable because native PC-capture/debugger test
  modules import Windows `kernel32`; the native Windows full suite above is the
  authoritative cross-module regression for those modules.
