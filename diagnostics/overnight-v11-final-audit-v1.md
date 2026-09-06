# AlphaZuma V1.1 overnight final audit

This document is a post-hoc audit of the preregistered overnight run. It does
not modify the frozen protocol, model-selection rule, seeds, or Gate decisions.

## Verdict

- Controller status: `COMPLETE`
- Completed: `2026-08-13T15:12:30.728317+00:00`
- Frozen report deadline: `2026-08-13T15:30:00Z`
- Within deadline: yes
- All frozen success Gates passed: **no (6/8 passed)**
- Selected source: `adaptive-mid-13631488`
- Selected model SHA-256:
  `0d2193c864d68644a716711fd77abb553f6727807f7de2f0a81acf7e3bec1579`
- Selected model validation: ZIP clean, 13,631,488 internal timesteps,
  3,326 updates, 24 optimizer states, 321,828 parameters.

The overnight run produced a large and statistically unambiguous improvement
over the frozen baseline, but it did not satisfy the complete campaign Gate.
The two failures were target-level breadth (13/16 versus 14/16 required) and
`village6` (0/16 versus 4/16 required).

## Training and selection

The recovery training ran until the preregistered 09:30 local wall-clock stop:

| Route | New recovery steps | Final internal steps | Episodes | Throughput |
|---|---:|---:|---:|---:|
| Adaptive / RTX 5090 | 24,571,920 | 27,717,648 | 3,331 | 790.1 steps/s |
| Uniform / RTX 5080 | 23,005,136 | 25,102,288 | 3,147 | 739.7 steps/s |
| Total | 47,577,056 | - | 6,478 | - |

The frozen 4-attempt-per-level selection set ranked five candidates as follows:

| Candidate | Levels with a win | Wins |
|---|---:|---:|
| baseline-v1 | 6/17 | 9/68 |
| adaptive-mid-13631488 | **14/17** | **46/68** |
| adaptive-final-27717648 | 13/17 | 45/68 |
| uniform-mid-12582912 | 14/17 | 44/68 |
| uniform-final-25102288 | 12/17 | 42/68 |

The preregistered lexicographic rule therefore selected the adaptive midpoint.
The final checkpoints were not assumed to be best; the observed selection data
showed mild late-training regression.

## Frozen morning blind results

### Sixteen target levels

| Metric | Selected | Baseline | Difference |
|---|---:|---:|---:|
| Wins | 168/256 | 48/256 | +120 |
| Win rate | 65.625% | 18.750% | +46.875 pp |
| Wilson 95% CI | 59.61%-71.17% | 14.44%-23.98% | - |
| Levels with at least one win | 13/16 | 10/16 | +3 |
| Village wins | 67/128 | 5/128 | +62 |
| Median winning time | 44.505 s | 63.420 s | -18.915 s |

Paired outcomes on identical seeds were 48 both-win, 120 selected-only-win,
0 baseline-only-win, and 88 both-fail. The exact two-sided McNemar p-value was
`1.504632769052528e-36`.

| Level | Curves | Selected wins | Baseline wins | Selected terminal mix |
|---|---:|---:|---:|---|
| Jungle1 | 1 | 16/16 | 7/16 | 16 W |
| Jungle3 | 1 | 16/16 | 8/16 | 16 W |
| Jungle4 | 1 | 16/16 | 11/16 | 16 W |
| jungle6 | 1 | 16/16 | 6/16 | 16 W |
| Jungle7 | 1 | 0/16 | 0/16 | 14 L, 2 T |
| Jungle8 | 1 | 16/16 | 7/16 | 16 W |
| Jungle9 | 2 | 6/16 | 0/16 | 6 W, 9 L, 1 T |
| Jungle10 | 1 | 15/16 | 4/16 | 15 W, 1 T |
| village1 | 1 | 15/16 | 2/16 | 15 W, 1 L |
| village2 | 1 | 16/16 | 1/16 | 16 W |
| village4 | 1 | 13/16 | 1/16 | 13 W, 2 L, 1 T |
| village5 | 1 | 2/16 | 0/16 | 2 W, 14 T |
| village6 | 2 | 0/16 | 0/16 | 16 L |
| village7 | 1 | 13/16 | 1/16 | 13 W, 1 L, 2 T |
| village8 | 1 | 8/16 | 0/16 | 8 W, 1 L, 7 T |
| village10 | 1 | 0/16 | 0/16 | 13 L, 3 T |

### Jungle2 anchor

| Metric | Selected | Baseline |
|---|---:|---:|
| Wins | 32/32 | 23/32 |
| Win rate | 100.0% | 71.875% |
| Wilson 95% CI | 89.28%-100.0% | 54.63%-84.44% |
| Median winning time | 25.44 s | 48.54 s |
| Best winning time | 13.03 s | 27.01 s |

Paired outcomes were 23 both-win, 9 selected-only-win, 0 baseline-only-win,
and 0 both-fail. The exact two-sided McNemar p-value was `0.00390625`.

## Frozen Gate audit

| Gate | Actual | Required | Result |
|---|---:|---:|---|
| Target levels cleared | 13 | 14 | **FAIL** |
| Target wins | 168 | 77 | PASS |
| Target win rate | 65.625% | 30% | PASS |
| Village wins | 67 | 20 | PASS |
| Village win rate | 52.344% | 15% | PASS |
| Jungle2 wins | 32 | 24 | PASS |
| Jungle9 wins | 6 | 4 | PASS |
| village6 wins | 0 | 4 | **FAIL** |

The Gate failure is substantive, not a reporting artifact: the model remains
unable to clear `village6`, and the three zero-win levels (`Jungle7`,
`village6`, `village10`) cap breadth at 13/16.

## Incident and recovery disclosure

At 00:16 local time the original uniform route stopped with an `IndexError`
after the balanced color chooser returned absent color slot 6. The root cause
was a zero-weight boundary case in the captured QRand selection loop. The
failure was preserved before intervention. The minimal fix restricts selection
to live ordered support and uses its last member only as a rounding fallback.

Validation after the fix included the focused seven-test subset, all 100
`test_revenge_core.py` tests, 700,000 support fuzz cases, and the captured
retail QRand vector. The recovery resume smoke verified policy parameters and
all 24 Adam optimizer states bit-for-bit. Formal recovery used newly frozen,
disjoint training seeds. Selection and blind seed ranges remained unchanged and
disjoint. No error recurred in 6,478 recovery episodes.

The discarded work was only the post-checkpoint partial uniform segment for
which no intact model snapshot existed. The preserved source checkpoints were
adaptive 3,145,728 and uniform 2,097,152 steps. This incident is part of the
provenance and does not convert the failed Gates into passes.

## Completeness and provenance checks

- Selection matrices: 160/160 and 180/180 records, `COMPLETE`.
- Target blind matrices: 256/256 and 256/256 records, `COMPLETE`.
- Jungle2 matrix: 64/64 records, `COMPLETE`.
- Every matrix had a unique model/seed key for every expected record.
- Selection seeds covered 610000000-610000067.
- Target blind seeds covered 710000000-710000255.
- Jungle2 blind seeds covered 720000000-720000031.
- Every validation, shard, summarizer, and controller stderr file was empty.
- Controller terminal state was `COMPLETE`; all aggregate statuses were
  `COMPLETE`.

Key frozen artifacts:

- Recovery preregistration SHA-256:
  `90087e58c8135a90ec3bab8f1d52d279075a0310cdc430ba9be08fb742f882b9`
- Final report JSON SHA-256:
  `a7b37e7cb5b820594758c2c85f7b59b1b51deb6494b7a05a35f941386a9e428d`
- Target aggregate SHA-256:
  `b82e017511d39540f17e28ab50bfa451cdac9804f2b67b7764762eb94d8621c8`
- Jungle2 aggregate SHA-256:
  `64aa75a808623e160ead8c0adc7ba33ad0c2a9572c41299837ed0b48c67b0636`
- Selection aggregate SHA-256:
  `c44f0c8611d72553ddb5d1ecc724295dfce7e08617c88ce3262c332f0f862b47`
- QRand incident receipt SHA-256:
  `54af1b55c7bc372a0b0599361f537f50a53e1f292787a30765fe264308736d3e`

## Scientific interpretation

This is strong evidence that the multilevel curriculum transferred beyond
Jungle2 under the frozen structured-state environment and human-limited input
profile. It is not evidence that all 16 target levels are solved, nor is it an
original-game visual-agent record. The next focused stage should preserve this
checkpoint as the generalist anchor and target `Jungle7`, `village6`, and
`village10` without consuming another general blind set.
