# AlphaZuma V1 vs. human Adv 17 record

## Bottom line

The 24.64 s AlphaZuma V1 run is a real deterministic simulator result, but it is
not an original-client world record.  The 16.36 s numerical lead over the human
41 s result is best explained by a combination of a rare favourable RNG/policy
interaction, near-continuous gun-limited execution, aggressive colour swapping,
and exact structured-state targeting.  It is not explained by a uniquely strong
combo strategy: the human run itself reaches Max Combo x3 and Max Chain x13.

## Sources

- V1 replay: `../aispeedrun-alpha-v1/adv17-seven-sides-24.64s.mp4`
- V1 replay receipt: `../aispeedrun-alpha-v1/adv17-seven-sides-24.64s.json`
- Human leaderboard run: <https://www.speedrun.com/zumas_revenge/runs/y805d3dm>
- Human source VOD: <https://www.twitch.tv/videos/1785197918?t=11m34s>
- Human result frame: `human-level17-results.png`
- Deterministic V1 event audit: `v1-event-timeline.json`

## Timing domains

The human category uses the original client's post-level displayed time.  The
result screen shows Ace Time 1:00 and Your Time 0:41.  The V1 value is 2,464
simulator-native ticks at 100 Hz, measured from reset to the irreversible win
boundary.  These clocks are not formally interchangeable.  Visual inspection
does show that the human active gameplay interval is about 41 s, however, so
the full 16.36 s difference cannot be dismissed as a transition or rounding
artifact.

## V1 verified event audit

- Seed: `710000218`
- Time: 2,464 ticks / 24.64 s
- Score: 4,920
- Actions: 71 fires, 54 swaps, 2,339 waits
- Fire cadence: 0.33 s median, 0.34 s mean, 0.38 s maximum interval
- Projectile results: 64 hits and insertions from 71 fires (90.1% hit rate)
- Match events: 40
- Balls removed: 121
- Maximum consecutive clears: 12
- Zuma threshold: 10.59 s
- Remaining cleanup: 14.05 s from Zuma threshold to win
- Power-ups spawned/triggered: 0 / 0
- Fruits spawned/collected: 0 / 0

The event audit replays to the published trajectory SHA-256
`9480cb831d58831f2156d9a888cbe42904a82676fe8afda0d9f9dcd36a32c221`.

## Human result evidence

The original-client result screen reports:

- Your Time: 0:41
- Max Combo: x3
- Max Chain: x13
- Gap Shots: x1
- Fruit: x1
- Perfect Level Bonus: 2,000

The footage visibly contains a large early cascade and later gap/chain bonuses.
The human runner therefore already understands and executes the key macro
strategy.  V1's advantage is concentrated in relentless micro-execution and
the sparse cleanup phase, not discovery of a combo concept unavailable to the
human runner.

## The strongest confounder: selection and RNG

The checkpoint that produced 24.64 s won only 1 of its 16 paired blind attempts
on this level; 24.64 s was that sole win.  The other frozen model won 13 of 16
and had a 32.02 s best and 67.06 s median.  On the exact record seed, that other
model took 43.51 s.  The published rule validly selected the fastest blind win,
but the result is an extreme best-seed record, not evidence of a generally
24-second policy.

## Training implication

Do not optimize for larger combo counters.  Preserve V1's fast colour-management
and sparse-tail behaviour while training it to generalize across seeds.  Track
two separate objectives:

1. best-seed record performance; and
2. robust unseen-seed performance (win rate, median time, and upper quantiles).

The next causal experiment should preregister fresh seeds and ablate swapping,
state precision, and motor limits while reporting time before and after the Zuma
threshold separately.  Original-client record claims should wait for a matched
retail clock and an original-client control/vision bridge.
