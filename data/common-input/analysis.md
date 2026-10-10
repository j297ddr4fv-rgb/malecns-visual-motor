# Same-input connection-structure comparison

Each network receives identical paired retinal pixels and identical fitted T4/T5 currents. Retinal movies replay the nine intact-body reference trajectories; network output does not change these movies. The original and each rewired graph have three initialization seeds. Rewiring realizations and initializations are separate sampling levels.

| Network | Stimulus | Neural contrast (Hz, mean ± SD) | Mean neural P |
|---|---|---:|---:|
| original | positive60 | +81.167 ± 2.315 | 0.2847 |
| original | negative60 | -64.333 ± 1.843 | 0.2897 |
| shuffle_20261010 | positive60 | -3.444 ± 2.334 | 0.3148 |
| shuffle_20261010 | negative60 | -1.889 ± 1.875 | 0.3176 |
| shuffle_20261011 | positive60 | +1.389 ± 2.299 | 0.3350 |
| shuffle_20261011 | negative60 | +3.083 ± 2.036 | 0.3358 |
| shuffle_20261012 | positive60 | -0.194 ± 1.569 | 0.3525 |
| shuffle_20261012 | negative60 | +0.556 ± 2.258 | 0.3532 |

All per-condition retinal and fitted-drive stream identities match the intact reference. No body is simulated in this replay: P is evidence of retained neural locomotor drive, not a measured walking endpoint. The contrast uses unsmoothed DNp15 event rates before the steering decoder, with each motion condition corrected by its matched static condition.

The three sampled null graphs can indicate recurrence across these realizations, but cannot establish a population-wide property of rewired networks or locate a selective biological pathway. Global rewiring preserves marginal degrees and signed strengths while changing many circuit partners simultaneously. The replay removes differing sensory input between network arms; it does not establish a fully biological neural-to-muscle mapping.
