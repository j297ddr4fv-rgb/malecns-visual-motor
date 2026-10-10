# Development and calibration evidence

These files document interface development and the source of the stationary-stripe baseline. They are separate from the 302-run behavioral batch, 18 embodied topology-control recordings, and 36 common-input replay runs.

- `diagnostic-comparison.json`: per-condition gate and raw-record equality diagnostics from the same source analysis.
- `diagnostic-comparison.csv`: original-versus-rerun comparison from the matched-input development diagnostic. Seventeen comparable conditions retained identical raw locomotor-drive and bilateral DNp15 records over their 800-frame sequences. The control reference and gate were evaluated separately. Source: `/data/artifacts/by-project/malecns-gpu/engineering/standardized-repair-20261007-v1/analysis/repair-comparison.csv` and its parent `REPORT.md`.
- `static-baseline-payload.json`: fixed-head stationary-stripe baseline, contrast 1 and 60-degree spatial period; DNp15 L/R = 0/142.5 Hz. Source: `/data/artifacts/by-project/malecns-gpu/engineering/flygym-dnp15-template-steering-decoder-20261007-v1/static-baseline-payload.json`. This payload supplies the shared reference used in subsequent evaluation.

Development sequence was checked against the original project sessions: the clear open-loop response was recorded on 7 October 2026 at 16:24 Beijing time, followed by an online response at 17:59. These preliminary recordings are not pooled with the repeated behavioral batch. The batch began on 9 October at 17:16 and ended on 10 October at 19:48, with pauses and case-boundary resumes. Original project-session evidence is retained locally; full session logs are not part of the reviewer bundle.
