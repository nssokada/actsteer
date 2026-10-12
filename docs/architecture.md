# Architecture

ActSteer has one directed package boundary:

```text
actsteer.emotion -> actsteer.core
```

The core owns model instrumentation, normalized records, activation capture,
interventions, readouts, artifacts, and execution. The emotion package owns
emotion-dataset adapters, vector-extraction presets, and emotion calibration.
Downstream repositories own task semantics, statistical analysis, plots, and
scientific claims.

See [the architecture and migration plan](plans/architecture-and-migration.md)
for the full design.

