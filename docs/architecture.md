# Architecture

ActSteer separates model instrumentation from domain-specific extraction logic:

```text
your experiment
      |
      +--> actsteer.emotion
      |          |
      |          v
      +----> actsteer.core
```

`actsteer.core` never imports `actsteer.emotion`. This keeps the core useful for
non-emotion concepts and allows additional domain packages to build on the same
capture and intervention machinery.

## Core package

The core package owns:

- normalized text and chat records;
- deterministic dataset fingerprints;
- model-family adapters and native prompt rendering;
- semantic activation sites, layer selectors, and token selectors;
- reduced residual-stream capture;
- intervention windows and dose allocation;
- immutable artifact manifests and storage; and
- shared configuration and CLI primitives.

Model-family differences are isolated behind `ModelAdapter`. Capture and
steering code asks for a transformer block by semantic layer index instead of
containing Gemma-, Llama-, or Qwen-specific attribute checks.

## Emotion package

The emotion package owns:

- optional emotion-dataset adapters;
- contrastive and neutral-PC-removed extraction presets; and
- activation-distribution calibration helpers.

It does not own a particular emotion ontology, prompt collection, behavioral
task, or response classifier.

## Downstream experiment boundary

An experiment should remain a normal Python project that imports ActSteer. It
owns:

- the research question and experimental conditions;
- task and evaluation prompts;
- train/validation/evaluation separation;
- outcome coding, statistical analysis, and visualization; and
- interpretation of the results.

This boundary prevents task assumptions from becoming hidden defaults in the
instrumentation library.

## Runtime flow

```text
records
  -> validated ConceptDataset
  -> model-native rendering and token spans
  -> reduced activation capture
  -> direction extraction and calibration
  -> targeted residual intervention
  -> downstream measurement and analysis
```

Every expensive reusable result should carry enough provenance to reconstruct
the path on the left. See [Artifacts and provenance](artifacts.md).

## Extension points

New model families should implement the `ModelAdapter` contract and expose
transformer blocks without changing capture or intervention code. New concept
domains can depend on `actsteer.core` in the same direction as
`actsteer.emotion`. New task readouts should generally remain downstream unless
they measure a model primitive that is reusable across unrelated experiments.
