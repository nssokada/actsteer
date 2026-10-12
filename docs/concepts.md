# Core concepts

## Residual-stream sites

ActSteer v0.1 captures and modifies the residual stream at transformer block
boundaries. A site specifies whether the value is observed before or after the
block. Attention-head, MLP, and neuron-level sites are intentionally deferred.

## Layer selection

Layers can be selected by absolute index, normalized model depth, or a
normalized depth window. Relative selectors make an experiment portable across
models with different numbers of blocks; resolved absolute indices should
always be stored with the result.

## Token selection

Available selectors include all valid tokens, the last valid token, the last
`k` valid tokens, tokens after an offset, an explicit interval, and a named
prompt section. All selectors intersect with the attention mask, so left- and
right-padded batches are handled explicitly.

## Extraction presets

| Preset | Center | Neutral PC removal | Normalization |
|---|---|---:|---:|
| `contrastive_mean` | Neutral-example mean | No | L2 |
| `sofroniew_global` | Mean of selected concept means | Yes | L2 |
| `sofroniew_neutral` | Neutral-example mean | Yes | L2 |

`contrastive_mean` is the most direct contrast between concept and neutral
examples. The Sofroniew presets remove leading principal components estimated
from neutral examples before normalization. With `sofroniew_global`, changing
the selected concept population also changes the center and therefore every
direction.

## Calibration

Raw coefficients are not comparable across layers or models. ActSteer supports:

- `projection_sd`: one unit is one standard deviation of activation projected
  onto the selected unit direction;
- `residual_norm`: one unit is a fraction of the mean residual-vector norm.

Projection-SD is the recommended default. Residual-norm scaling is useful when
reproducing work that used norm-relative coefficients.

## Multi-layer dose

The default `total_budget` policy divides a standardized dose by the square
root of the number of selected layers. This keeps the aggregate standardized
norm fixed. The `per_layer` policy applies the full strength independently at
every layer and should be selected explicitly.

## Intervention windows

- `encode_release`: steer selected prompt tokens during prefill, then release
  the model for response generation;
- `response_only`: leave prompt encoding unchanged and steer decoded tokens;
- `continuous`: steer selected prompt tokens and continue through decoding.

These windows test different causal questions. They should be treated as
experimental conditions rather than interchangeable implementation details.
