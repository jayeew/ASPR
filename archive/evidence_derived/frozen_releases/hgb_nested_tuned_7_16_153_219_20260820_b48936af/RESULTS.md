# Horizon-specific nested HGB tuning

Each outer fold selects one HGB configuration using only Primary16 predictions from four inner expanding-time folds. The selected configuration is then applied to all four feature sets in that outer fold. Outer-test labels never participate in selection.

## OOF comparison

| Horizon | Set | Fixed rho | Tuned rho | Delta |
|---:|---|---:|---:|---:|
| D3 | broad_t0 | 0.699440 | 0.705702 | +0.006262 |
| D3 | expanded | 0.699667 | 0.704126 | +0.004459 |
| D3 | primary | 0.716019 | 0.718465 | +0.002446 |
| D3 | strict | 0.697210 | 0.699474 | +0.002265 |
| D5 | broad_t0 | 0.728232 | 0.735880 | +0.007648 |
| D5 | expanded | 0.729834 | 0.731970 | +0.002136 |
| D5 | primary | 0.737112 | 0.739570 | +0.002457 |
| D5 | strict | 0.720492 | 0.722368 | +0.001876 |
| D8 | broad_t0 | 0.771518 | 0.775562 | +0.004044 |
| D8 | expanded | 0.770773 | 0.774463 | +0.003690 |
| D8 | primary | 0.772332 | 0.773814 | +0.001482 |
| D8 | strict | 0.757467 | 0.759082 | +0.001616 |

All 12 tuned comparisons exceed their fixed-medium counterpart. These are point estimates; no statistical-significance claim is made.

## Best tuned set by horizon

- D3: primary, rho=0.718465 (delta=+0.002446)
- D5: primary, rho=0.739570 (delta=+0.002457)
- D8: broad_t0, rho=0.775562 (delta=+0.004044)

## Selected configuration counts

- D3: large_regularized × 5
- D3: medium_flexible × 3
- D5: large_regularized × 3
- D5: medium_flexible × 4
- D8: large_regularized × 1
- D8: medium_flexible × 4
- D8: medium_slow × 1

Canonical validation and tuning-specific validation both passed.
