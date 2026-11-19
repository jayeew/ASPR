# Relaxed 7/16/153/219 HGB OOF results

Validation passed with 84/84 checkpoints and 3,739,712 unique OOF predictions.

| Horizon | Strict 7 | Primary 16 | Expanded 153 | Broad T0 219 |
|---|---:|---:|---:|---:|
| D3 | 0.697210 | **0.716019** | 0.699667 | 0.699440 |
| D5 | 0.720492 | **0.737112** | 0.729834 | 0.728232 |
| D8 | 0.757467 | **0.772332** | 0.770773 | 0.771518 |

The 16-feature Primary set is best at every horizon. Removing the two constant fields from the former 154/221 sets leaves predictions and Spearman values unchanged, as expected. These sets were reconstructed from historically evaluated v3 memberships, so this is a deterministic reproduction and exploratory model-development comparison rather than a new outcome-blind confirmatory selection.
