# GPT-2 Activation Patching: Two Exploratory Experiments

**Status:** Completed exploratory methodological demonstrations (single prompt pair); **not** a validated circuit discovery or a sociolinguistic-register experiment.

## Overview

This project uses GPT-2 Small, Python, PyTorch, and TransformerLens to test how replacing residual-stream activations affects next-token predictions. Two experiments compare interventions at the country-name token and the final prompt token across transformer layers. The goal is to establish a reproducible methodological foundation for future investigations into sociolinguistic register, robustness, and generalization.

## Research question

**How does the token position and transformer layer of a residual-stream intervention affect the model's next-token prediction?**

## Experimental setup

- **Clean prompt:** `The capital of France is`
- **Corrupted prompt:** `The capital of Italy is`
- **Continuations compared:** ` Paris` versus ` Rome`
- **Intervention:** Replace the residual-stream activation at one specified layer and token position with the corresponding vector from the other prompt (`blocks.{layer}.hook_resid_post`).
- **Outcome:** Paris-minus-Rome logit difference; normalized bidirectional rescue/damage scores.
- **Control:** Patch an unchanged prefix position. Because the prefix activations are identical, this is a basic implementation check rather than a strong negative control.

### Baseline logit differences

| Measurement | Logit difference | Interpretation |
|---|---:|---|
| Clean prompt (France) | +1.405 | Prefers Paris over Rome |
| Corrupted prompt (Italy) | −4.417 | Prefers Rome over Paris |
| Behavioral separation | +5.822 | Exceeds the notebook's 0.5 exploratory threshold |

The separation is an effect-size measure for this example, **not** a statistical significance test.

## Experiment 1: Country-token patching

The source activation is inserted at the position of `France` / `Italy`, separately at each layer. Rescue and reverse-damage effects are near 1 through layers 0–7, decline at later layers, and reach 0 at layer 11. The unchanged-prefix control is 0 throughout.

![Experiment 1 — country-token activation patching](results/patching_results.png)

[View Experiment 1 numerical results](results/patching_results.csv).

## Experiment 2: Final-token patching

The intervention is repeated at the final prompt position (`is`). Recovery is small at early layers, approximately 0.699 at layer 9, 0.852 at layer 10, and 1.000 at layer 11. Reverse-direction effects are similar; the unchanged-prefix control is again 0.

![Experiment 2 — final-token activation patching](results_final_position/patching_results.png)

[View Experiment 2 numerical results](results_final_position/patching_results.csv).

## Interpretation

The contrasting layer profiles are consistent with GPT-2's causal attention structure. Information at an earlier country-token position can affect the final prediction through downstream attention; the residual activation at the final position directly feeds the next-token readout.

**Important:** Full recovery after patching the *entire final-layer residual vector at the final prompt position* is expected from the intervention design. It does **not** identify a specific neuron, head, feature, or circuit. Likewise, a zero effect from patching an unchanged prefix is not a demanding negative control.

## Repository contents

```text
01_activation_patching.ipynb
results/
  patching_results.csv
  patching_results.png
results_final_position/
  patching_results.csv
  patching_results.png
README.md
```

## Reproduce

1. Download or open [`01_activation_patching.ipynb`](01_activation_patching.ipynb) in Google Colab.
2. Select a Python 3 runtime (T4 GPU recommended when available).
3. Run all cells from the top. The notebook installs the Python packages and downloads the public GPT-2 model; a private Hugging Face token is not normally required.
4. Compare the printed baseline scores, CSV results, and graphs in the two results folders. Values can vary slightly across software environments.

## Limitations and next steps

These demonstrations use **one** prompt pair, lack confidence intervals across independently sampled pairs, and do not establish robustness beyond this example. The interventions replace full residual-stream vectors rather than isolating attention heads or model features.

Planned next steps include preregistered additional prompt pairs, stronger negative controls, attention-head-level interventions, and a separate exploratory mechanistic follow-up to [Register Feature Family](https://github.com/Rinetta1981/register-feature-family). The prior register experiment's failed out-of-distribution generalization gate should remain documented; no generalizable register mechanism is claimed on its basis.

## Tools

- Python / PyTorch
- TransformerLens
- Google Colab
- pandas / Matplotlib
