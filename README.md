GPT-2 Activation Patching: Two Exploratory Experiments
Status: Completed methodological demonstrations; not a validated circuit discovery or a sociolinguistic-register experiment.
Overview
This project uses GPT-2 small, Python, PyTorch, and TransformerLens to explore how interventions on transformer residual-stream activations affect next-token predictions. Two experiments compare patching the country-name token with patching the final prompt token at each transformer layer.

The goal is to develop and document practical mechanistic-interpretability methods before applying them to my research on sociolinguistic register, model robustness, and generalization.
Research question
How does the position and layer of a residual-stream activation intervention affect a transformer's next-token prediction?
Experimental setup
- Clean prompt: The capital of France is
- Corrupted prompt: The capital of Italy is
- Compared next-token logits:  Paris versus  Rome
- Intervention: Replace the residual-stream activation at one chosen layer and token position with the corresponding activation from the other prompt, using blocks.{layer}.hook_resid_post.
- Measurements: Clean-to-corrupted rescue, corrupted-to-clean damage, and an unchanged-prefix control.
The baseline measure is the Paris-minus-Rome logit difference. The observed values were:
### Baseline Logit Difference

| Measurement | Logit Difference | Interpretation |
|---|---:|---|
| Clean prompt (France) | +1.405 | Prefers Paris over Rome |
| Corrupted prompt (Italy) | -4.417 | Prefers Rome over Paris |
| Behavioral separation | +5.822 | Passes the initial validity threshold |


Thus, the model's relative preference between the two continuations changes in the expected direction. The separation exceeds the notebook's 0.5 exploratory check; this is not a statistical significance test.
Experiment 1 — Country-token activation patching
Here, the activation at the token distinguishing France from Italy is replaced, one layer at a time.

Observed pattern: Bidirectional interventions produce near-full normalized effects through layers 0–7; the effect declines at later layers and is zero at layer 11. The unchanged-prefix control is zero throughout.

[Experiment 1 numerical results](results/patching_results.csv)
Experiment 2 — Final-token activation patching
Here, the activation at the last token of the prompt is replaced, one layer at a time, using the same two prompts and the same next-token measure.

Observed pattern: Early-layer effects are small; the rescue effect reaches approximately 0.699 at layer 9, 0.852 at layer 10, and 1.000 at layer 11. Reverse-direction effects are similar. The unchanged-prefix control is zero throughout.

[Experiment 2 numerical results](results_final_position/patching_results.csv)
Interpretation
The contrasting layer profiles are consistent with the causal structure of a decoder-only transformer. Information at an earlier country-token position can influence later positions through downstream attention, whereas the final token's residual activation becomes directly relevant to predicting the next token.

The full effect of patching the entire final-position residual activation in the last layer is expected from the intervention design; it is not evidence of a newly identified neuron, feature, or circuit. Likewise, a zero unchanged-prefix control is expected for identical source activations and is not, by itself, a demanding negative control.

These experiments demonstrate a working activation-patching workflow and position-sensitive intervention effects for one illustrative prompt pair. They do not establish a unique internal mechanism, and they do not test sociolinguistic register.
Repository contents
- 01_activation_patching.ipynb — Colab notebook with both experiments (ensure the second experiment's cell is saved in the uploaded notebook).
- results/ — Experiment 1 CSV and graph.
- results_final_position/ — Experiment 2 CSV and graph.
Reproduce
1. Open 01_activation_patching.ipynb in Google Colab.
2. Select a Python 3 runtime, preferably with a T4 GPU (CPU may also work).
3. Run the cells in order. The notebook installs the required Python packages and downloads the public GPT-2 model on first use.
4. Compare the printed baseline logit differences, the two result tables, and the graphs. Results may vary with package/model versions and numerical settings.
No Hugging Face private access token is normally needed for the public GPT-2 model. Do not store secrets in notebook outputs or commits.
Limitations and next steps
This is exploratory, not confirmatory: it uses one prompt pair, lacks uncertainty intervals, and has not yet established robustness across independently selected examples. Next steps include multiple preregistered prompt pairs, stronger negative controls, sensitivity analyses, and attention-head-level or more localized interventions.

The longer-term goal is to transfer these methods to Register Feature Family, while preserving that project's earlier failed out-of-distribution behavioral gate. Any new study of register representations requires new held-out evaluations and appropriate controls before making mechanistic claims.
Tools
- TransformerLens
- PyTorch
- Google Colab
