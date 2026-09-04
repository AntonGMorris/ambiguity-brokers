# Ambiguity Brokers

Code and data release for **"Ambiguity Brokers: A Functional Role for Polysemanticity in LLM Inference"** — Anton Morris, MATS Winter 2027 (Nanda Stream) submission.

## What's here

- `ambiguity_brokers_pipeline.py` — the complete experimental pipeline (setup + SAE feature injection + broker survey), designed for Google Colab. Toggles at the top control which phases run.

## Setup

Requires Google Colab Pro (Tesla T4, 15.6 GB VRAM) with a HuggingFace token in Colab secrets as `HF_TOKEN`.

The pipeline installs its own dependencies:
- `git+https://github.com/anthropics/jacobian-lens.git`
- `sae_lens`

## Model and tools used

- **Model**: `google/gemma-2-2b-it` (bfloat16, eager attention — SDPA breaks the J-lens autograd)
- **Pre-fit Jacobian lens**: `neuronpedia/jacobian-lens`, artefact `gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt`
- **SAE**: `gemma-scope-2b-pt-res`, ID `layer_20/width_16k/average_l0_71`
- **Feature interpretations**: Neuronpedia public API

## How to run

1. Open the file in a fresh Colab notebook (Pro, T4 GPU, high-RAM runtime).
2. Set `HF_TOKEN` in Colab secrets (padlock icon on the left).
3. Toggle `RUN_INJECTION` / `RUN_BROKER_SURVEY` at the top based on what you want.
4. Run the single cell. Broker survey alone takes ~20 min; full pipeline ~60 min.

## Reproducing the paper results

- Table 1 (broker existence in matched triples) → set `RUN_INJECTION=True`, uses `PREVIOUSLY_TESTED` polysemes
- Table 2 (broker survey outcome across 24 ambiguous prompts) → set `RUN_BROKER_SURVEY=True`
- Tables 3-7 (injection results, workspace effects, output patterns, matched pair, controls) → set `RUN_INJECTION=True`, `SKIP_PREVIOUSLY_TESTED=False`
- Table 8 (per-polyseme injection metadata) → derived from Phase 3 VERIFICATION step

## Notes

- Neuronpedia's keyword-search API has known coverage gaps — e.g. querying "poultry" returns no L20 matches despite feature F8002 being labelled "laying hens and poultry". This is why 12 polysemes in the broker survey are marked undetermined rather than definitively non-broker.
- Gemma Scope SAEs were trained on base Gemma-2-2B, we use the instruction-tuned Gemma-2-2B-IT. In practice the SAE activations remain interpretable but note this instruction-tuning delta.

## Contact

Anton Morris — antonglennmorris@googlemail.com
