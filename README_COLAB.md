# ConceptTransformer — Colab Quickstart

Quick instructions for running this repo in Google Colab. Covers setup, data download, and training for both MNIST and CUB-200-2011 datasets.

---

## Requirements

- Google Colab
- **Recommended GPU:** A100 or G4 (standard GPU runtimes are too slow for CUB training)
  - Set via: *Runtime → Change runtime type → GPU → A100 or G4*

---

## Usage

Follow the cells in **`Colab_Concept_Transformer.ipynb`** in order. The notebook handles everything:

1. **Setup**: Clones the repo and installs dependencies
2. **MNIST**: Downloads data automatically, trains and evaluates the model
3. **CUB-200-2011** — Downloads the bird dataset, then trains and evaluates

No virtual environment setup is needed as the notebook runs directly in the Colab environment.

---

## Expected Results

| Dataset | Test Accuracy |
|---------|--------------|
| MNIST   | 0.979        |
| CUB-200-2011 (100 epochs) | 0.821 |

---

## Known Issues

**Duplicate progress bars** — A Colab rendering/refresh issue causes multiple progress bar lines to print per training step for both MNIST and CUB. This is only a cosmetic issue and training still runs correctly.