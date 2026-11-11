# CT-based multimodal lung adenocarcinoma classification

Compact research code for **CT-based multimodal models for lung adenocarcinoma spectrum identification and invasiveness classification**.

The repository contains the primary model components and their training, preprocessing and prediction workflows. Neural modules and training calculations are extracted from the study implementation; file handling and command-line entry points are consolidated for this release.

| Task | Reference classes | Final probability combination |
| --- | --- | --- |
| Spectrum identification | Non-spectrum / adenocarcinoma spectrum | 0.75 × CT–semantic + 0.25 × clinical |
| Invasiveness classification | AAH/AIS / MIA / IAC | 0.50 × CT-only + 0.50 × BiCA |

Component probabilities are averaged across seeds **3407, 2025 and 2026** before combination. The CT branch uses a 3D ResNet-18-style encoder with GroupNorm. Structured measurements use a feature-token Transformer. Identification uses CT-query residual fusion; invasiveness uses bidirectional cross-attention. Clinical information is incorporated for identification through elastic-net logistic regression.

## Installation

Run commands from the repository root. The study used Python 3.9.21, PyTorch 2.6.0 and scikit-learn 1.6.1.

```bash
conda env create -f environment.yml
conda activate lung-idclass
```

Alternatively, use a Python 3.9 environment and `python -m pip install -r requirements.txt`. For GPU training, install the PyTorch 2.6.0 wheel that matches your platform using the [official installation commands](https://pytorch.org/get-started/previous-versions/#v260). The neural training entry point defaults to CUDA; `--device cpu` supports small smoke runs.

## Inputs

Create one development CSV per task, containing **train and val rows only**, and a separate evaluation CSV. `case_id` identifies a lesion; `patient_id` keeps lesions from the same patient in one partition. NPZ paths are relative to the CSV location or absolute.

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for the complete schema and [examples/](examples/) for header-only templates. Clinical records, CT images, segmentations, patient-level predictions and trained weights are not distributed with this repository.

Prepare a reviewed single-lesion CT/ROI pair:

```bash
python -m lung_idclass.preprocessing --ct CT.nii.gz --mask ROI.nii.gz --output data/case.npz
```

This creates a 1 mm isotropic NPZ with normalized CT and a 5 mm expanded lesion mask. CT and ROI must have matching physical geometry. The loader applies the study's mask-based crop, outside-mask value −1, maximum dimension 128, training flips and dynamic batch padding.

## Train the primary components

The following Linux/bash example trains the identification components. For invasiveness, change `task` to `invasiveness` and use that task's manifest. The task-specific configuration selects residual fusion or BiCA automatically.

```bash
task=identification
for seed in 3407 2025 2026; do
  python -m lung_idclass.train --task "$task" --model ct \
    --manifest "data/${task}_development.csv" --seed "$seed" \
    --output "outputs/${task}/seed${seed}/ct"
  python -m lung_idclass.train --task "$task" --model semantic \
    --manifest "data/${task}_development.csv" --seed "$seed" \
    --output "outputs/${task}/seed${seed}/semantic"
  python -m lung_idclass.train --task "$task" --model fusion \
    --manifest "data/${task}_development.csv" --seed "$seed" \
    --ct-init "outputs/${task}/seed${seed}/ct/best.pt" \
    --semantic-init "outputs/${task}/seed${seed}/semantic/best.pt" \
    --output "outputs/${task}/seed${seed}/fusion"
done
```

Each component saves `best.pt`, validation predictions, training history and its configuration. Semantic components also save the training-fitted transformer and token schema. Checkpoints are selected by validation AUC. Identification fusion freezes the initialized encoders for eight epochs, then unfreezes their upper layers; BiCA keeps the initialized encoders frozen.

Train the clinical identification branch with the published settings:

```bash
python -m lung_idclass.clinical fit \
  --csv data/identification_development.csv --output outputs/clinical
```

Add `--cv` to repeat training-only regularisation selection over the original parameter grid. The default uses the reported `C=0.1`, `l1_ratio=0.5`, balanced class weights and 5000 maximum iterations.

## Predict and combine

Export one probability CSV per neural seed. For example:

```bash
for seed in 3407 2025 2026; do
  python -m lung_idclass.predict \
    --checkpoint "outputs/identification/seed${seed}/fusion/best.pt" \
    --manifest data/identification_external.csv --batch-size 2 \
    --output "predictions/id_fusion_${seed}.csv"
done
python -m lung_idclass.clinical predict \
  --csv data/identification_external.csv --models outputs/clinical \
  --output predictions/clinical.csv
python -m lung_idclass.fusion --task identification \
  --left predictions/id_fusion_3407.csv predictions/id_fusion_2025.csv predictions/id_fusion_2026.csv \
  --right predictions/clinical.csv --right-averaged \
  --threshold 0.7400414428163694 --output predictions/identification_final.csv
```

For invasiveness, export CT-only and BiCA predictions for each seed, then combine:

```bash
python -m lung_idclass.fusion --task invasiveness \
  --left predictions/inv_ct_3407.csv predictions/inv_ct_2025.csv predictions/inv_ct_2026.csv \
  --right predictions/inv_bica_3407.csv predictions/inv_bica_2025.csv predictions/inv_bica_2026.csv \
  --output predictions/invasiveness_final.csv
```

Use `--model fusion` during training to obtain BiCA for the invasiveness task. For prediction, the study component batch sizes are CT-only **4**, semantic-only **32**, and fusion **2**. Keep the selected batch size and input row order fixed: dynamic spatial padding can change predictions. When reproducing existing predictions, retain their original complete inference batches before applying any final cohort subset.

The identification threshold shown above is the full-precision value retained in the study configuration (reported as 0.7400 in the paper). For newly trained models, use validation predictions to determine and save the corresponding threshold. `fusion --select-validation data/identification_validation.csv` selects the identification blend and threshold from validation labels, writing them to the output JSON. Apply those saved values using `--weight` and `--threshold` on evaluation data. Invasiveness uses the fixed equal-weight combination and argmax.

## Configuration and scope

[CONFIGURATION.md](docs/CONFIGURATION.md) lists the study settings. The `configs/` JSON files hold the task presets. This compact release covers primary preprocessing, CT/semantic models, clinical models, training and final probability fusion. Supplementary architecture comparisons, manuscript plotting and the complete statistical reporting pipeline are outside its scope.

The package supplies code and data-format templates. Reproducing the paper's numerical results requires the study data, patient partitions, fitted preprocessing and trained checkpoints. The consolidated entry points have not been used to retrain the full study cohorts.

## Check the installation

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic arrays and records only. They cover model outputs, a gradient update, preprocessing, probability alignment and train-only feature transformations. See [VALIDATION.md](docs/VALIDATION.md) for the execution checks performed on this release.

## Citation

If you use this code, cite the manuscript title above. Publication details will be added when available.
