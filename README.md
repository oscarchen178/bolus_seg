# Bolus Segmentation (Fluoroscopy)

Detecting oral contrast bolus regions in 512×512 grayscale fluoroscopy frames. Experiments compare single-frame baselines, two-stage clean→bolus pipelines, temporal models, and deterministic imaging priors. Metrics and reproduction notes are consolidated in `BOLUS_SEGMENTATION_REPORT.md`.

## Quick start
- Create env: `conda env create -f environment.yml && conda activate bolus` (Python 3.10, PyTorch, scikit-image/learn, OpenCV, etc.).
- Data layout: place frames under `images/` and matching binary masks under `masks/`; filenames follow `<sequence><frame_idx>_*.png` (sequence = 6 alphanum chars).
- Optional diff data (for `diff_unetpp.ipynb`): `python data_explore/build_diff_dataset.py --images images --masks masks --out-images d_images --out-masks d_masks`.
- Launch notebooks after activating the env: `jupyter notebook model_notebooks/<notebook>.ipynb` (device auto-selects MPS/CUDA/CPU).

## Repository map
- `model_notebooks/` — training/eval runs (self contained).
  - `baseline_unet_smp.ipynb` — single-frame UNet (MobileNetV2) baseline; Dice 0.669 / IoU 0.542.
  - `two_unet_bolus_pipeline.ipynb` — two-stage SmallUNets (clean frame2→frame1 + bolus head on `[frame2, clean_pred]`); Dice 0.628 / IoU 0.519 / AUC 0.995.
  - `temporal_bolus_segnet.ipynb` — Temporal Context UNet over 3-frame clips; Dice 0.644 / IoU 0.515 / AUC 0.996.
  - Variants: `frame2_to_frame1_masked*.ipynb`, `frame2_to_frame1_unetpp.ipynb`, `frame2_to_frame1_jit.ipynb`, `frame2_to_frame1_cross.ipynb`, `frame2_bolus_attn_only.ipynb`, `diff_unetpp.ipynb`, `jit_deblur_bolus.ipynb` (see report for scores; some runs underperform or failed to converge), `two_vit.ipynb` (empty).
- `data_explore/` — preprocessing and deterministic baselines.
  - `bolus_pipeline.ipynb` — CLAHE + high-pass + TV denoise + priors + logistic fusion (Dice 0.425 / IoU 0.273 on 24 frames).
  - `sequence_explore.ipynb` — exploratory filtering/differencing.
  - `build_diff_dataset.py` — build |Δ frame| npy + XOR masks; `count_sequences.py` — quick frame/sequence counts.
- `images/`, `masks/` — primary data; `d_images/`, `d_masks/` — diff dataset outputs.
- `checkpoints/` — saved weights aligned to notebook names (e.g., `baseline_unet_mobilenetv2.pth`, `temporal_bolus_best.pth`, `unetpp_fusion_weights.pth`, `xf_fusion_best.pth`).
- Docs: `BOLUS_SEGMENTATION_REPORT.md`, `NOTEBOOK_SUMMARY.md`, `EXPERIMENT_REPORT.md`, `rubric.md`.
- Papers/drafts: `report/` (ICCP template + figures), `project_paper/` references.

## Running & evaluation tips
- Notebooks define datasets, models, and training loops inline; run cells top-to-bottom. Metrics use threshold 0.5 unless specified.
- Sequence-aware splits are built in; do not shuffle frames independently.
- Checkpoint inference: load the corresponding `*.pth` from `checkpoints/` inside each notebook to reproduce reported scores.
- MPS note: `frame2_to_frame1_cross.ipynb` may hit an autograd issue on MPS—use CPU/CUDA if that occurs.
- Outputs can be large; clear notebook outputs before committing if size matters.

## References
- For detailed metrics, splits, and pipeline descriptions see `BOLUS_SEGMENTATION_REPORT.md` and `NOTEBOOK_SUMMARY.md`.
