# Bolus Segmentation Notebook Documentation

## Common Data & Splits
- Images: grayscale 512x512 PNGs in `images/`; masks with matching names in `masks/`. Inputs normalized to `[0,1]`.
- Filename convention: `<sequence><frame_idx>_*.png` where `sequence` is a 6-char id; `frame_idx` parsed via `^(?P<prefix>[A-Za-z0-9]{6})(?P<frame>\d+)_`.
- Standard sequence split (used by frame-pair and temporal notebooks): 60 train / 13 val / 14 test sequences -> 4,761 train, 843 val, 733 test pairs or clips. Single-frame baseline uses frames directly: 4,821 train, 856 val, 747 test.
- Augmentation: optional horizontal/vertical flips; otherwise identity.

## Segmentation Metrics (test)
- Threshold 0.5 for IoU/Dice; AUC is ROC AUC over pixels when available.

| Notebook | Model (head) | IoU | Dice | AUC |
| --- | --- | --- | --- | --- |
| baseline_unet_smp | UNet (MobileNetV2 encoder) | 0.542 | 0.669 | - |
| frame2_to_frame1_cross | Clean ViT + cross-frame fusion | 0.4477 | 0.5781 | 0.9888 |
| frame2_to_frame1_jit | ViT-style + window-attn fusion | 0.4855 | 0.6087 | 0.9875 |
| frame2_to_frame1_masked | Masked AE clean + channel+spatial attn | 0.4675 | 0.5950 | 0.9851 |
| frame2_to_frame1_masked_2 | Masked UNet clean + channel+spatial attn | 0.4349 | 0.5609 | 0.9772 |
| frame2_to_frame1_unetpp | UNet clean + UNet++ fusion | 0.4706 | 0.6019 | 0.9899 |
| temporal_bolus_segnet | Temporal TCM-UNet | 0.5152 | 0.6435 | 0.9960 |

## Reconstruction Metrics (test clean models)
- L1 is masked where noted; PSNR uses the same mask handling as the loss.

| Notebook | Clean model | L1 | PSNR |
| --- | --- | --- | --- |
| frame2_to_frame1_cross | ViT decoder | 0.0112 (L1) | 31.36 dB |
| frame2_to_frame1_jit | ViT-style | 0.0077 (masked L1) | 33.37 dB |
| frame2_to_frame1_masked | Conv autoencoder | 0.0059 (masked L1) | 38.15 dB |
| frame2_to_frame1_masked_2 | TinyUNet | 0.0055 (masked L1) | 41.18 dB |
| frame2_to_frame1_unetpp | TinyUNet | 0.0056 (L1) | 40.23 dB |
| baseline_unet_smp | - | - | - |
| temporal_bolus_segnet | - (temporal segmentation only) | - | - |

## Notebook Details

### baseline_unet_smp.ipynb — Single-frame baseline
- Architecture: `smp.Unet` with MobileNetV2 encoder (no pretrained weights), 1 input channel, 1 output channel, ~6.63M params.
- Training: BCEWithLogits + soft Dice; Adam (lr 1e-3, wd 1e-4); ReduceLROnPlateau on val Dice; batch 8; 40 epochs. Split by sequences (individual frames).
- Checkpoint: `baseline_unet_mobilenetv2.pth`.
- Test segmentation: Dice 0.669, IoU 0.542.

### frame2_to_frame1_cross.ipynb — Clean ViT + cross-frame fusion (unmasked)
- Data: frame2 -> frame1 reconstruction (no masking).
- Clean stage: `CleanFrameViT_XPred` (patch 32, embed 64, depth 4, heads 4; ~0.37M params) with 5-stage upsampling decoder. Loss = L1 + 0.1×SSIM-like; AdamW lr 3e-4. Checkpoint `xf_clean_weights.pth`. Test reconstruction: L1 0.0112, PSNR 31.36.
- Fusion stage: `CrossFrameBolusNet` (~0.35M params) with ConvStem to 1/8 scale, global cross-attention (Q=frame1, K/V=clean prediction), dilated context, upsampling head. Joint fine-tune clean (lr 1e-4) + fusion (lr 1e-3) with BCEWithLogits + Dice. Fusion weights `xf_fusion_best.pth`.
- Test segmentation: IoU 0.4477, Dice 0.5781, AUC 0.9888.

### frame2_to_frame1_jit.ipynb — ViT-style + windowed fusion (masked L1 clean)
- Data: frame2 -> frame1 with bolus mask ignored in clean loss (valid = 1 - mask).
- Clean stage: `CleanFrameJiT` ViT-style encoder-decoder (patch 16, embed 64, depth 6, heads 4; ~0.46M params). Loss = masked L1; Adam lr 1e-3. Checkpoint `jit_clean_weights.pth`. Test reconstruction: masked L1 0.0077, PSNR 33.37.
- Fusion stage: `FusionBolusJiT` (~0.23M params) with stride-2 stem to 1/4 scale, windowed MSA (win 8), dilated context, upsampling head. Input: concat(frame1, clean_pred). Loss = BCE + 0.5×Dice; lr 1e-3 fusion, 1e-4 clean during joint fine-tune. Weights `jit_fusion_weights.pth`.
- Test segmentation: IoU 0.4855, Dice 0.6087, AUC 0.9875.

### frame2_to_frame1_masked.ipynb — Masked clean autoencoder + channel/spatial attention fusion
- Data: masked L1 ignores bolus pixels when reconstructing frame1 from frame2.
- Clean stage: `ConvAutoencoder` (base 32, ~0.42M params); Adam lr 1e-3. Checkpoint `my_model_weights.pth`. Test reconstruction: masked L1 0.0059, PSNR 38.15.
- Fusion stage: `AttentionBolusNet` (base 32) on concat(frame1, frame2, clean_pred); channel attention + spatial attention; loss BCE + 0.5×Dice; lr 1e-3 fusion, 1e-4 clean. Weights `attn_model_weights2.pth`.
- Test segmentation: IoU 0.4675, Dice 0.5950, AUC 0.9851.

### frame2_to_frame1_masked_2.ipynb — Masked clean UNet + channel/spatial attention fusion
- Data/loader identical to previous, but clean model uses skips.
- Clean stage: `TinyUNet` (features 16/32/64; ~0.48M params). Loss = masked L1; Adam lr 1e-3. Checkpoint `my_model_weights4.pth`. Test reconstruction: masked L1 0.0055, PSNR 41.18.
- Fusion stage: `AttentionBolusNet` (base 32, inputs: frame1 + clean_pred). Loss BCE + 0.5×Dice; lr 1e-3 fusion, 1e-4 clean. Weights `attn_model_weights4.pth`.
- Test segmentation: IoU 0.4349, Dice 0.5609, AUC 0.9772.

### frame2_to_frame1_unetpp.ipynb — Clean UNet + UNet++ fusion
- Data: same frame-pair split/loader (no mask in clean loss).
- Clean stage: `TinyUNet` (features 32/64/128; ~1.93M params). Loss = L1 + 0.1×SSIM-like; AdamW lr 1e-3, wd 1e-4, 40 epochs. Checkpoint `unet_clean_weights.pth`. Test reconstruction: L1 0.0056, PSNR 40.23.
- Fusion stage: `UNetPlusPlus` (~2.07M params) with inputs (frame1 + clean_pred). Joint fine-tune with BCEWithLogits + 0.8×Dice; lr 1e-3 fusion, 1e-4 clean. Extra manual fusion epoch saved to `unetpp_fusion_manual_epoch1.pth`; main weights `unetpp_fusion_weights.pth`.
- Test segmentation: IoU 0.4706, Dice 0.6019, AUC 0.9899.

### temporal_bolus_segnet.ipynb — Temporal Context Module UNet
- Model: `TemporalBolusSegNet` (~0.60M params). Lite encoder/decoder with depthwise separable convs; Temporal Context Modules per scale blend features across a clip (`seq_len=3`, `target_index=1`). Supports predict-one or predict-all; training uses predict-one.
- Training: BCE + Dice; AdamW lr 1e-3, wd 1e-4; batch 4; 20 epochs; checkpoint `temporal_bolus_best.pth`.
- Test segmentation: Dice 0.6435, IoU 0.5152, AUC 0.9960, Sensitivity 0.7204.

### two_vit.ipynb
- Notebook is empty (0 cells); no architecture, training, or metrics recorded.

