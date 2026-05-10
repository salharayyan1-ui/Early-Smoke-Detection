# YOLO11-Smoke-Attention: Wildfire Detection via DCT & Attention Gating

## Overview
Fixed tower-camera systems are used extensively in smoke detection to spot fires before they spiral out of control. However, high false-positive rates make many alert systems unusable. 

We systematically augmented a YOLO11-S baseline with DCT (Discrete Cosine Transform) and skip connection gating to solve this. We trained and tested on a unified 56k image dataset. Several other attention mechanisms were experimented upon with different placement studies, some of which are novel in this domain.

## Architecture: DCT + AG Skips
Our best-performing model modifies YOLO11-S by targeting both channel and spatial features:
* **DCT (Frequency-Domain Attention):** Placed at Layer 10 (post-SPPF bottleneck) to distinguish the mid-frequency texture of smoke from low-frequency haze.
* **Attention Gates (AG):** Inserted at Layers 13 and 17 on the P4 and P3 FPN skip connections. This stops irrelevant background spatial noise from bleeding into the detection neck.

## What We Tried (Ablation & Experiments)
We did not just build one model. We ran a systematic ablation study to see what actually works for surveillance smoke detection. Here is a brief breakdown of our experimental process:

1. **Trained the Baseline:** We started by training a vanilla YOLO11-S model on our unified dataset to establish a performance floor.
2. **Attention & Placement Search:** We experimented with adding CBAM at various bottleneck and neck positions. We found that stacking CBAM in the neck actually degraded performance by suppressing subtle smoke gradients. This led us to test Attention Gates (AG) on the skip connections and DCT frequency filtering, which proved much more effective.
3. **Threshold Sweeping & Spatial Filtering:** We swept confidence thresholds and tried a spatial "sky-zone" filter to reduce false positives. The spatial filter proved too costly, losing roughly 1.35 percentage points of recall for every 1 point of FP rate saved.
4. **CLIP Post-Hoc Filtering:** We attempted to use CLIP (ViT-L/14) as a zero-shot filter to cull false positives. This failed entirely because CLIP struggles with the severe domain gap of heavily compressed, low-light surveillance crops.
5. **Single-Class Training:** We tested a smoke-only hypothesis. It turned out that keeping the "fire" class during training actually improves smoke detection recall due to better boundary supervision.
6. **Hard Negative Mining (HNM):** Finally, we took our best architecture and finetuned it on explicitly mined hard negatives. This slashed our false-positive rate by 47.4% on a blind cross-dataset test set while only costing 1.09 percentage points of recall.

## Results
Our final finetuned **DCT + AG Skips** model achieves exceptional cross-geography generalization. On our blind test set featuring unseen cameras from California, Oregon, and Chile, the model maintained a **72.29% recall** with a highly operational **13.51% False Positive Rate**. 

## Repository Structure

Here is a quick breakdown of how the files in this repository are organized:

##  Repository Structure

Here is a quick breakdown of how the files in this repository are organized:

```text
├── notebooks/   # Jupyter notebooks containing the executed code, experiments, and cell outputs
└── results/     # Exported metrics, including CSVs, Excel spreadsheets, and data tables
```
## References
* Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11.
* Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
* Qin, Z., et al. (2021). FcaNet: Frequency Channel Attention Networks. *ICCV*.
* Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *MIDL*.
* Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.
* Shrivastava, A., et al. (2016). Training Region-Based Object Detectors with Online Hard Example Mining. *CVPR*.
