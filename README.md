# DeCHUD: Gated DeAOT for Video Object Segmentation

**ECE 4990 Final Project**

> **Abstract:** This repository implements the experiments from our paper:
> *"A Detailed Study of DeAOT Enhancements via Gated Global Propagation Memory and Loss Function Modifications."*
> We introduce a **Gated Global Propagation Memory (GPM)** that achieves a **+3.1% improvement** over the baseline by adaptively fusing visual and identity embeddings.

---

## Project Variants (Paper Sections)

This repository contains the code for the three variants discussed in our paper:

| Variant | Paper Section | Description | Code Location |
|---------|---------------|-------------|---------------|
| **1. Baseline Control** | Sec IV | Standard DeAOT with fixed summation fusion. | `variants/1_baseline_control/` |
| **2. Gated GPM (Ours)** | **Sec V** | **Main contribution.** Learnable sigmoid gating for feature fusion. | `networks/layers/transformer.py` |
| **3. Loss-Modified** | Sec VI | Adds additional regularization via modified loss terms. | `variants/3_loss_modified/` |

---

## Architecture (Sections III & V)

We replace static feature summation in DeAOT's Global Propagation Memory with a learnable gating mechanism:

$$
f_{fused} = \sigma(W_g[f_{vis}; f_{id}]) \odot f_{id} + (1 - \sigma(W_g[f_{vis}; f_{id}])) \odot f_{vis}
$$

### Implementation (`networks/layers/transformer.py`)

```python
class GatedGPMFusion(nn.Module):
    def forward(self, f_vis, f_id):
        concat = torch.cat([f_vis, f_id], dim=1)
        gate = torch.sigmoid(self.gate_mlp(concat))
        f_fused = gate * f_id + (1.0 - gate) * f_vis
        return f_fused, gate
```

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependencies (for demo visualization):

```bash
pip install scikit-image opencv-python tqdm
```

> **Note:** `spatial_correlation_sampler` is optional. If unavailable, the model falls back to a slower implementation.

---

## Dataset Setup

### YouTube-VOS 2019 (Not Included)

```
datasets/
└── YTB/
    └── 2019/
        ├── train/
        │   ├── JPEGImages/
        │   ├── Annotations/
        │   ├── meta.json
        │   └── meta_instance.json
        ├── valid/
        │   ├── JPEGImages/
        │   ├── Annotations/
        │   ├── meta.json
        │   └── meta_instance.json
        └── test/
            ├── JPEGImages/
            └── meta.json
```

---

## Training (tools/train.py)

All experiments use `tools/train.py`. Outputs are saved automatically under `result/`.

### 1. Baseline DeAOT (Control)

```bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="native"

python tools/train.py \
  --exp_name ytb2019_deaot_r50_control \
  --stage YTB2019 \
  --model r50_deaotl \
  --gpu_num 1 \
  --batch_size 2 \
  --datasets youtubevos \
  --amp
```

### 2. Gated GPM DeAOT (Main Result)

```bash
python tools/train.py \
  --exp_name ytb2019_deaot_gated_r50 \
  --stage YTB2019 \
  --model r50_deaotl \
  --gpu_num 1 \
  --batch_size 2 \
  --datasets youtubevos \
  --amp
```

### 3. Gated GPM + Modified Loss

```bash
python tools/train.py \
  --exp_name ytb2019_deaot_r50_loss \
  --stage YTB2019 \
  --model r50_deaotl \
  --gpu_num 1 \
  --batch_size 2 \
  --datasets youtubevos \
  --amp
```

---

## Evaluation

### Generate Predictions

```bash
python tools/eval.py \
  --exp_name ytb2019_deaot_gated_r50 \
  --stage YTB2019 \
  --model r50_deaotl \
  --dataset youtubevos \
  --split val \
  --gpu_num 1 \
  --amp
```

Predictions are saved under:

```
results/youtubevos2019/
```

### Metrics Reported

**Accuracy**
- **J (Mean IoU)** – region similarity
- **F (Contour Accuracy)** – boundary quality
- **J&F Mean** – standard DAVIS metric

**Performance**
- FPS
- Peak GPU Memory

YouTube-VOS metrics are computed locally. DAVIS-2017 metrics use the official evaluation code.

---

## Qualitative Demos (tools/demo.py)

Generate annotated videos:

```bash
python tools/demo.py \
  --model r50_deaotl \
  --ckpt_path result/ytb2019_deaot_gated_r50_R50_DeAOTL/YTB2019/ckpt/save_step_100000.pth \
  --data_path demo_ytb2019_val \
  --output_path demo_out_ytb2019_gated \
  --gpu_id 0 \
  --amp
```

Output:

```
demo_out_ytb2019_gated/*.avi
```

---

## Repository Structure

```
.
├── configs/              # Dataset & stage configs
├── networks/
│   ├── engines/          # AOT / DeAOT engines
│   ├── layers/           # Transformer, GPM, loss
│   └── models/           # Model definitions
├── tools/
│   ├── train.py
│   ├── eval.py
│   └── demo.py
├── datasets/             # Not tracked
├── result/               # Checkpoints & logs
└── results/              # Evaluation outputs
```

---

## Key Contributions

- **Learnable Gated Global Propagation Memory**
- **+3.1% Mean IoU improvement** on YouTube-VOS 2019
- Improved temporal stability
- Minimal compute overhead
