# DeCHUD: Gated DeAOT for Video Object Segmentation
**ECE 4715 Final Project | Paper Implementation**

> **Abstract:** This repository implements the experiments from our paper: *"Deformable Attention Object Tracking (DeAOT) variants evaluated on YouTube-VOS 2019."* We introduce a **Gated Global Propagation Memory (GPM)** that achieves a **+3.1% improvement** over the baseline by adaptively fusing visual and identity embeddings.

---

## 🔬 Project Variants (Paper Sections)
This repository contains the code for the three variants discussed in our paper:

| Variant | Paper Section | Description | Code Location |
| :--- | :--- | :--- | :--- |
| **1. Baseline Control** | Sec IV | Standard DeAOT with fixed summation fusion. | `variants/1_baseline_control/` |
| **2. Gated GPM (Ours)** | **Sec V** | **(Main Branch)** Uses learnable sigmoid gating for feature fusion. **Best Performance.** | `networks/layers/transformer.py` |
| **3. Loss-Modified** | Sec VI | Adds Temporal Consistency Loss ($L_{gpm}$) to regularize gating. | `variants/3_loss_modified/loss.py` |

---

## 🏗️ Architecture (Section III & V)
As described in **Section V** of our paper, we modify the `GatedGPMFusion` block to replace static summation with a learnable gate:

$$f_{fused} = \sigma(W_g[f_{vis}; f_{id}]) \odot f_{id} + (1 - \sigma(W_g[f_{vis}; f_{id}])) \odot f_{vis}$$

### Implementation (`networks/layers/transformer.py`)
```python
# From Listing 1 in Paper
class GatedGPMFusion(nn.Module):
    def forward(self, f_vis, f_id):
        concat = torch.cat([f_vis, f_id], dim=1)
        gate = torch.sigmoid(self.gate_mlp(concat))
        f_fused = gate * f_id + (1.0 - gate) * f_vis
        return f_fused, gate
