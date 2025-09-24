# TCSAFormer

**Efficient Vision Transformer with Token Compression and Sparse Attention for Medical Image Segmentation**

---

## 🚀 Overview
TCSAFormer is a U-Net style Transformer architecture designed for **efficient and accurate medical image segmentation**.  
It tackles the high computational cost of self-attention and the limited spatial modeling of standard FFNs by introducing:

- **Compressed Attention (CA):** Reduces redundant tokens and applies sparse attention.  
- **Dual-Branch Feed-Forward Network (DBFFN):** Captures local + regional features with parallel convolutional branches.  

---

## 🏗️ Architecture

### Overall Design
- **U-shaped encoder–decoder** with skip connections.  
- **Patch Embedding / Merging / Expanding** for multiscale feature extraction.  
- Each stage built from **TCSAFormer blocks**, containing:  
  - Depthwise conv (3×3) for local positional encoding.  
  - **Compressed Attention (CA).**  
  - **DBFFN (Dual-Branch FFN).**  
  - LayerNorm + residual connections.  

Number of blocks per stage: `[2, 2, 8, 1, 1, 8, 2, 2]`.

---

### 🔹 Compressed Attention (CA)
Efficient attention module with three steps:

1. **Token Compression Pipeline (TCP)**  
   - *Prune*: Remove unimportant tokens using importance scores.  
   - *Merge*: Fuse redundant tokens via similarity graph matching.  

2. **Top-k Sparse Attention**  
   - Each query attends only to its top-k most relevant keys.  
   - Reduces quadratic cost to near-linear.  

3. **Token Decompression Pipeline (TDP)**  
   - Unmerge + restore pruned tokens through residual shortcut.  
   - Recovers full token set for downstream processing.  

---

### 🔹 Dual-Branch FFN (DBFFN)
Replacement for vanilla Transformer FFN:

- `1×1 conv` → channel mixing.  
- Two parallel branches:  
  - **3×3 depthwise conv** → fine local features.  
  - **7×7 depthwise conv** → coarse regional features.  
- Concatenate + fuse → `1×1 conv` → output.  

Provides **multi-scale spatial feature extraction** inside the Transformer block.

---

## 📊 Key Benefits
- ✅ Reduced FLOPs & parameters vs. standard ViTs.  
- ✅ Strong segmentation accuracy on **ISIC-2018, CVC-ClinicDB, Synapse** datasets.  
- ✅ Better trade-off between efficiency and accuracy.  
- ⚡ Modular design: CA + DBFFN can be reused in other architectures.  

---

## 🔧 My Modification: Spatial-Sparse Block (**SSBlock**)

To make the architecture more **straightforward while still efficient**, I replaced the original TCSAFormer block with a unified **SSBlock**, inspired by **bi-level sparse attention works (e.g. BiFormer, BigBird, Focal Transformer)**.

### Key Features of SSBlock
- **Three Attention Paths**:  
  1. **Compressed Global** — attention over all block means (global summaries).  
  2. **Selective Global** — attention restricted to top-k most relevant blocks.  
  3. **Local Window** — per-query sliding window attention for fine detail.  

- **Gated Fusion**: A learned gating mechanism mixes the three attention outputs adaptively.  
- **DBFFN Inside**: Retains the dual-branch convolutional FFN (3×3 + 7×7).  
- **Logging-friendly**: Exposes intermediate tensors (block indices, attention maps) for analysis.  
- **Encoder/Decoder Compatible**: Same block works in both stages.  

### Advantages
- ✅ Simplified design (one unified block for all stages).  
- ✅ Keeps global, sparse, and local context jointly.  
- ✅ Still efficient: avoids quadratic attention cost.  
- ✅ More transparent for debugging and profiling.  
