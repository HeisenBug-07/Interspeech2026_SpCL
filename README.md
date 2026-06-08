# This paper has been accepted at Interspeech 2026, Sydney
# SpCL: Semi-positive Contrastive Learning for Spoken Language Identification

**SpCL** is a high-performance framework for robust **Spoken Language Identification (LID)** across 12 Indian languages. By introducing a novel **semi-positive contrastive loss** and an audio-text dual encoder, it masters **cross-domain generalization** (e.g., training on broadcast TV and testing on YouTube) without requiring explicit target-domain data.

---

## 🚀 Key Features

* **Cross-Domain Generalization:** Eliminates performance drops caused by shifting recording environments.
* **Dual-Encoder Architecture:** Jointly embeds speech waveforms and phoneme-based text captions into a shared 512-D space during training.
* **Zero Overhead at Inference:** The text branch is dropped completely at inference time—only the audio encoder and classifier are used.
* **Supported Models:** Built-in variants for both **Wav2Vec2** and **Whisper**.

---

## 🛠️ Repository Structure

```text
├── spcl_w2v2/           # Wav2Vec2 + RoBERTa variant
│   ├── model_w2v2_roberta.py
│   ├── train_w2v2_roberta.py
│   ├── eval_w2v2_rob.py
│   └── dataloader.py
└── spcl_whisper/        # Whisper Encoder + RoBERTa variant
    ├── model_whisper_roberta.py
    ├── train_whisper_roberta.py
    ├── eval_whisper_rob.py
    └── dataloader.py

```

---

## 📐 Architecture & Loss Design

### The Core Loss Function

The model optimizes a joint loss function ($\lambda = 0.3$ by default):

$$L_{total} = \lambda \cdot L_{con} + (1 - \lambda) \cdot L_{cls}$$

#### 1. Semi-Positive Contrastive Loss ($L_{con}$)

Instead of binary positive/negative pairs, SpCL dynamically weights pairs based on their domain context:

| Pair Relationship | Language | Domain | Loss Weight |
| --- | --- | --- | --- |
| **Strong Positive** | Same | Same | `1.0` |
| **Semi-Positive** | Same | Different | `α` $\in (0, 1)$ |
| **Negative** | Different | Any | `0.0` |

#### 2. Classification Loss ($L_{cls}$)

Standard Cross-Entropy Loss applied to the audio embedding for direct language classification.

---

### Dual Encoder Configurations

| Variant | Audio Backbone | Text Backbone | Unfreezing Strategy |
| --- | --- | --- | --- |
| `spcl_w2v2` | `wav2vec2-lv-60-espeak-cv-ft` | `roberta-base` | Top 4 Wav2Vec2 layers |
| `spcl_whisper` | `whisper-base` (Encoder only) | `roberta-base` | Top 2 Whisper blocks |

* **Note:** For both variants, the top 4 layers of RoBERTa, both linear projection heads, and the language classifier remain fully unfrozen.

---

## 📦 Data Format

Metadata `.txt` files must follow this space-separated format:

```text
/path/to/audio.wav "phoneme caption text" <class_label> <domain_label>

```

> 💡 **Tip:** Using **Dynamic Captions** (actual utterance-level phoneme sequences) yields far better out-of-distribution generalization than **Static Captions** (fixed templates per language).

---

## ⚡ Quick Start

### Installation

```bash
pip install torch torchaudio transformers scikit-learn tqdm numpy soundfile librosa

```

### Training Examples

```bash
# Train Wav2Vec2 variant (Dynamic Captions, α = 0.7)
cd spcl_w2v2
python train_w2v2_roberta.py \
    --train_txt ../train_dynamic.txt \
    --val_txt ../val_dynamic.txt \
    --ckpt_dir ckpt_w2v2_dynamic \
    --contrast_weight 0.7 \
    --class_weight 0.3

# Train Whisper variant (Dynamic Captions, α = 0.7)
cd ../spcl_whisper
python train_whisper_roberta.py \
    --train_txt ../train_dynamic.txt \
    --val_txt ../val_dynamic.txt \
    --ckpt_dir ckpt_whisper_dynamic \
    --contrast_weight 0.7 \
    --class_weight 0.3

```

### Evaluation

```bash
python eval_whisper_rob.py \
    --ckpt_path ckpt_whisper_dynamic/dual_enc_ep50.pt \
    --eval_txt ../test_unseen.txt

```

---

## 📊 Performance Benchmarks

### Cross-Domain Robustness (Accuracy %)

Models were trained on indoor/broadcast data and evaluated on completely unseen out-of-distribution domains (**DatasetM YouTube** and **IndicVoices**):

| Model Strategy | Seen Domains | DatasetM (YT) | IndicVoices (IV) |
| --- | --- | --- | --- |
| Baseline (MFCC-Conformer) | 92.7% | 17.7% | 10.4% |
| Baseline (UDA / GRL) | 85.6% | 83.2% | 47.6% |
| **SpCL w2v2 (Dynamic)** | 97.5% | 67.5% | 34.3% |
| **SpCL whisp (Dynamic)** | **98.8%** | **91.4%** | **54.5%** |

### The Impact of Semi-Positive Weight ($\alpha$)

Setting $\alpha$ between `0.5` and `0.7` consistently outperforms standard contrastive learning ($\alpha = 0$) on out-of-distribution datasets:

| Model | Metric Split | $\alpha = 0$ | $\alpha = 0.5$ | $\alpha = 0.7$ |
| --- | --- | --- | --- | --- |
| **SpCL whisp** | DatasetM (YT) | 90.58% | 88.53% | **91.42%** |
|  | IndicVoices (IV) | 51.55% | 52.12% | **54.52%** |

```

```
