# 🌱 RhizomeML  
### _A Fine-Tuning Pipeline for Memory-Infused Dialogic AI_

> A modular, CPU-friendly pipeline for preparing, embedding, labeling, and fine-tuning dialogue and document data using **LoRA**, **SentenceTransformers**, and **FAISS-backed memory** — entirely offline.

---

## 📦 Prerequisites

**Platform:** Linux (tested inside [Distrobox](https://distrobox.it/))  
**Minimum Specs:** 8-core CPU, 24 GB RAM (32 GB+ recommended)  
**GPU:** Optional (Compute Capability ≥ 6.0)  
**Reality check:** Runs on decade-old Xeons. Patience required.

To reduce memory use, tweak these inside `train_script.py`:

* `per_device_train_batch_size`
* `gradient_accumulation_steps`

---

## 🔧 Environment Setup

```
# Inside Distrobox or native Linux
distrobox create --name rhizome-dev --image rhizome-devbox
distrobox enter rhizome-dev
````

Install dependencies:

```
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

---

## 📂 Folder Structure

```
RhizomeML/
├── PDFs/                     # Optional raw PDFs
├── data_finetune/            # Auto-generated clean datasets
├── DeepSeek-R1-Distill-Qwen-1.5B-finetuned_*  # Output checkpoints
│
├── pdf_to_json.py            # Converts PDFs → structured JSON
├── batch_embedder.py         # Embeds and indexes semantic memory
├── data_formatter.py         # Cleans, dedups, and labels data
├── train_script.py        # Fine-tunes models (CPU-optimized)
├── requirements.txt
└── README.md
```

---

## 🧱 Pipeline Overview

### 1️⃣ Convert PDFs

```bash
python3 pdf_to_json.py ./PDFs/
```

> Converts PDFs into structured text chunks with metadata.

---

### 2️⃣ Add Chat History

Place your exported conversation logs in the project root:

```
conversations.json      # ChatGPT
conversations2.json     # Claude (optional)
```

---

### 3️⃣ Embed & Index Semantic Memory (uses CPU by default but can change use_gpu=False to True for GPU)

```
python3 batch_embedder.py
```

> Generates semantic embeddings with SentenceTransformer and builds FAISS-ready arrays:
>
> * `memory_texts.npy`
> * `memory_metadata.pkl`

---

### 4️⃣ Generate Fine-Tuning Dataset

```
python3 data_formatter.py \
  --enable-semantic-labeling \
  --semantic-mode normal \
  --semantic-method hybrid
  --force-cpu (Forces it to use the CPU don't it uses GPU)
```

> Merges, cleans, and labels all sources.
> Outputs ready-to-train datasets in `data_finetune/`:
>
> * `dataset_train.jsonl`
> * `dataset_validation.jsonl`
> * `dataset_test.jsonl`
> * `dataset_metadata.json`

---

### 5️⃣ Train the Model (LoRA Fine-Tune)

```
python3 train_script.py
```

> Fine-tunes **DeepSeek-R1-Distill-Qwen-1.5B** (or other compatible models).
> CPU-aware, resumable, and logs everything for reproducibility.

Outputs go into:

```
DeepSeek-R1-Distill-Qwen-1.5B-finetuned_YYYYMMDD_HHMMSS/
```

---

## 🧠 Features

* **Runs entirely on CPUs** — perfect for offline or low-resource systems
* **Adaptive semantic labeling** for smarter dataset curation
* **FAISS-backed recall** for context-aware augmentation
* **LoRA adapters** for incremental, lightweight updates
* **Resumable fine-tuning** with RNG-state preservation
* **No proprietary APIs** or cloud calls
* **Modular design**: use any component standalone

---

## ⚙️ Example Hardware

| Component | Spec                                             |
| :-------- | :----------------------------------------------- |
| CPU       | Intel Xeon E5-2680 v4 (28 threads)               |
| RAM       | 32 GB ECC DDR4                                   |
| GPU       | NVIDIA Quadro M2000 (Compute 5.2 → CPU fallback) |
| Storage   | Samsung NVMe 512 GB                              |
| OS        | Debian / Pop!_OS                                 |

---

## 🍷 Notes

* The pipeline rewards **quality over quantity** — curate before you train.
* **Keyphrase extraction** improves semantic richness but increases runtime; enable only for smaller datasets.
* Training runs can take days on CPU — use screen/tmux and log output.
* Included is gradio_chat_tts.py a TTS → TTS using [Vosk](https://alphacephei.com/vosk/models)) (will need to download one of the models and place into the root Dir) and [Kokoro](https://github.com/pinguy/kokoro-tts-addon)). Need to place the UCS_v3_4_1.py with it. At the moment Alpha stage but other interfaces are available.
---
