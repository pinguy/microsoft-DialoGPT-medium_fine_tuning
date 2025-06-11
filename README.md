# Fine-Tuning Pipeline for Memory-Infused Dialogic AI

A modular pipeline for preparing, embedding, indexing, and fine-tuning dialogue data using LoRA and FAISS-backed memory on resource-constrained systems.

---

## 📦 Prerequisites

**Platform:** Linux (tested in Distrobox)

**Minimum Specs:** CPU with 8 threads and 24 GB RAM for training. You can reduce memory usage by adjusting the following values in `train.py`:

* `per_device_train_batch_size`
* `gradient_accumulation_steps`

**Recommended:** No GPU required. With a few tweaks, it’ll even run on a potato.

---

## 🔧 Environment Setup

```bash
distrobox create --name rhizome-dev --image rhizome-devbox
distrobox enter rhizome-dev
```

Install dependencies:

```bash
# Install packages
pip3 install -r requirements.txt

# Download spacy English model
python3 -m spacy download en_core_web_sm
```

---

## 📂 Folder Structure

```
AI_Fine_Tuning_Pipeline/
├── PDFs/                   # Raw PDFs to be parsed into text
├── data_finetune/          # Clean Q&A dataset generated for fine-tuning. Created once `data_formatter.py` is ran
├── dialogpt-finetuned/     # Checkpoint outputs from training. Created once `train.py` is ran 
├── conversations.json      # Your exported conversation history. The one here is just a placeholder
├── batch_embedder.py
├── chat.py
├── data_formatter.py
├── embedding_config.json   # Created once `batch_embedder.py` is ran 
├── memory.index            # Created once `batch_embedder.py` is ran 
├── memory_texts.npy        # Created once `batch_embedder.py` is ran 
├── memory_metadata.pkl     # Created once `batch_embedder.py` is ran 
├── pdf_to_json.py
├── rhizome.py
├── train.py
├── README.md
```

---

## 🧱 Pipeline Overview

### 1. Convert PDFs

```bash
python3 pdf_to_json.py
```

> Converts each PDF into chunked text with metadata.

---

### 2. Add Chat History

Rename your largest `conversations.json` export from ChatGPT (or other AI logs) and place it in the root folder.

---

### 3. Embed and Index Memory

```bash
python3 batch_embedder.py
```

> Creates semantic memory using FAISS, producing:

- `memory.index`
- `memory_texts.npy`
- `memory_metadata.pkl`

---

### 4. Generate Fine-Tuning Dataset

```bash
python3 data_formatter.py
```

> Cleans, deduplicates, and formats Q\&A pairs into `data_finetune/`.
> 
> If you run into issues, tweak `self.quality_score_threshold`.
> Higher values give fewer examples but better quality.

---

### 5. Train the Model (LoRA)

```bash
python3 train.py
```

> LoRA fine-tuning on DialoGPT. Outputs go into `dialogpt-finetuned/`.

---

### 6. Interact with Memory (Optional)

```bash
python3 rhizome.py
```

> Query the FAISS memory index interactively.  
> ⚠️ Back up the files named memory or avoid running if you need untouched index for data_formatter.py.

---

### 7. Chat with Your Fine-Tuned Model

```bash
python3 chat.py
```

> Loads latest `checkpoint-*` from `dialogpt-finetuned/` and runs in an interactive loop.

---

## 🧠 Features

- Built for CPUs (LoRA + SentenceTransformer)
- Semantic memory recall using FAISS
- Fully reproducible from raw PDFs or chat history
- Modular and interpretable stages
- No reliance on proprietary APIs

---

## 🍷 Notes

- Works best with well-curated, conversational datasets
- Memory-backed recall enables enhanced introspective evaluation
- Logs and training diagnostics are saved automatically

---

## 📄 License

This project is licensed under the WTFPL – *Do What the Fuck You Want to Public License*.
See [wtfpl.net](http://www.wtfpl.net/) for more.

---
