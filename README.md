# CLIP Multimodal Training

## Overview

This project impements a **custom CLIP (Contrastive Language–Image Pretraining) model** for learning joint representations of images and text. The model is trained to align image embeddings and caption embeddings in a shared vector space, enabling tasks such as **image-text retrieval** and **semantic search**.

The training pipeline is built using **PyTorch** and leverages datasets such as **Flickr30k** and **MS COCO**.

---

## Key Features

* Dual encoder architecture (Vision + Text)
* Contrastive learning for image-text alignment
* Support for multiple datasets (Flickr30k, COCO)
* Efficient data loading using HuggingFace Datasets
* Modular training pipeline
* GPU-compatible training

---

## Architecture

The model consists of:

* **Vision Encoder**
  Extracts feature representations from images

* **Text Encoder**
  Based on `DistilBERT (distilbert-base-multilingual-cased)` to encode captions

* **Projection Layers**
  Map both modalities into a shared embedding space

* **Contrastive Loss**
  Maximizes similarity between matching image-caption pairs while minimizing others

---

## Dataset

### Flickr30k

* ~31,000 images
* 5 captions per image
* Used 2 captions per image (~62,000 pairs)

### MS COCO (optional)

* ~164,000 image-text pairs
* Faster convergence due to larger dataset

Datasets are automatically downloaded via:

```python
load_dataset("nlphuji/flickr30k")
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/clip-multimodal-training.git
cd clip-multimodal-training
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Training

```bash
python clip_training.py
```

### Switch Dataset

In `clip_training.py`:

```python
coco_dataset = False  # Change to True for COCO
```

---

## Example Output

The dataset provides image-caption pairs:

* Image → encoded using vision encoder
* Caption → encoded using text encoder
* Both projected into shared embedding space

The model learns to match correct pairs via contrastive loss.

---

## Training Details

* Epochs: 3
* Batch size: 128
* Embedding dimension: 512
* Text model: DistilBERT
* Optimizer: Adam

### Sample Training Logs

```
Epoch [0/3], Loss: 4.85
Epoch [1/3], Loss: 3.18
Epoch [2/3], Loss: 3.09
Epoch [3/3], Loss: 3.16
```

---

## Project Structure

```
clip-multimodal-training/
│
├── src/
│   ├── custom_model.py
│   ├── config.py
│   ├── model_loss.py
│   └── clip_dl.py
│
├── notebooks/
│   ├── flicker30kclip_model.ipynb
│   └── coco2017clip_model.ipynb
│
├── clip_training.py
├── requirements.txt
├── README.md
└── show_image.py
```

---

## Applications

* Image search engines
* Multimodal retrieval systems
* Vision-language understanding
* AI-powered recommendation systems

---

## Future Improvements

* Fine-tuning encoders instead of freezing
* Larger datasets (LAION, WebDataset)
* Better evaluation metrics (Recall@K)
* Deployment via API (FastAPI)

---

## Author

Nazish Pervaiz
MSc Data Science — University of Naples Federico II

---
