# Image-to-LaTeX Conversion of Mathematical Expressions

A deep learning project exploring neural architectures for converting images of mathematical expressions into structured LaTeX markup.

This project implements and compares four encoder-decoder architectures on the IM2LATEX-100K benchmark dataset, progressively incorporating Transformer decoding, attention mechanisms, row encoders, and higher-resolution feature extraction to improve mathematical expression recognition.

---

## Key Results

| Model | Exact Match | BLEU-4 |
|---------|------------|---------|
| ResNet-18 + LSTM | 0.06% | 20.40% |
| ResNet-18 + Transformer | 5.37% | 62.85% |
| ResNet-18 + Row Encoder + Attention | 14.17% | 75.35% |
| ResNet-18-V2 + Row Encoder + Attention | **36.78%** | **87.62%** |

The strongest model achieved a 613× improvement in exact-match accuracy over the baseline LSTM architecture and demonstrated that preserving spatial resolution in encoder feature maps is one of the most important factors for image-to-markup generation.

---

## Project Overview

Mathematical OCR is substantially more difficult than traditional text recognition because the model must infer two-dimensional structure and generate syntactically correct LaTeX code. Fractions, superscripts, subscripts, summations, and nested expressions all require understanding spatial relationships between symbols.

To investigate which architectural choices matter most, this project implements four progressively more sophisticated models:

- ResNet-18 + LSTM Decoder
- ResNet-18 + Transformer Decoder
- ResNet-18 + Row Encoder + Bahdanau Attention
- ResNet-18-V2 + Higher-Resolution Attention Architecture

---

## Dataset

This project uses the **IM2LATEX-100K** dataset containing over 100,000 image-formula pairs.

The dataset is loaded directly through the HuggingFace Datasets API at runtime. Users do **not** need to manually download the dataset before running the project. The dataset will be automatically downloaded and cached during the first execution.

Dataset source:

https://huggingface.co/datasets/yuntian-deng/im2latex-100k

---

## Technologies

- Python
- PyTorch
- Torchvision
- HuggingFace Datasets
- NumPy
- Matplotlib
- NLTK
- PIL

---

## Repository Structure

```text
.

├── README.md
├── requirements.txt
├── notebooks/
│   └── csc_331_final_project_final.ipynb
├── reports/
│   └── ECE_CSC_331_FinalReport.pdf
└── src/
    └── csc_331_final_project_final.py
```

---

## Installation

```bash
git clone https://github.com/ushnakhann/image-to-latex-deep-learning.git
cd image-to-latex-deep-learning

pip install -r requirements.txt
```

---

## Running the Project

```bash
python csc_331_final_project_final.py
```
---

## Main Finding

The largest performance improvement came from modifying the ResNet backbone to preserve more spatial information.

Increasing the number of annotation positions available to the attention mechanism from **20 to 80** improved exact-match accuracy from **14.17% to 36.78%**, showing that feature-map resolution is a dominant factor in attention-based image-to-LaTeX generation.

---

## Future Improvements

- Beam search decoding instead of greedy decoding
- Larger annotation grids
- Learned positional embeddings
- Vision Transformer (ViT) encoders
- Rendering-based evaluation metrics
- Larger-scale training and hyperparameter tuning

---

## Authors

**Kshitij Agarwal**  
Computer Science & Economics, Union College

**Ushna Khan**  
Computer Science, Union College

---

## Report

The complete project report is available in:

`ECE_CSC_331_FinalReport.pdf`

It includes:
- Literature review
- Model architectures
- Training methodology
- Evaluation metrics
- Experimental results
- Architecture diagrams
- Training curves
- Future work discussion
