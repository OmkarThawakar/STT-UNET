# STT-UNET
3D Mitochondria Instance Segmentation with Spatio-Temporal Transformers

[Omkar Thawakar](https://scholar.google.com/citations?user=flvl5YQAAAAJ&hl=en),[Rao Muhammad Anwer](https://scholar.google.com/citations?user=_KlvMVoAAAAJ&hl=zh-CN), [Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ&hl=en), [Orly Reiner](https://scholar.google.co.il/citations?user=hGbXUosAAAAJ&hl=en), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en) 

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.12073)

## :rocket: News

* **(Feb 23, 2023)**
  * Code uploaded V1.0.
 * **(Feb 21, 2023)**
  * Arxiv Release.

<hr />

#### Abstract

Accurate 3D mitochondria instance segmentation in electron microscopy (EM) is a challenging problem and serves as a prerequisite to empirically analyze their distributions and morphology. Most existing approaches employ 3D convolutions to obtain representative features. However, these convolution-based approaches struggle to effectively capture long-range dependencies in the volume mitochondria data, due to their limited local receptive field. To address this, we propose a hybrid encoder-decoder framework based on a split spatio-temporal attention module that efficiently computes spatial and temporal selfattentions in parallel, which are later fused through a deformable convolution. Further, we introduce a semantic foreground-background adversarial loss during training that aids in delineating the region of mitochondria instances from the background clutter. Our extensive experiments on three benchmarks, Lucchi, MitoEM-R and MitoEM-H, reveal the benefits of the proposed contributions achieving state-of-theart results on all three datasets.

### Installation

First, clone the repository locally:

```bash
git clone https://github.com/OmkarThawakar/STT-UNET.git
```

Then, create environment from yml or create conda env and install dependencies:

```bash
conda env create -f mito.yml
```
OR 
```
conda create -n mito python=3.8
conda activate mito
pip install -r requirements.txt

```

Install dependencies and pycocotools for MitoEM:

```bash
pip install torchsummary waterz malis
```
