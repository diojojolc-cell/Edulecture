# Edulecture: Fine-Grained Evidence Alignment for Educational Video Retrieval

This repository contains the official implementation and the data of the **Edulecture** project. The focus of this research is on fine-grained evidence alignment for retrieving specific content from long-form educational and STEM videos.

## 📝 Introduction

Retrieving precise information from K-12 and STEM educational videos requires robust alignment between text queries and video segments. In this work, we developed a framework to address the challenge of fine-grained evidence alignment. The model processes a given video $V$ and a text query $Q$ to retrieve the most relevant evidence segment $E$. 

To achieve this, the Video-Text Matching (VTM) objective evaluates the similarity between character-based video segments and text queries.

## 🛠️ Environment Setup

To ensure reproducibility and avoid issues with weight loading, please strictly adhere to the following core library versions.

* **PyTorch:** 2.8
* **Transformers:** 4.55

Create the environment and install dependencies:

```bash
conda create -n edulecture python=3.10
conda activate edulecture
pip install torch==2.8.0 transformers==4.55.0
```
## 📊 Datasets
The annotations of the dataset are located in the datasets/ directory.

Data Access Policy
JSON Annotations (Open Sourced): We only provide the JSON files containing the text transcripts, temporal boundaries, and evidence labels.

Video Files (Not Included): Due to copyright restrictions, raw video files are not included. You must download the videos directly from the respective educational websites (https://basic.smartedu.cn and https://gzclass.gztv.com) using the URLs provided in the JSON files.
