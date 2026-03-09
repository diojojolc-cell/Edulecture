# Constructing a Educational Lecture Video Dataset with Fine-Grained Annotations for Video Retrieval Tasks

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

The annotations of the dataset are located in the `datasets/` directory.

### Data Access Policy

* **JSON Annotations (Open Sourced):** We only provide the JSON files containing the text transcripts, temporal boundaries, and evidence labels.
* **Video Files (Not Included):** Due to copyright restrictions, raw video files are not included. You must download the videos directly from the respective educational websites using the URLs provided in the JSON files. 

### Video Source URLs

Below are the specific source links for the educational videos used in this project:

**Platform: basic.smartedu.cn**
* [Video Source 1](https://basic.smartedu.cn/syncClassroom?defaultTag=e7bbce2c-0590-11ed-9c79-92fc3b3249d5%2F44bebf7c-54e6-11ed-9c34-850ba61fa9f4%2F44bebcde-54e6-11ed-9c34-850ba61fa9f4%2Fff8080814371757b01437c363a187b0a%2Fff8080814371757b014390f883db0453%2F5136342961)
* [Video Source 2](https://basic.smartedu.cn/syncClassroom?defaultTag=e7bbce2c-0590-11ed-9c79-92fc3b3249d5%2F44bec67a-54e6-11ed-9c34-850ba61fa9f4%2F44bebcde-54e6-11ed-9c34-850ba61fa9f4%2Fff8080814371757b01437c363a187b0a%2Fff8080814371757b014390f883db0453%2F5136342961)
* [Video Source 3](https://basic.smartedu.cn/syncClassroom?defaultTag=e7bbce2c-0590-11ed-9c79-92fc3b3249d5%2F44bec0c6-54e6-11ed-9c34-850ba61fa9f4%2Fe7bbcfee-0590-11ed-9c79-92fc3b3249d5%2Fff8080814371757b01437c363a187b0a%2F8ae7e58b77b3bac901783dd80dee0c44%2F5136342961)

**Platform: gzclass.gztv.com**
* [Video Source 4](https://gzclass.gztv.com/gksubjecpc/index2.html?columnName=%E5%88%9D%E4%B8%80&uuid=625&Gradeindex=6)
* [Video Source 5](https://gzclass.gztv.com/gksubjecpc/index2.html?columnName=%E5%88%9D%E4%B8%80&uuid=625&Gradeindex=6)
* [Video Source 6](https://gzclass.gztv.com/gksubjecpc/index2.html?columnName=%E5%88%9D%E4%B8%80&uuid=625&Gradeindex=6)
* [Video Source 7](https://gzclass.gztv.com/gksubjecpc/index2.html?columnName=%E5%88%9D%E4%B8%80&uuid=625&Gradeindex=6)
