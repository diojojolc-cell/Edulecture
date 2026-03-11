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


### Dataset Construction

To ensure the high quality and reliability of the data, the construction of the dataset followed a rigorous three-stage pipeline: **Data Collection**, **Segment Extraction**, and **Evidence Annotation**.

#### 1. Data Collection
We collected long-form educational videos from official K-12 platforms. The videos including Physics, Chemistry, and Biology. 

#### 2. Segment Extraction and Caption Generation
We partitioned the videos using a strict character-based splitting strategy (implemented in `text_split.py`). Following the segmentation, we utilized Large Language Models (LLMs), specifically GPT-4o, Moonshot-v1, and Gemini-2.5-Pro, to generate high-quality, descriptive captions for each segment.

#### 3. Manual Refinement and Verification
To ensure the high quality of the dataset, trained human annotators systematically reviewed the machine-generated outputs. The annotators performed necessary merging or re-segmentation of the clips to preserve the semantic completeness of the instructional context.

### Data Access Policy

* **JSON Annotations (Open Sourced):** We only provide the JSON files containing the text transcripts, temporal boundaries, and evidence labels.
* **Video Files (Not Included):** Due to copyright restrictions, raw video files are not included in this repository. The copyrights of the original videos belong to the respective official educational platforms.

### Preprocessing

Note that the video segmentation of the dataset relies entirely on **character-based splitting**. We did not use semantic models (such as m3e) for segmentation. Please ensure your video downloading and preprocessing pipeline aligns with the character boundaries defined in the JSON files.

### Video Downloading

#### 1. Official Sources
You can access the original videos directly from the official educational platforms. 

**Platform: basic.smartedu.cn (Smart Education of China)**
* [Video Source 1](https://basic.smartedu.cn/syncClassroom?defaultTag=e7bbce2c-0590-11ed-9c79-92fc3b3249d5%2F44bebf7c-54e6-11ed-9c34-850ba61fa9f4%2F44bebcde-54e6-11ed-9c34-850ba61fa9f4%2Fff8080814371757b01437c363a187b0a%2Fff8080814371757b014390f883db0453%2F5136342961)


**Platform: gzclass.gztv.com (Guangzhou TV Classroom)**
* [Video Source 4](https://gzclass.gztv.com/gksubjecpc/index2.html?columnName=%E5%88%9D%E4%B8%80&uuid=625&Gradeindex=6)


#### 2. Third-party Mirrors (Convenience Downloads)
For the convenience of researchers, we also provide third-party mirror links (hosted on Bilibili) where the videos can currently be downloaded.

> **⚠️ Disclaimer:** These Bilibili links are provided by third-party uploaders. We do not own the copyrights to these videos and are merely providing these links to facilitate academic research. If these mirror links become invalid or are removed, users will need to source the videos directly from the original platforms mentioned above.

**Middle School Physics**
| Content | Textbook Edition | Download Link (Mirror) |
| :--- | :--- | :--- |
| Grade 8 (First Semester) | PEP (人教版) | [Bilibili Video](https://www.bilibili.com/video/BV14jgdzLE9Q) |
| Grade 8 (Second Semester) | PEP (人教版) | [Bilibili Video](https://www.bilibili.com/video/BV1fGTwzqE31) |
| Grade 9 (Full Year) | PEP (人教版) | [Bilibili Video](https://b23.tv/yLyCQ7w) |
| Grade 8 (First Semester) | Beijing Normal Univ. (北师大版) | [Bilibili Video](https://www.bilibili.com/video/BV1FouEz8ENC) |
| Grade 8 (Second Semester) | Beijing Normal Univ. (北师大版) | [Bilibili Video](https://b23.tv/jlx40Eo) |
| Grade 9 (Full Year) | Beijing Normal Univ. (北师大版) | [Bilibili Video](https://b23.tv/hkLAQ6P) |

**Middle School Chemistry**
| Content | Textbook Edition | Download Link (Mirror) |
| :--- | :--- | :--- |
| Grade 9 (First Semester) | PEP (人教版) | [Bilibili Video](https://www.bilibili.com/video/BV1Y1gyzdENa) |
| Grade 9 (Second Semester) | PEP (人教版) | [Bilibili Video](https://www.bilibili.com/video/BV1SuuPzNEN1) |

**Middle School Biology**
| Content | Textbook Edition | Download Link (Mirror) |
| :--- | :--- | :--- |
| Grade 7 (First Semester) | Universal | [Bilibili Video](https://www.bilibili.com/video/BV1pH4y1P7GK) |
| Grade 7 (Second Semester) | Universal | [Bilibili Video](https://www.bilibili.com/video/BV1sZ421772x) |
| Grade 8 (First Semester) | Universal | [Bilibili Video](https://www.bilibili.com/video/BV1Ut421F77L) |
| Grade 8 (Second Semester) | Universal | [Bilibili Video](https://www.bilibili.com/video/BV1BC411E7z8) |

