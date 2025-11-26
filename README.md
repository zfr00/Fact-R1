# Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning

## FakeVV dataset

Please note that the training sets for the third stage, which include [fakett](https://github.com/ICTMCG/FakingRecipe) and [fakesv](https://github.com/ICTMCG/FakeSV), are not displayed due to access restrictions.


### FakeVV dataset test set
The complete test dataset is defined in:
- **`data_config/fakevv_test_data.json.json`**

Includes webpage url
- **`data_config/fakevv_test_data_with_urls.json`**

The associated visual resources are hosted on Hugging Face:

- **Video frames:**  
  https://huggingface.co/datasets/fanrui00/FakeVV_testset

- **Downloaded (crawled) videos:**  
  https://huggingface.co/datasets/fanrui00/FakeVV_testset_video


### FakeVV Three-Stage Training Dataset

The annotation files for the three-stage FakeVV training data are hosted at:

- **Annotations & Metadata:**  
  https://huggingface.co/datasets/fanrui00/fact-r1-train-jsons  
  This repository also includes metadata for each video. Further details can be found in its included **README.md**.

- **Corresponding Video Files (.mp4):**  
  https://huggingface.co/datasets/fanrui00/FakeVV_trainset_video

- **Corresponding Video Frames (optional):**  
  https://huggingface.co/datasets/fanrui00/FakeVV_trainset


## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.49.0
- flash-attn>=2.4.3
- vllm>=0.7.3

### Hardware Requirements

\* *estimated*

| Method                   | Bits |  1.5B  |   3B   |   7B   |
| ------------------------ | ---- | ------ | ------ | ------ |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB |

### Installation

```bash
cd fact-r1
pip install -e .
```

### GRPO Training

```bash
bash examples/qwen2_5_vl_7b_fact_r1_grpo.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir path_to_your_last_actor_checkpoint
```


> [!NOTE]
> We will not provide scripts for Long-CoT Instruction Tuning and Preference Alignment via DPO in this project. If you have such requirements, we recommend using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).


## Thanks

We would like to thank the following repos for their great work: 

- This work is built upon the [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl).
