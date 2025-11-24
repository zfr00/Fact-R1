# Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning

## FakeVV dataset
Here, we present a portion of the dataset used for the three-stage training process. Please note that the training sets for the third stage, which include [fakett](https://github.com/ICTMCG/FakingRecipe) and [fakesv](https://github.com/ICTMCG/FakeSV), are not displayed due to access restrictions.

- data_config/long_cot_random_sampled_data.json
- data_config/dpo_training_data_sampled.json
- data_config/grpo_training_data_sampled.jsonl

### FakeVV dataset testset
The complete test dataset is defined in:
- **`data_config/reasoning_training_data_image_test_with_audio.json`**
Includes webpage url
- **`data_config/reasoning_training_data_image_test_with_audio_urls.json`**

The associated visual resources are hosted on Hugging Face:

- **Video frames:**  
  https://huggingface.co/datasets/fanrui00/FakeVV_testset

- **Downloaded (crawled) videos:**  
  https://huggingface.co/datasets/fanrui00/FakeVV_testset_video


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
