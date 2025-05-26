set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

export WANDB_CONSOLE=off
export WANDB_MODE=offline

MODEL_PATH=qwen2_vl_lora_sft_reasoning   # replace it with your local file path

SYSTEM_PROMPT="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant \
first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., \
<think> reasoning process here </think><answer> answer here </answer>"""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data_config/d001.yaml \
    data.val_files=./data_config/d001.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.7 \
    trainer.experiment_name=Fact-R1-GRPOv2 \
    trainer.n_gpus_per_node=8
    # > train.log 2>&1