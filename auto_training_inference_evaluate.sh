#!/bin/bash


# eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
# conda activate lightthinker
# echo "Using python: $(which python)"


# ROOT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot"
# cd $ROOT_DIR
# export PYTHONPATH=$PYTHONPATH:$(pwd)

# echo "[INFO] Working directory: $(pwd)"


# echo "=======🚀 lightthinker开始训练 ======="
# bash ./our_train_lighthinker.sh
# echo "=======🚀 lightthinker结束训练 ======="

# echo "=======🚀 lightthinker开始推理 ======="
# bash ./our_inference_repe.sh
# echo "=======🚀 lightthinker结束推理 ======="


# echo "=======🚀 lightthinker_EPL开始训练 ======="
# bash ./our_train_lighthinker_epl.sh
# echo "=======🚀 lightthinker_EPL结束训练 ======="


# echo "=======🚀 lightthinker_EPL开始推理 ======="
# bash ./our_inference_epl_repe.sh
# echo "=======🚀 lightthinker_EPL结束推理 ======="


# echo "=======🚀 lightthinker_EPL_MTP开始训练 ======="
# bash ./our_train_lighthinker_epl_mtp.sh
# echo "=======🚀 lightthinker_EPL_MTP结束训练 ======="

# echo "=======🚀 lightthinker_EPL_MTP开始推理 ======="
# bash ./our_inference_epl_mtp_repe.sh
# echo "=======🚀 lightthinker_EPL_MTP结束推理 ======="

# echo "=======🚀 vanilla开始训练 ======="
# bash ./our_train_baseline.sh
# echo "=======🚀 vanilla结束训练 ======="


# source /mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh
# conda activate niah
# echo "Using python: $(which python)"

# echo "=======🚀 vanilla开始推理 ======="
# bash ./our_sglang_infer.sh
# echo "=======🚀 vanilla结束推理 ======="

# source /mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh
# conda activate lightthinker
# echo "Using python: $(which python)"

# echo "=========================================="
# echo "      🚀 lightthinker评估开始     "
# echo "=========================================="
# output_path="/tmp/hx/rrcot/lightthinker"
# model_tag="lightthinker"
# ckpt="1305"
# output_tag="${output_path}/${model_tag}/inference"
# bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh
# bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa
# bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k
# bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu
# echo "=========================================="
# echo "      🚀 lightthinker评估完成      "
# echo "=========================================="



# echo "=========================================="
# echo "      🚀 lightthinker_epl评估开始     "
# echo "=========================================="
# output_path="/tmp/hx/rrcot/lightthinker_epl"
# model_tag="lightthinker_epl"
# ckpt="1305"
# output_tag="${output_path}/${model_tag}/inference"
# bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh
# bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa
# bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k
# bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu
# echo "=========================================="
# echo "      🚀 lightthinker_epl评估完成      "
# echo "=========================================="


# echo "=========================================="
# echo "      🚀 lightthinker_epl_mtp评估开始     "
# echo "=========================================="
# output_path="/tmp/hx/rrcot/lightthinker_epl_mtp"
# model_tag="lightthinker_epl_mtp"
# ckpt="1305"
# output_tag="${output_path}/${model_tag}/inference"
# bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh
# bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa
# bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k
# bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu
# echo "=========================================="
# echo "      🚀 lightthinker_epl_mtp评估完成      "
# echo "=========================================="

# 换vanilla环境


# echo "=========================================="
# echo "      🚀 vanilla评估开始     "
# echo "=========================================="
# output_path="/tmp/hx/rrcot/vanilla"
# model_tag="vanilla"
# ckpt="1305"
# output_tag="${output_path}/${model_tag}/inference"
# bash our_evaluate.sh --method normal --dataset bbh --base_path ${output_tag}/bbh
# bash our_evaluate.sh --method normal --dataset gpqa --base_path ${output_tag}/gpqa
# bash our_evaluate.sh --method normal --dataset gsm8k --base_path ${output_tag}/gsm8k
# bash our_evaluate.sh --method normal --dataset mmlu --base_path ${output_tag}/mmlu
# echo "=========================================="
# echo "      🚀 vanilla评估完成      "
# echo "=========================================="



# echo "=======🚀 lightthinker_EPL_MTP_midlayer开始训练 ======="
# bash ./our_train_lighthinker_epl_mtp_midlayer.sh
# echo "=======🚀 lightthinker_EPL_MTP_midlayer结束训练 ======="

# echo "=======🚀 lightthinker_EPL_MTP_midlayer开始推理 ======="
# bash ./our_inference_epl_mtp_midlayer.sh
# echo "=======🚀 lightthinker_EPL_MTP_midlayer结束推理 ======="


# echo "=========================================="
# echo "      🚀 lightthinker_epl_mtp_midlayer评估开始     "
# echo "=========================================="
# output_path="/tmp/hx/rrcot/lightthinker_epl_mtp_midlayer"
# model_tag="lightthinker_epl_mtp_midlayer"
# ckpt="1305"
# output_tag="${output_path}/${model_tag}/inference"
# bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh
# bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa
# bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k
# bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu
# echo "=========================================="
# echo "      🚀 lightthinker_epl_mtp_midlayer评估完成      "
# echo "=========================================="



# cross attention MTP 对比试验

echo "=======🚀 lightthinker_epl_mtp_128开始训练 ======="
bash ./our_train_lighthinker_epl_mtp_128.sh
echo "=======🚀 lightthinker_epl_mtp_128结束训练 ======="

echo "=======🚀 lightthinker_epl_mtp_128开始推理 ======="
bash ./our_inference_epl_mtp_repe_128.sh
echo "=======🚀 lightthinker_epl_mtp_128结束推理 ======="

echo "=========================================="
echo "      🚀 lightthinker_epl_mtp_128评估开始     "
echo "=========================================="
output_path="/tmp/hx/rrcot/lightthinker_epl_mtp_128"
model_tag="lightthinker_epl_mtp_128"
ckpt="650"
output_tag="${output_path}/${model_tag}/inference"
bash our_evaluate.sh --method anchor-thought --dataset bbh --base_path ${output_tag}/bbh
bash our_evaluate.sh --method anchor-thought --dataset gpqa --base_path ${output_tag}/gpqa
bash our_evaluate.sh --method anchor-thought --dataset gsm8k --base_path ${output_tag}/gsm8k
bash our_evaluate.sh --method anchor-thought --dataset mmlu --base_path ${output_tag}/mmlu
echo "=========================================="
echo "      🚀 lightthinker_epl_mtp_128评估完成      "
echo "=========================================="