# # 替换成你的conda path
# eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
# which conda
# conda activate lightthinker
# which python

# # cd到你的项目路径
# cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot

export PYTHONPATH=$PYTHONPATH:$(pwd)
# we will load model from `output/{model_tag}/checkpoint-{args.ckpt}`


# ============================================== 修改评测模型换这里的配置就行 =========================================================
model_tag="lightthinker_epl_mtp_128"
model_short_tag="lightthinker_epl_mtp_128"
repetition_penalty=1.1
ckpt=650
output_path="/tmp/hx/rrcot/lightthinker_epl_mtp_128"
output_tag="${output_path}/${model_tag}/inference"
model_path="/tmp/hx/rrcot/lightthinker_epl_mtp_128/output/lightthinker_epl_mtp_128/checkpoint-650"
# ================================== zrs修改保存路径 ==========================================
# ================================================================================================================================


model_type="qwen"
tokenizer_path="/tmp/hx/Qwen/Qwen2.5-1.5B-Instruct"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
compress_config="./configs/LightThinker/qwen/v1.json"

# `model_path` is an optional argument
# if you set the `model_path`, the arguments `ckpt` and `model_tag` will be ignored.
# see line 1460 of the code in LightThinker/inference.py for more details.
max_new_tokens=10240

root_dir="./LightThinker"

prefix=""
diagonal="false"
see_current="false"
compress_prompt="false"
rolling_rope="false"
bi_directional="false"
exclude_continue="false"
output_compress_instruction="None"
prefill_compress="false"
update_attention_method="local"


# check "ours_infer_log" 
if [ ! -d "${output_path}/ours_infer_log" ]; then
    echo "Creating ${output_path}/ours_infer_log directory..."
    mkdir -p "${output_path}/ours_infer_log"
fi

subfolders=("true_true" "true_false" "false_false" "false_true")
for subfolder in "${subfolders[@]}"; do
    folder_path="${output_path}/ours_infer_log/${subfolder}"
    if [ ! -d "$folder_path" ]; then
        echo "Creating $folder_path directory..."
        mkdir -p "$folder_path"
    fi
done



echo "Inference model: ${model_tag}..."

#用于设置总共几张卡和开多少进程
target_gpus=( 0 1 2 3 )
process_per_gpu=4
gpu_count=${#target_gpus[@]}
# 自动计算总切片数 (假如用了2张卡，每张3进程，split_size就是6)
split_size=$((gpu_count * process_per_gpu))

logical_id=0
for device in "${target_gpus[@]}"
do
    # 计算当前显卡负责的 "0-based" 索引范围 (例如 0,1,2)
    start_index_0based=$((logical_id * process_per_gpu))
    end_index_0based=$((start_index_0based + process_per_gpu - 1))
    echo ">>> Launching on Physical GPU ${device}" 
    for ((idx=start_index_0based; idx<=end_index_0based; idx++))
    do
        real_index=$((idx + 1))
        
        echo "    Starting task index ${real_index}/${split_size}..."

        # 评测EPL训练模型时 --EPL=True 
        CUDA_VISIBLE_DEVICES=$device nohup python "${root_dir}/inference_repe.py" \
            --model_tag $model_tag \
            --model_short_tag $model_short_tag \
            --ckpt $ckpt \
            --tokenizer_path $tokenizer_path \
            --compress_config $compress_config \
            --max_new_tokens $max_new_tokens \
            --repetition_penalty $repetition_penalty \
            --output_tag $output_tag \
            --model_type $model_type \
            --bos_token $bos_token \
            --eos_token $eos_token \
            --rolling_rope $rolling_rope \
            --diagonal $diagonal \
            --bi_directional $bi_directional \
            --see_current $see_current \
            --exclude_continue $exclude_continue \
            --output_compress_instruction $output_compress_instruction \
            --prefill_compress $prefill_compress \
            --compress_prompt $compress_prompt \
            --update_attention_method $update_attention_method \
            --split_size $split_size \
            --use_EPL True \
            --model_path $model_path \
            --index $real_index > "${output_path}/ours_infer_log/${rolling_rope}_${compress_prompt}/${real_index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &
        
        sleep 5
    done
    ((logical_id++))
done

echo ""
echo "=========================================="
echo "All processes launched. Waiting for completion..."
echo "Started at: $(date)"
echo "=========================================="

wait  # 等待所有后台进程（&）完成

echo ""
echo "=========================================="
echo "All processes completed!"
echo "Finished at: $(date)"
echo "=========================================="