export PYTHONPATH=$PYTHONPATH:$(pwd)
# The optional values for the method argument are 'anchor-token', 'normal', 'kvcache', and 'anchor-thought'.
method="anchor-thought"
tokenizer_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-1.5B-Instruct"
comp_config="configs/LightThinker/qwen/v1.json"
model_type="qwen"
dataset="gsm8k" # gsm8k gpqa mmlu bbh
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
folder="inf_lightthinker_r1distillqwen1.5b"
ckpt=1305
file1="inference_results/${folder}/${dataset}/${ckpt}/1-4${folder}.jsonl"
file2="inference_results/${folder}/${dataset}/${ckpt}/2-4${folder}.jsonl"
file3="inference_results/${folder}/${dataset}/${ckpt}/3-4${folder}.jsonl"
file4="inference_results/${folder}/${dataset}/${ckpt}/4-4${folder}.jsonl"
# python evaluation/eval_file.py \
#   --method $method \
#   --tokenizer_path $tokenizer_path \
#   --comp_config $comp_config \
#   --model_type $model_type \
#   --dataset $dataset \
#   --files $file1 $file2 $file3 $file4 \
#   --cache_size $cache_size \
#   --bos_token $bos_token \
#   --eos_token $eos_token 
#   # --interaction 


# 评估sglang推理的norml
method="normal"
dataset="mmlu" # gsm8k gpqa mmlu bbh
folder="inf_baseline_r1distillqwen1.5b"
file="sglang_inference_results/${folder}/${dataset}_result.jsonl"
python evaluation/eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files $file \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token 
  # --interaction 