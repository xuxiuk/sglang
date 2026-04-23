python3 -m sglang.launch_server \
    --model /home/shared/models/Qwen/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning


CUDA_VISIBLE_DEVICES=2,3 SGLANG_TORCH_PROFILER_DIR=/home/zhouxuwen/sglang/profiles \
python3 -m sglang.launch_server \
    --model-path /home/shared/models/Qwen/Qwen3.5-35B-A3B \
    --tp 2 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 3 \
    --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.5 \
    --mamba-scheduler-strategy extra_buffer \
    --watchdog-timeout 1200 \
    --disable-cuda-graph \
    --log-level warning
    
    
    \
    --cuda-graph-max-bs 8



CUDA_VISIBLE_DEVICES=4,5 SGLANG_TORCH_PROFILER_DIR=/home/zhouxuwen/sglang/profiles \
    sglang serve \
    --model-path /home/shared/models/Qwen/Qwen3.5-35B-A3B \
    --tp 2 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 3 \
    --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.7 \
    --mamba-scheduler-strategy extra_buffer \
    --watchdog-timeout 3000 \
    --disable-cuda-graph \
    --log-level warning \
    --port 30000


python3 -m sglang.bench_serving --port 30000 --model /home/shared/models/Qwen/Qwen3.5-35B-A3B --dataset-name random --num-prompts 20

python3 -m sglang.profiler \
    --num-steps 5 \
    --cpu \
    --gpu \
    --output-dir /home/zhouxuwen/sglang/profiles \
    --merge-profiles

curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/home/zhouxuwen/sglang/profiles", 
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "merge_profiles": true 
  }'

python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --model /home/shared/models/Qwen/Qwen3.5-35B-A3B \
    --dataset-name sharegpt \
    --num-prompts 10 \
    --max-concurrency 1 \
    --random-input-len 256 \
    --random-output-len 16