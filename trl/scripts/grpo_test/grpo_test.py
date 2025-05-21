# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import torch
import subprocess
import time
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer



# ==================== Dataset ==========================
raw_dataset = load_dataset("gsm8k", "main", split={"train": "train[:10%]", "test": "test"})
train_dataset = raw_dataset["train"].map(lambda x: {"prompt": x["question"], "completion": x["answer"]})
eval_dataset = raw_dataset["test"]

checkpoint_dir = os.path.join("/sgl-workspace/ryang/trl", "checkpoints/sgl")
os.makedirs(checkpoint_dir, exist_ok=True)

# ==================== Reward Function ==================
def reward_exact_answer(completions, references, **kwargs):
    return [float(pred.strip() == ref.strip()) for pred, ref in zip(completions, references)]

# ==================== Evaluation Automation ============
def restart_sglang_server(checkpoint_path, port=30001, device=1):
    subprocess.run(["pkill", "-f", f"sglang.launch_server --port {port}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    launch_cmd = f"CUDA_VISIBLE_DEVICES={device} nohup python3 -m sglang.launch_server --model-path {checkpoint_path} --port {port} > sglang_eval_log.txt 2>&1 &"
    subprocess.run(launch_cmd, shell=True)


def wait_for_eval_server_ready(port=30001, timeout=60):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health")
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang eval server on port {port} failed to start.")


def run_benchmark(port=30001, num_questions=100, result_file="gsm8k_scores.txt"):
    cmd = [
        "python3", "bench_sglang.py",
        "--num-questions", str(num_questions),
        "--result-file", result_file,
        "--backend", "srt",             
        "--port", str(port)        
    ]
    subprocess.run(cmd)

# ==================== Training Config ===================
# (Removed explicit GPU allocation; set via command-line instead)  # Allocates GPU 1 and 2 for training

training_args = GRPOConfig(
    output_dir=os.path.join(checkpoint_dir, "Qwen2.5_output"),
    logging_steps=10,
    use_sglang=True,
    sglang_device="cuda:1",
    sglang_gpu_memory_utilization=0.9,
    sglang_server_url="http://127.0.0.1:30000",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_exact_answer,
    args=training_args,
    train_dataset=train_dataset,
)

training_args.checkpoint_path = checkpoint_dir

# ==================== Training Loop =====================
train_dataloader = trainer.get_train_dataloader()
train_iter = iter(train_dataloader)

eval_every = 10
max_steps = 2000
step = 0

result_file = "gsm8k_scores_all_steps.txt"

while step < max_steps:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        batch = next(train_iter)

    # Single training step
    loss = trainer.training_step(trainer.model, batch)
    step += 1

    # Optional: print loss
    # print(f"[Step {step}] loss = {loss.item() if hasattr(loss, 'item') else loss}")

    # Run evaluation checkpoint
    if step % eval_every == 0:
        ckpt_path = os.path.join(training_args.output_dir, f"step_{step:05d}")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        trainer.model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path) 
        # trainer.tokenizer.save_pretrained(ckpt_path)

        print(f"[Step {step}] Restarting SGLang with checkpoint: {ckpt_path}")
        restart_sglang_server(ckpt_path, port=30001, device=1)
        wait_for_eval_server_ready(port=30001)
        benchmark_output = run_benchmark(
            port=30001,
            num_questions=100,
            result_file= None
        )

        with open(result_file, "a") as f:
            f.write(f"\n=== Step {step} ===\n")
            f.write(benchmark_output.strip() + "\n")

        print(f"[Step {step}] Benchmark complete.\n")

# ==================== Final Save ========================
final_ckpt_path = os.path.join(training_args.output_dir, "final")
trainer.model.save_pretrained(final_ckpt_path)
print("[Final] Saved final model checkpoint.")