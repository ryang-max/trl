import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm


def generate_outputs(model, tokenizer, prompts, device, max_new_tokens=100):
    outputs = []
    for prompt in tqdm(prompts, desc="Generating"):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output)
    return outputs


def compute_rouge(generations, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, gen) for ref, gen in zip(references, generations)]

    def avg(metric):
        return sum(s[metric].fmeasure for s in scores) / len(scores)

    return {
        "rouge1": avg("rouge1"),
        "rouge2": avg("rouge2"),
        "rougeL": avg("rougeL"),
    }


def run_benchmark():
    split = "train[-1%:]" 
    max_samples = 100
    base_model_path = "Qwen/Qwen2-0.5B-Instruct"
    finetuned_model_path = "/sgl-workspace/ryang/trl/checkpoints/sgl/Qwen2.5_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("trl-lib/tldr", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    prompts = [x["prompt"] for x in dataset]
    references = [x["completion"] for x in dataset]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to(device)
    base_outputs = generate_outputs(base_model, tokenizer, prompts, device)
    base_scores = compute_rouge(base_outputs, references)

    print("\n=== Base Model ROUGE ===")
    for k, v in base_scores.items():
        print(f"{k}: {v:.4f}")

    # === Finetuned Model ===
    finetuned_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    finetuned_model.load_state_dict(torch.load(os.path.join(finetuned_model_path, "sgl_weights.pt")))

    finetuned_outputs = generate_outputs(finetuned_model, tokenizer, prompts, device)
    finetuned_scores = compute_rouge(finetuned_outputs, references)

    print("\n=== Finetuned Model ROUGE ===")
    for k, v in finetuned_scores.items():
        print(f"{k}: {v:.4f}")

    # print("\n=== Sample Outputs ===")
    # for i in range(min(3, len(prompts))):
    #     print(f"\n--- Prompt {i+1} ---\n{prompts[i]}")
    #     print(f"[Base] {base_outputs[i]}")
    #     # print(f"[Finetuned] {finetuned_outputs[i]}")
    #     print(f"[Reference] {references[i]}")


if __name__ == "__main__":
    run_benchmark()
