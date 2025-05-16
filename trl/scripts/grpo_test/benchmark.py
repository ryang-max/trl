
from datasets import load_dataset
from sglang import create_app, completion, GenerationConfig
from tqdm import tqdm

dataset = load_dataset("trl-lib/tldr", split="validation[-1%]")
prompts = [f"Summarize: {item['article']}" for item in dataset]

def reward_len(completions):
    return [-abs(20 - len(comp)) for comp in completions]

app = create_app("http://127.0.0.1:30000")

completions = []
for prompt in tqdm(prompts):
    response = completion(
        app,
        prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["\n"],
        config=GenerationConfig()
    )
    completions.append(response.text)

rewards = reward_len(completions)
avg_reward = sum(rewards) / len(rewards)

print(f"Average Reward: {avg_reward:.2f}")
