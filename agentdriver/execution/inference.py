from pathlib import Path
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from agentdriver.main.language_agent import LanguageAgent
from agentdriver.llm_core.api_keys import FINETUNE_PLANNER_NAME

# Load model
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if __name__ == "__main__":
    data_path = Path('data/')
    split = 'val'

    language_agent = LanguageAgent(
        data_path,
        split,
        model=model,
        tokenizer=tokenizer,
        planner_model_name=FINETUNE_PLANNER_NAME,
        finetune_cot=False,
        verbose=False
    )

    current_time = time.strftime("%D:%H:%M").replace("/", "_").replace(":", "_")
    save_path = Path("experiments") / current_time
    save_path.mkdir(exist_ok=True, parents=True)

    with open("data/finetune/data_samples_val.json", "r") as f:
        data_samples = json.load(f)

    planning_traj_dict = language_agent.inference_all(
        data_samples=data_samples,
        data_path=data_path / split,
        save_path=save_path,
    )