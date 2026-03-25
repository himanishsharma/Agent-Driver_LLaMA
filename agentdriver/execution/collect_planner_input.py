from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
from peft import PeftModel

from agentdriver.main.language_agent import LanguageAgent

# ✅ YOUR ACTUAL MODEL
base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_path = "/home/jovyan/xz/LLaMA-Factory/saves/qwen2_5vl-3b/lora/sft/checkpoint-5848"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# base model
model = AutoModelForVision2Seq.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# attach YOUR trained adapter
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()


def qwen_chat(messages, max_tokens=512, temperature=0.7):

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


if __name__ == "__main__":
    data_path = Path("data/")
    split = "train"

    language_agent = LanguageAgent(
        data_path,
        split,
        model_name="qwen2",
        finetune_cot=False,
        verbose=False
    )

    language_agent.llm = qwen_chat

    language_agent.collect_planner_input(invalid_tokens=None)