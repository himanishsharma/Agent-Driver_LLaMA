from transformers import AutoTokenizer, AutoModelForVision2Seq
from peft import PeftModel

base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_path = "/home/jovyan/xz/LLaMA-Factory/saves/qwen2_5vl-3b/lora/sft/checkpoint-5848"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

# ✅ correct model class
model = AutoModelForVision2Seq.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True
)

# attach adapter
model = PeftModel.from_pretrained(model, adapter_path)

print("Loaded VL model successfully ✅")