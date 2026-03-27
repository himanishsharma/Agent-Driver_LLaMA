from transformers import AutoTokenizer, AutoModelForVision2Seq
from peft import PeftModel

class ChatLLAMA():
    def __init__(self):
        base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
        adapter_path = "/home/jovyan/xz/LLaMA-Factory/saves/qwen2_5vl-3b/lora/sft/checkpoint-5848"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(model, adapter_path)
        
        print("Loaded VL model successfully ✅")

    def chat(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()

        return response
