## finetuning motion planner using Qwen2

import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from agentdriver.execution.gen_finetune_data import generate_traj_finetune_data


MODEL_NAME = "Qwen/Qwen2-7B-Instruct"   # change if using different size


if __name__ == "__main__":

    print("Generating fine-tuning data ...")

    generate_traj_finetune_data(
        data_path="data/finetune",
        data_file="data_samples_train.json",
        sample_ratio=0.1,
        use_gt_cot=False
    )

    dataset_path = "data/finetune/finetune_planner_10.json"


    # -----------------------
    # Load dataset
    # -----------------------

    dataset = load_dataset(
        "json",
        data_files=dataset_path,
        split="train"
    )


    # -----------------------
    # Load tokenizer + model
    # -----------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )


    # -----------------------
    # LoRA configuration
    # -----------------------

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    # -----------------------
    # Training configuration
    # -----------------------

    training_args = TrainingArguments(

        output_dir="finetuned_qwen_planner",

        num_train_epochs=1,

        per_device_train_batch_size=2,

        gradient_accumulation_steps=4,

        learning_rate=2e-4,

        logging_steps=10,

        save_steps=200,

        fp16=True,

        report_to="none"
    )


    # -----------------------
    # Trainer
    # -----------------------

    trainer = SFTTrainer(

        model=model,

        train_dataset=dataset,

        tokenizer=tokenizer,

        args=training_args,

        peft_config=peft_config,

        dataset_text_field="text",   # field in JSON file
    )


    print("Starting Qwen2 fine-tuning...")

    trainer.train()


    print("Saving model...")

    trainer.save_model("finetuned_qwen_planner")

    tokenizer.save_pretrained("finetuned_qwen_planner")


    print("Fine-tuning complete.")