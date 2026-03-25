import pickle
import json
import ast
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import os
import torch
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdriver.planning.planning_prmopts import planning_system_message as system_message
from agentdriver.reasoning.collision_check import collision_check
from agentdriver.reasoning.collision_optimization import collision_optimization


# -------------------------------
# Load Qwen Model
# -------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # change to 3B if needed

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Fix pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------------
# Qwen conversation function
# -------------------------------

def run_qwen_conversation(system_message, user_message, temperature=0.0):

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    response = response[len(text):].strip()

    return {"content": response}


# -------------------------------
# Safe trajectory extraction
# -------------------------------

def extract_trajectory(text):
    try:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return np.array(ast.literal_eval(match.group()))
    except Exception as e:
        print("Trajectory parsing error:", e)
    return None


# -------------------------------
# Message Generator
# -------------------------------

def generate_messages(data_sample, use_peception=True, use_short_experience=True, verbose=True, use_gt_cot=False):

    token = data_sample["token"]
    ego = data_sample["ego"]
    perception = data_sample["perception"]
    commonsense = data_sample["commonsense"]
    experiences = data_sample["experiences"]
    reasoning = data_sample["reasoning"]

    long_experiences = data_sample.get("long_experiences", None)
    chain_of_thoughts = data_sample.get("chain_of_thoughts", "")
    planning_target = data_sample.get("planning_target", None)

    user_message = ego

    if use_peception:
        user_message += perception

    if use_short_experience:
        if experiences:
            user_message += experiences
    else:
        if long_experiences:
            user_message += long_experiences

    user_message += commonsense

    if use_gt_cot:
        user_message += chain_of_thoughts
    else:
        user_message += reasoning

    assistant_message = planning_target

    if verbose:
        print(user_message)
        print(assistant_message)

    return token, user_message, assistant_message


# -------------------------------
# Single Planning Inference
# -------------------------------

def planning_single_inference(
        planner_model_id,
        data_sample,
        data_dict=None,
        self_reflection=True,
        safe_margin=1.,
        occ_filter_range=5.0,
        sigma=1.0,
        alpha_collision=5.0,
        verbose=True
):

    token, user_message, assistant_message = generate_messages(data_sample, verbose=False)

    response_message = run_qwen_conversation(
        system_message=system_message,
        user_message=user_message,
        temperature=0.0
    )

    result = response_message["content"]

    if verbose:
        print(token)
        print(f"Qwen Planner:\n{result}")
        print(f"Ground Truth:\n{assistant_message}")

    output_dict = {
        "token": token,
        "Prediction": result,
        "Ground Truth": assistant_message,
    }

    traj = extract_trajectory(result)

    if traj is None:
        raise ValueError("Invalid trajectory format from model output")

    if self_reflection:
        assert data_dict is not None

        collision = collision_check(traj, data_dict, safe_margin=safe_margin, token=token)

        if collision.any():

            traj = collision_optimization(
                traj,
                data_dict,
                occ_filter_range=occ_filter_range,
                sigma=sigma,
                alpha_collision=alpha_collision
            )

            if verbose:
                print("Collision detected!")
                print(f"Optimized trajectory:\n{traj}")

    return traj, output_dict


# -------------------------------
# Batch Inference
# -------------------------------

def planning_batch_inference(data_samples, planner_model_id, data_path, save_path, self_reflection=True, verbose=False):

    save_path.mkdir(parents=True, exist_ok=True)
    save_file_name = save_path / "pred_trajs_dict.pkl"

    if os.path.exists(save_file_name):
        with open(save_file_name, "rb") as f:
            pred_trajs_dict = pickle.load(f)
    else:
        pred_trajs_dict = {}

    invalid_tokens = []

    for data_sample in tqdm(data_samples):

        token = data_sample["token"]

        try:
            data_dict_path = Path(data_path) / f"{token}.pkl"

            with open(data_dict_path, "rb") as f:
                data_dict = pickle.load(f)

            traj, output_dict = planning_single_inference(
                planner_model_id=planner_model_id,
                data_sample=data_sample,
                data_dict=data_dict,
                self_reflection=self_reflection,
                safe_margin=0.,
                occ_filter_range=5.0,
                sigma=1.265,
                alpha_collision=7.89,
                verbose=verbose
            )

            pred_trajs_dict[token] = traj

        except Exception as e:
            print("Error:", e)
            print(f"Invalid token: {token}")
            invalid_tokens.append(token)
            continue

    print("#### Invalid Tokens ####")
    print(invalid_tokens)

    with open(save_file_name, "wb") as f:
        pickle.dump(pred_trajs_dict, f)

    return pred_trajs_dict


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":

    current_time = time.strftime("%D:%H:%M").replace("/", "_").replace(":", "_")

    save_path = Path("experiments/outputs") / current_time

    # Replace with real dataset
    data_samples = []
    data_path = Path("data")

    pred_trajs_dict = planning_batch_inference(
        data_samples=data_samples,
        planner_model_id="qwen2",
        data_path=data_path,
        save_path=save_path,
        verbose=False,
    )