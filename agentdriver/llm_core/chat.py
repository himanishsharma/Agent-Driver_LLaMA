# Basic Chat Completion Functions (LLaMA / Qwen version)

import json
from typing import List, Dict
from agentdriver.llm_core.timeout import timeout


# 🔥 LOCAL LLM CALL
def local_llm_chat(model, tokenizer, messages, temperature=0.0):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=(temperature > 0)
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response_text


@timeout(15)
def run_one_round_conversation(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str,
        model,
        tokenizer,
        temperature: float = 0.0,
    ):
    """
    Perform one round of conversation using LOCAL LLM
    """

    message_for_this_round = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ] if system_message else [{"role": "user", "content": user_message}]
    
    full_messages.extend(message_for_this_round)

    # 🔥 LLaMA/Qwen call
    response_text = local_llm_chat(model, tokenizer, full_messages, temperature)

    response_message = {
        "role": "assistant",
        "content": response_text
    }

    # Append assistant's reply
    full_messages.append(response_message)

    return full_messages, response_message


def run_one_round_conversation_with_functional_call(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str, 
        functional_calls_info: List[Dict],
        model,
        tokenizer,
        temperature: float = 0.0,
    ):
    """
    Functional calls NOT supported directly in LLaMA
    → fallback to normal generation
    """

    message_for_this_round = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ] if system_message else [{"role": "user", "content": user_message}]
    
    full_messages.extend(message_for_this_round)

    # 🔥 Just generate text
    response_text = local_llm_chat(model, tokenizer, full_messages, temperature)

    response_message = {
        "role": "assistant",
        "content": response_text
    }

    full_messages.append(response_message)
    
    return full_messages, response_message