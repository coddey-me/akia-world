#!/usr/bin/env python3
"""
stage2_client.py
Two modes:
1. Simple inputs ("hi", "hello") -> AI small talk using local model
2. Real queries -> Stage2Agent summary + sources
3. Follow-up Q&A -> uses local AI with summary as context
"""

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import stage2_agentic as s2mod
from stage2_agentic import Stage2Agent
from stage2_client import try_stage2_agent, fallback_via_ddgs, is_useful_result, final_output_and_save

# -------------------------
# Load a free local AI model (distilgpt2 = lightweight)
# -------------------------
AI_MODEL_NAME = "distilgpt2"
_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(AI_MODEL_NAME)

def local_ai_chat(prompt, max_new_tokens=80):
    """Generate chat-like response using local GPT2 model."""
    inputs = _tokenizer.encode(prompt, return_tensors="pt")
    outputs = _model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=_tokenizer.eos_token_id,
    )
    text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


# -------------------------
# Modes
# -------------------------

def is_small_talk(user_input: str) -> bool:
    """Detect trivial greetings or chit-chat."""
    simple_inputs = {"hi", "hello", "hey", "yo", "sup", "how are you"}
    return user_input.lower().strip() in simple_inputs

def research_mode(query: str):
    """Run Stage2Agent pipeline for real queries."""
    agent = Stage2Agent(s2mod.INTENT_MODEL_DIR, s2mod.SUMMARIZER_MODEL)
    res = try_stage2_agent(agent, query)
    if not is_useful_result(res):
        res = fallback_via_ddgs(agent, query)
    final_output_and_save(res)
    return res

def followup_mode(summary: str, followup: str):
    """Use AI to expand/explain follow-up questions using summary as context."""
    prompt = f"Summary: {summary}\n\nUser follow-up: {followup}\n\nAnswer:"
    return local_ai_chat(prompt, max_new_tokens=100)


# -------------------------
# Main loop
# -------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage2_client.py \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()

    if is_small_talk(query):
        # Small talk mode
        reply = local_ai_chat(query)
        print(reply)
        return

    # Research mode
    res = research_mode(query)
    if not res:
        print("No reliable information found.")
        return

    # Extract summary for follow-ups
    summary = res.get("combined_result", {}).get("combined_summary", "")
    if not summary:
        summary = res.get("merged_intents", [{}])[0].get("summary", "")

    # Enter interactive loop for follow-up
    while True:
        follow = input("\nAsk a follow-up (or type 'exit'): ").strip()
        if follow.lower() in {"exit", "quit"}:
            break
        reply = followup_mode(summary, follow)
        print(reply)


if __name__ == "__main__":
    main()
