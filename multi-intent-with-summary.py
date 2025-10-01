# multi_intent_with_summary.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json, os
from typing import List, Dict

# ---------------------------
# Load model, tokenizer, labels
# ---------------------------
MODEL_DIR = "checkpoints/intent"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, "label2id.json")) as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ---------------------------
# Utility: chunk text
# ---------------------------
def chunk_text(text: str, max_tokens: int = 256) -> List[Dict]:
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append({"text": chunk_text, "start_token": i, "end_token": i + len(chunk_tokens)})
    return chunks

# ---------------------------
# Predict intent per chunk
# ---------------------------
def predict_chunk_intent(text_chunk: str) -> Dict:
    inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]
    
    pred_id = int(torch.argmax(probs))
    confidence = float(probs[pred_id])
    
    return {
        "intent": id2label[pred_id],
        "confidence": confidence
    }

# ---------------------------
# Merge adjacent chunks with same intent
# ---------------------------
def merge_chunks(chunks: List[Dict], threshold: float = 0.5) -> List[Dict]:
    merged = []
    current = None

    for c in chunks:
        if c["confidence"] < threshold:
            continue
        if current is None:
            current = c.copy()
        elif c["intent"] == current["intent"]:
            # Merge text and update end_token and confidence
            current["text"] += " " + c["text"]
            current["end_token"] = c["end_token"]
            current["confidence"] = (current["confidence"] + c["confidence"]) / 2
        else:
            merged.append(current)
            current = c.copy()
    if current:
        merged.append(current)
    return merged

# ---------------------------
# Optional simple extractive summary
# ---------------------------
def summarize_text(text: str, max_sentences: int = 3) -> str:
    # Simple split by sentence periods
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")

# ---------------------------
# Full multi-intent pipeline with merged summaries
# ---------------------------
def understand_multi_intent_summary(text: str, chunk_size: int = 256, threshold: float = 0.5) -> Dict:
    chunks = chunk_text(text, max_tokens=chunk_size)
    
    # Predict intent per chunk
    chunk_results = []
    for c in chunks:
        res = predict_chunk_intent(c["text"])
        res.update({"start_token": c["start_token"], "end_token": c["end_token"], "text": c["text"]})
        chunk_results.append(res)
    
    # Merge adjacent chunks
    merged_chunks = merge_chunks(chunk_results, threshold=threshold)
    
    # Summarize each merged chunk
    for mc in merged_chunks:
        mc["summary"] = summarize_text(mc["text"])
        mc["confidence"] = round(mc["confidence"], 4)
    
    return {
        "original_text": text,
        "merged_intents": merged_chunks
    }

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    long_text = """
    Compare Tesla and BYD in terms of battery and range. 
    Also provide a brief summary of their pricing models. 
    Then explain which one is better for long-distance travel. 
    Finally, include any news about recent EV launches.
    """

    result = understand_multi_intent_summary(long_text, threshold=0.2)
    print("Merged Intents with Summaries:\n")
    for i, r in enumerate(result["merged_intents"]):
        print(f"Intent {i+1}: {r['intent']}, Confidence: {r['confidence']}")
        print(f"Summary: {r['summary']}\n")

