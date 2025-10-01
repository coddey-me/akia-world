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


# advanced_web_search.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import re
from collections import defaultdict

# ---------------------------
# URL Utilities
# ---------------------------
def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)

def clean_url(url: str) -> str:
    """
    Remove tracking parameters or query strings that break URL fetching.
    """
    url = url.split('&rut=')[0].split('?')[0]
    return url

# ---------------------------
# Webpage Content Fetching
# ---------------------------
def fetch_webpage_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, 'html.parser')
    # Remove non-content tags
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'form', 'aside']):
        tag.decompose()
    text = ' '.join(p.get_text(separator=' ', strip=True) for p in soup.find_all('p'))
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------------------
# HTML-based Search
# ---------------------------
def html_search(query: str, max_results: int = 5) -> list:
    """
    Returns a list of valid URLs from DuckDuckGo HTML search page.
    """
    search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Search request failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    urls = []

    for a in soup.find_all('a', href=True):
        href = a['href']
        # DuckDuckGo HTML redirect parsing
        match = re.search(r'uddg=(.+)', href)
        if match:
            url = unquote(match.group(1))
            url = clean_url(url)
            if is_valid_url(url):
                urls.append(url)
        if len(urls) >= max_results:
            break

    # Optional: check if URLs are reachable
    valid_urls = []
    for u in urls:
        try:
            r = requests.head(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                valid_urls.append(u)
        except:
            continue
    return valid_urls

# ---------------------------
# Main pipeline
# ---------------------------
def search_and_analyze(query: str, num_results: int = 5, chunk_size: int = 256, threshold: float = 0.5):
    """
    Full pipeline: search online, fetch pages, analyze content, merge summaries.
    """
    urls = html_search(query, max_results=num_results)
    if not urls:
        return {"query": query, "merged_intents": [], "message": "No URLs found"}

    combined_text = ""
    url_map = {}  # map chunk -> source URL
    for url in urls:
        content = fetch_webpage_text(url)
        if content:
            combined_text += content + " "
            url_map[content[:100]] = url  # map first 100 chars of chunk to source

    if not combined_text.strip():
        return {"query": query, "merged_intents": [], "message": "No content could be fetched"}

    # Run multi-intent summarization
    result = understand_multi_intent_summary(combined_text, chunk_size=chunk_size, threshold=0.0)

    # ---------------------------
    # Filter & merge by confidence
    # ---------------------------
    merged_intents = defaultdict(list)
    for r in result.get('merged_intents', []):
        if r['confidence'] >= threshold:
            merged_intents[r['intent']].append(r['summary'])

    # Create final list
    final_result = []
    for intent, summaries in merged_intents.items():
        merged_summary = " ".join(summaries)
        final_result.append({
            "intent": intent,
            "confidence": max([r['confidence'] for r in result.get('merged_intents', []) if r['intent']==intent]),
            "summary": merged_summary,
            "sources": urls
        })

    result['merged_intents'] = final_result
    return result

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    user_query = "Who established quantum mechanics, albert einstein or neils bohr"
    result = search_and_analyze(user_query, num_results=5, threshold=0.5)

    print(f"Query: {user_query}\n")
    print("Merged Intents with Summaries:\n")
    for i, r in enumerate(result.get('merged_intents', [])):
        print(f"Intent {i+1}: {r['intent']}, Confidence: {r['confidence']}")
        print(f"Summary: {r['summary']}")
        print(f"Sources: {', '.join(r['sources'])}\n")
