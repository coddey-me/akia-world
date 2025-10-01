#!/usr/bin/env python3
"""
Stage 2: Agentic Query Processing for Akia World

Usage:
    python stage2_agentic.py

This script expects a trained intent model saved under checkpoints/intent
with tokenizer files and label2id.json.

Outputs a JSON-like structure printed to stdout with:
 - decomposed queries (subqueries)
 - per-intent merged summaries with sources and confidence
 - coverage / gap report for canonical facets
"""

import re
import time
import json
import math
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, unquote
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

# ---------- CONFIG ----------
INTENT_MODEL_DIR = "checkpoints/intent"   # your fine-tuned intent model dir
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # CPU-friendly summarizer
USER_AGENT = "Mozilla/5.0 (compatible; AkiaStage2/1.0; +https://example.com)"
DEFAULT_MAX_CANDIDATES = 15
MIN_CONTENT_CHUNKS = 3   # if less than this, agent will retry with more candidates
CHUNK_WORDS = 300
INTENT_CONFIDENCE_THRESHOLD = 0.45  # include chunk-level intent predictions above this
FACETS = ["research", "comparison", "range", "autonomy", "safety", "manufacturing", "sales", "charging", "software"]

# ---------- Utilities ----------
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

def cleanup_rut_query_param(url: str) -> str:
    """
    Remove tracking query parameters such as 'rut' and others, preserving other queries.
    """
    try:
        parsed = urlparse(url)
        query_params = parse_qsl(parsed.query, keep_blank_values=True)
        # Filter out tracking params
        filtered_params = [(k, v) for k, v in query_params if k.lower() not in ('rut', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content')]
        # Rebuild query string without tracking params
        new_query = urlencode(filtered_params, doseq=True)
        cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ''))
        return cleaned
    except Exception:
        return url

import re

def fix_malformed_path(url: str) -> str:
    """
    Fix URLs where tracking params are appended as path segments starting with &rut= or &.
    Remove trailing '/&rut=...' or '/&...' at the end of path.
    """
    # Unquote URL first in case URLs are encoded
    url = unquote(url)

    parsed = urlparse(url)

    # 1) Fix malformed trailing path: remove segments like /&rut=... or /&...
    new_path = re.sub(r'/&rut=[^/?#]*$', '', parsed.path)
    new_path = re.sub(r'/&[^/?#]*$', '', new_path)

    # 2) Remove 'rut' and UTM tracking params from query string
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_params = [(k, v) for k, v in query_params if k.lower() not in ('rut', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content')]
    new_query = urlencode(filtered_params, doseq=True)

    # Rebuild cleaned URL
    cleaned = urlunparse((
        parsed.scheme,
        parsed.netloc,
        new_path,
        parsed.params,
        new_query,
        ''  # strip fragment as well
    ))

    # Normalize trailing slash off
    if cleaned.endswith('/'):
        cleaned = cleaned[:-1]

    return cleaned



def strip_all_query(url: str) -> str:
    """Remove the entire query and fragment, keeping only scheme+netloc+path."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

def is_valid_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def safe_head(url: str, timeout=5) -> bool:
    """Return True if HEAD returns 200. Use GET fallback when HEAD not allowed."""
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return True
        # Some servers block HEAD; try GET with range 0 to minimize cost
        r2 = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if r2.status_code == 200:
            return True
    except Exception:
        return False
    return False

def fetch_page_text(url: str, timeout=12) -> str:
    """Fetch page HTML and extract main paragraph text; return empty string on error."""
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        # remove boilerplate tags
        for t in soup(["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]):
            t.decompose()
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paragraphs if p)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"Error parsing HTML from {url}: {e}")
        return ""

def chunk_text_by_words(text: str, max_words: int = CHUNK_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

# ---------- Search (DuckDuckGo HTML) ----------
def duckduckgo_html_search(query: str, max_candidates: int = 15) -> List[str]:
    """Return list of raw URLs (may include redirect wrappers)."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.requote_uri(query)}"
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        urls = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # DuckDuckGo HTML uses /l/?kh=-1&uddg=<encoded_url> or direct links
            m = re.search(r"uddg=(.+)", href)
            if m:
                candidate = unquote(m.group(1))
            else:
                candidate = href
            # Add candidate
            if candidate and is_valid_url(candidate):
                urls.append(candidate)
            if len(urls) >= max_candidates:
                break
        return urls
    except Exception as e:
        print(f"Search failed: {e}")
        return []

# ---------- Load intent model ----------
def load_intent_model(model_dir: str):
    """Loads tokenizer, model and label mapping from model_dir; returns predict_fn."""
    import os
    label_map_path = os.path.join(model_dir, "label2id.json")
    if not os.path.exists(model_dir) or not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Intent model/labels not found in {model_dir}. Ensure you've saved tokenizer and label2id.json.")
    with open(label_map_path, "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict(texts: List[str]) -> List[List[Tuple[str, float]]]:
        """Return for each text a list of (label, score) sorted desc."""
        batches = []
        results = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
                logits = out.logits.cpu().numpy()
                probs = softmax(logits)
                for p in probs:
                    # p is array of probabilities per label index in label2id order
                    label_scores = [(id2label[idx], float(p[idx])) for idx in range(len(p))]
                    label_scores.sort(key=lambda x: x[1], reverse=True)
                    results.append(label_scores)
        return results

    return predict

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ---------- Summarizer ----------
def load_summarizer(model_name: str = SUMMARIZER_MODEL):
    try:
        summarizer = pipeline("summarization", model=model_name, device=-1)  # device -1 = CPU
        return summarizer
    except Exception as e:
        print(f"Failed to load summarizer {model_name}: {e}")
        raise

# ---------- Query decomposition (simple) ----------
def decompose_query(user_query: str) -> List[str]:
    """Produce sub-queries focusing on canonical facets. Very simple rule-based decomposition."""
    base = user_query.strip().rstrip(".?")
    subqueries = [base]  # include base
    # Append facet-specific subqueries
    for f in FACETS:
        subqueries.append(f"{base} {f}")
        subqueries.append(f"{base} {f} comparison")
    # also add short variants (token-based)
    tokens = re.split(r"[,\;]", user_query)
    for t in tokens:
        t = t.strip()
        if t and len(t.split()) > 1:
            subqueries.append(t)
    # make unique preserving order
    seen = set()
    unique = []
    for q in subqueries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    # keep it reasonable
    return unique[:20]




# ---------- Agentic pipeline ----------
class Stage2Agent:
    def __init__(self,
                 intent_model_dir: str = INTENT_MODEL_DIR,
                 summarizer_model: str = SUMMARIZER_MODEL):
        print("Loading intent model from:", intent_model_dir)
        self.intent_predict = load_intent_model(intent_model_dir)
        print("Loading summarizer model:", summarizer_model)
        self.summarizer = load_summarizer(summarizer_model)
    def combine_top_results(self, merged_intents: List[Dict], top_n: int = 3) -> Dict:
        """
        Combine top N intents by confidence into one aggregate summary and source list.
        """
        top_intents = merged_intents[:top_n]
        all_sources = set()
        combined_sentences = []
        for intent_data in top_intents:
            # Aggregate sources
            for src in intent_data.get("sources", []):
                all_sources.add(src)
            # Split summary into sentences and add unique
            for sent in re.split(r"(?<=[.!?])\s+", intent_data["summary"]):
                sent = sent.strip()
                if sent and sent not in combined_sentences:
                    combined_sentences.append(sent)
        # Create combined summary text (limit length e.g. first 8 sentences)
        combined_summary = " ".join(combined_sentences[:8])
        combined_sources = list(all_sources)
    
        # Optionally average confidence across top intents
        avg_confidence = float(np.mean([i["confidence"] for i in top_intents])) if top_intents else 0.0
    
        return {
            "combined_summary": combined_summary,
            "combined_sources": combined_sources,
            "average_confidence": round(avg_confidence, 4),
            "included_intents": [i["intent"] for i in top_intents]
        }

    def expand_and_search(self, query: str, candidates: int = DEFAULT_MAX_CANDIDATES) -> List[str]:
        """Run decomposition and search to build candidate URL list (raw)."""
        # decompose query and run separate searches for some subqueries to get diverse sources
        subs = decompose_query(query)
        # for top N subqueries perform a search
        all_candidates = []
        per_sub = max(3, candidates // min(len(subs), 5))
        for sub in subs[:5]:
            cand = duckduckgo_html_search(sub, max_candidates=per_sub)
            all_candidates.extend(cand)
        # dedupe preserving order
        seen = set()
        cleaned = []
        for u in all_candidates:
            u = fix_malformed_path(u)
            # remove rut param early
            u2 = cleanup_rut_query_param(u)
            # Also remove fragment-only anomalies
            u2 = u2.split("#")[0]
            if not is_valid_url(u2):
                continue
            # strip trailing slashes normalization
            u2 = u2.rstrip("/")
            if u2 not in seen:
                seen.add(u2)
                cleaned.append(u2)
        return cleaned

    def fetch_valid_sources(self, urls: List[str], needed: int = 5) -> List[str]:
        """Filter URLs by HEAD/GET up to 'needed' good sources. Retries are implicit by caller providing more candidates."""
        good = []
        for u in urls:
            # first clean any rut params again
            u_clean = cleanup_rut_query_param(u).rstrip("/")
            if not is_valid_url(u_clean):
                continue
            # quick HEAD check (some servers forbid HEAD)
            if safe_head(u_clean):
                good.append(u_clean)
            else:
                # try simple GET fallback once
                text = fetch_page_text(u_clean, timeout=8)
                if text:
                    good.append(u_clean)
            if len(good) >= needed:
                break
        return good

    def fetch_and_chunk_sources(self, urls: List[str], chunk_size_words: int = CHUNK_WORDS) -> Tuple[List[str], List[str]]:
        """Fetch text from each source and return (chunks, source_map) aligned lists."""
        chunks = []
        source_map = []
        for url in urls:
            txt = fetch_page_text(url)
            if not txt:
                continue
            cks = chunk_text_by_words(txt, max_words=chunk_size_words)
            if not cks:
                continue
            chunks.extend(cks)
            source_map.extend([url] * len(cks))
        return chunks, source_map

    def classify_and_summarize(self, chunks: List[str], source_map: List[str]) -> List[Dict]:
        """
        For each chunk:
         - get top intents + scores
         - if top score above threshold, summarize chunk
         - record results
        Returns list of dicts: {intent, confidence, summary, source}
        """
        results = []
        if not chunks:
            return results
        preds = self.intent_predict(chunks)  # returns list: for each chunk -> list of (label,score) sorted
        for i, chunk in enumerate(chunks):
            scores = preds[i]
            for label, score in scores:
                if score >= INTENT_CONFIDENCE_THRESHOLD:
                    # Summarize chunk (handle long chunk by capping)
                    try:
                        sumres = self.summarizer(chunk, max_length=120, min_length=25, do_sample=False)
                        summary_text = sumres[0]["summary_text"].strip()
                    except Exception as e:
                        # fallback: take first 2 sentences
                        summary_text = " ".join(chunk.split(".")[:2]).strip()
                    results.append({
                        "intent": label,
                        "confidence": float(score),
                        "summary": summary_text,
                        "source": source_map[i]
                    })
        return results

    def merge_by_intent(self, records: List[Dict]) -> List[Dict]:
        """Merge summaries and sources by intent and compute aggregate confidence."""
        merged = {}
        for r in records:
            intent = r["intent"]
            if intent not in merged:
                merged[intent] = {
                    "intent": intent,
                    "confidence_scores": [r["confidence"]],
                    "summaries": [r["summary"]],
                    "sources": [r["source"]]
                }
            else:
                merged[intent]["confidence_scores"].append(r["confidence"])
                merged[intent]["summaries"].append(r["summary"])
                if r["source"] not in merged[intent]["sources"]:
                    merged[intent]["sources"].append(r["source"])
        final = []
        for intent, v in merged.items():
            # fused summary: join unique sentences, preserve readability
            # simple fusion strategy: dedupe sentences
            sentences = []
            for s in v["summaries"]:
                for seg in re.split(r"(?<=[.!?])\s+", s):
                    seg = seg.strip()
                    if seg and seg not in sentences:
                        sentences.append(seg)
            fused = " ".join(sentences[:6])  # limit length
            avg_conf = float(np.mean(v["confidence_scores"])) if v["confidence_scores"] else 0.0
            final.append({
                "intent": intent,
                "confidence": round(avg_conf, 4),
                "summary": fused,
                "sources": v["sources"]
            })
        # sort by confidence desc
        final.sort(key=lambda x: x["confidence"], reverse=True)
        return final

    def detect_gaps(self, merged_intents: List[Dict], facets: List[str] = FACETS) -> Dict[str, bool]:
        """Return dict facet -> covered (True/False) by looking for facet keywords in summaries."""
        coverage = {}
        bigtext = " ".join([m["summary"] for m in merged_intents]).lower()
        for f in facets:
            coverage[f] = bool(re.search(r"\b" + re.escape(f.lower()) + r"\b", bigtext))
        return coverage

    def run(self, user_query: str, needed_sources=5, max_candidates=DEFAULT_MAX_CANDIDATES) -> Dict:
        # 1) Decompose
        subqueries = decompose_query(user_query)

        # 2) Search to build candidates
        candidates = self.expand_and_search(user_query, candidates=max_candidates)
        if not candidates:
            # fallback single search
            candidates = duckduckgo_html_search(user_query, max_candidates=max_candidates)

        # 3) Filter for reachable sources (grow candidate pool if needed)
        good_sources = self.fetch_valid_sources(candidates, needed=needed_sources)
        if len(good_sources) < needed_sources:
            # try fetching more candidates from search results beyond first block
            more_candidates = duckduckgo_html_search(user_query, max_candidates=max_candidates*2)
            combined = candidates + more_candidates
            # dedupe:
            seen = set(candidates)
            for u in combined:
                if u not in seen:
                    candidates.append(u); seen.add(u)
            good_sources = self.fetch_valid_sources(candidates, needed=needed_sources)

        # 4) Fetch and chunk
        chunks, source_map = self.fetch_and_chunk_sources(good_sources, chunk_size_words=CHUNK_WORDS)

        # If still too few chunks, try lowering chunk size or adding more candidates (agentic retry)
        if len(chunks) < MIN_CONTENT_CHUNKS and len(candidates) > len(good_sources):
            print("Not enough content chunks; retrying with wider candidate set...")
            more_good = self.fetch_valid_sources(candidates, needed=needed_sources*2)
            new_sources = [u for u in more_good if u not in good_sources]
            if new_sources:
                more_chunks, more_map = self.fetch_and_chunk_sources(new_sources)
                chunks.extend(more_chunks); source_map.extend(more_map)

        # 5) Classify & summarize
        records = self.classify_and_summarize(chunks, source_map)

        # 6) Merge records by intent
        merged = self.merge_by_intent(records)
        combined_result = self.combine_top_results(merged, top_n=3)


        # 7) Gap detection (coverage)
        coverage = self.detect_gaps(merged)

        # 8) Build final agentic plan: simple suggestions
        action_plan = []
        for m in merged[:5]:  # top 5 intents
            action_plan.append({
                "intent": m["intent"],
                "recommended_action": "summarize_and_compare",
                "notes": f"High confidence: {m['confidence']:.2f}" if m["confidence"]>0.6 else f"Medium confidence: {m['confidence']:.2f}"
            })

        result = {
            "query": user_query,
            "decomposed_queries": subqueries,
            "sources_tried": candidates,
            "sources_used": good_sources,
            "merged_intents": merged,
            "combined_result": combined_result,
            "coverage": coverage,
            "action_plan": action_plan,
            "diagnostics": {
                "num_chunks": len(chunks),
                "num_records": len(records)
            }
        }
        return result




# ---------- Main ----------
def pretty_print_result(res: Dict):
    combined_result = res.get("combined_result")
    print("\n=== Akia Stage 2 Agentic Result ===\n")
    print("Query:", res["query"])
    print("\nDecomposed sub-queries (sample):")
    for q in res["decomposed_queries"][:8]:
        print(" -", q)
    print("\nSources used:")
    for s in res["sources_used"]:
        print(" -", s)
    print("\n\nMerged Intents and Summaries:")
    for i, m in enumerate(res["merged_intents"], 1):
        print(f"\n[{i}] Intent: {m['intent']}  (confidence {m['confidence']})")
        print("Summary:", m['summary'])
        print("Sources:")
        for src in m.get("sources", []):
            print("  -", src)
    print("\nCoverage / Gaps:")
    for k, v in res["coverage"].items():
        print(f" - {k}: {'covered' if v else 'missing'}")
    print("\nAction plan (top):")
    for a in res["action_plan"]:
        print(" -", a)
    print("\nDiagnostics:", res["diagnostics"])
    print("\nCombined Top Results Summary:")
    print(combined_result["combined_summary"])
    print("\nCombined Sources:")
    for src in combined_result["combined_sources"]:
        print(" -", src)
    print(f"\nAverage Confidence: {combined_result['average_confidence']}")
    print(f"Included Intents: {combined_result['included_intents']}")

    print("\n=== End ===\n")

if __name__ == "__main__":
    # quick interactive example
    agent = Stage2Agent(INTENT_MODEL_DIR, SUMMARIZER_MODEL)
    test_query = "Compare Kenya to South Korea in terms of GDP"
    out = agent.run(test_query, needed_sources=5, max_candidates=30)
    pretty_print_result(out)
    # Save JSON
    with open("stage2_last_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
