#!/usr/bin/env python3
"""
stage2_client.py

Silent runner that uses Stage2Agent from stage2_agentic.py.
- Tries Stage2Agent aggressively with multiple attempts (increasing candidates/sources)
- If Stage2Agent cannot produce useful output, fallback to duckduckgo-search and re-run minimal pipeline
- Suppresses internal logs and errors from being printed; stores them in stage2_debug.log
- Prints only a single final merged summary and its sources (or a brief "no info" message)

Usage:
    python stage2_client.py "Compare Tesla and BYD electric cars"
"""

import sys
import json
import io
import time
from contextlib import redirect_stdout, redirect_stderr

# Import the Stage2Agent and helpers from your existing file
import stage2_agentic as s2mod
from stage2_agentic import Stage2Agent

# Optional fallback search package
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except Exception:
    HAS_DDGS = False

DEBUG_LOG = "stage2_debug.log"

def _capture_log(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) while capturing stdout/stderr into the debug log.
    Returns (result, captured_text)
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            res = fn(*args, **kwargs)
    except Exception as e:
        # ensure exception details captured to buffer
        buf.write(f"\nEXCEPTION: {e}\n")
        res = None
    captured = buf.getvalue()
    # append to persistent debug log
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} call: {fn.__name__} ---\n")
        f.write(captured)
        f.write("\n--- end ---\n")
    return res, captured

def is_useful_result(res: dict) -> bool:
    """Assess whether result contains useful info to show the user."""
    if not isinstance(res, dict):
        return False
    # Check merged intents, combined_summary, or sources_used
    merged = res.get("merged_intents") or []
    combined = res.get("combined_result", {}).get("combined_summary", "")
    sources = res.get("sources_used") or res.get("sources", []) or []
    diagnostics = res.get("diagnostics", {})
    num_chunks = diagnostics.get("num_chunks", 0)
    # Useful if we have merged intents OR combined summary text OR at least 1 successfully fetched source with chunks
    if merged and len(merged) > 0:
        return True
    if combined and combined.strip():
        return True
    if sources and num_chunks > 0:
        return True
    return False

def try_stage2_agent(agent: Stage2Agent, query: str):
    """
    Try multiple Stage2Agent runs with escalating parameters.
    Returns the best result (first useful) or last result if none useful.
    All internal prints are captured into the debug log.
    """
    attempts = [
        (5, 30),
        (8, 60),
        (12, 100)
    ]
    last_res = None
    for needed, candidates in attempts:
        res, _ = _capture_log(agent.run, query, needed, candidates)
        last_res = res or last_res
        if res and is_useful_result(res):
            return res
        # small backoff
        time.sleep(0.2)
    return last_res

def fallback_via_ddgs(agent: Stage2Agent, query: str, needed_sources=5, max_results=40):
    """
    If Stage2Agent fails, use duckduckgo-search (if available) to get candidate URLs,
    then use agent.fetch_valid_sources / fetch_and_chunk_sources / classify_and_summarize / merge_by_intent
    to create a final result.
    This function captures all internal output to the debug log.
    """
    if not HAS_DDGS:
        return None

    # 1) Gather raw results from DDGS
    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        # capture to debug log via _capture_log wrapper
        def _fail():
            raise e
        _capture_log(_fail)
        return None

    # Extract and clean URLs, keep titles/snippets for potential quick summarize
    raw_urls = []
    snippets = []
    for r in ddgs_results:
        href = r.get("href") or r.get("url") or r.get("link")
        title = r.get("title") or ""
        body = r.get("body") or r.get("snippet") or ""
        if href:
            # apply the same cleaning used in your stage2 module
            href2 = s2mod.fix_malformed_path(href)
            href2 = s2mod.cleanup_rut_query_param(href2)
            href2 = href2.split("#")[0].rstrip("/")
            if s2mod.is_valid_url(href2) and href2 not in raw_urls:
                raw_urls.append(href2)
        if body:
            snippets.append(title + ". " + body if title else body)

    if not raw_urls:
        # No URLs found via DDGS
        return None

    # 2) Filter reachable sources using agent.fetch_valid_sources (capture logs)
    good_sources, _ = _capture_log(agent.fetch_valid_sources, raw_urls, needed)
    if not good_sources:
        # Try with fewer required sources
        good_sources, _ = _capture_log(agent.fetch_valid_sources, raw_urls, max(1, needed//2))
    if not good_sources:
        # fallback to snippet summarization: summarize titles/snippets
        if snippets:
            combined_text = " ".join(snippets[:20])
            # capture summarizer call
            sumres, _ = _capture_log(agent.summarizer, combined_text, max_length=120, min_length=30, do_sample=False)
            try:
                summary_text = sumres[0]["summary_text"].strip() if sumres else ""
            except Exception:
                summary_text = combined_text[:400]
            fake_result = {
                "query": query,
                "decomposed_queries": [query],
                "sources_tried": raw_urls[:10],
                "sources_used": [],
                "merged_intents": [],
                "combined_result": {
                    "combined_summary": summary_text,
                    "combined_sources": [],
                    "average_confidence": 0.0,
                    "included_intents": []
                },
                "coverage": {},
                "action_plan": [],
                "diagnostics": {"num_chunks": 0, "num_records": 0}
            }
            return fake_result
        return None

    # 3) Fetch and chunk from good_sources
    chunks, source_map = _capture_log(agent.fetch_and_chunk_sources, good_sources)
    if not chunks:
        return None

    # 4) Classify & summarize
    records, _ = _capture_log(agent.classify_and_summarize, chunks, source_map)
    if not records:
        return None

    # 5) Merge
    merged, _ = _capture_log(agent.merge_by_intent, records)
    combined_result = _capture_log(agent.combine_top_results, merged)[0] or {}

    res = {
        "query": query,
        "decomposed_queries": [query],
        "sources_tried": raw_urls,
        "sources_used": good_sources,
        "merged_intents": merged,
        "combined_result": combined_result,
        "coverage": agent.detect_gaps(merged),
        "action_plan": [],
        "diagnostics": {"num_chunks": len(chunks) if chunks else 0, "num_records": len(records) if records else 0}
    }
    return res

def final_output_and_save(res: dict):
    """
    Print exactly one final result to user (combined summary + sources),
    and save full result JSON to stage2_last_result.json. No errors printed.
    """
    if not res:
        print("No reliable information found for that query. Try rephrasing.")
        # still write an empty result file
        with open("stage2_last_result.json", "w", encoding="utf-8") as f:
            json.dump({"query": None, "note": "no result"}, f, ensure_ascii=False, indent=2)
        return

    comb = res.get("combined_result", {})
    summary = comb.get("combined_summary", "").strip()
    sources = comb.get("combined_sources", []) or res.get("sources_used", []) or []

    if summary:
        print(summary)
    elif res.get("merged_intents"):
        # fallback: print first merged intent summary
        first = res["merged_intents"][0]
        print(first.get("summary", "").strip() or "No concise summary available.")
    else:
        print("No reliable information found for that query. Try rephrasing.")

    if sources:
        print("\nSources:")
        for s in sources:
            print("-", s)

    # save full result for debugging or downstream use
    with open("stage2_last_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage2_client.py \"your query here\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:]).strip()

    # Instantiate Stage2Agent but capture its prints to debug log
    agent, _ = _capture_log(Stage2Agent, s2mod.INTENT_MODEL_DIR, s2mod.SUMMARIZER_MODEL)
    if not agent:
        # If Stage2Agent couldn't be created (model missing or load error), try fallback immediately
        fallback_res = fallback_via_ddgs(None, query) if HAS_DDGS else None
        final_output_and_save(fallback_res)
        return

    # 1) Try Stage2Agent with escalating attempts (silently)
    res = try_stage2_agent(agent, query)

    # 2) If not useful, try fallback via duckduckgo-search (silently)
    if not is_useful_result(res):
        fallback_res = fallback_via_ddgs(agent, query)
        if is_useful_result(fallback_res):
            res = fallback_res

    # 3) Print only final result (summary + sources) and save full json
    final_output_and_save(res)

if __name__ == "__main__":
    main()
