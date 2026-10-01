"""
ISDO Lab C4 - Resolution / KB Agent
Searches ChromaDB for matching KB articles and drafts a resolution.
HIGH confidence on a non-P1 ticket -> auto-resolve. Otherwise -> HITL flag.

Run from the project root:   python agents/resolution_agent.py
Needs:  pip install anthropic python-dotenv chromadb
        .env in the project root with ANTHROPIC_API_KEY=...
        KB articles in data/kb/ (the 5 .md files from Lab C1)
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

sys.stdout.reconfigure(errors="replace")  # KB articles contain arrows etc.; never crash on print

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "kb"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"   # same folder Lab C1 wrote to
COLLECTION = "isdo_kb"

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ERROR: ANTHROPIC_API_KEY not found. Add it to the .env file in your project root.")
client = anthropic.Anthropic()

MODEL = "claude-opus-5"
MAX_TOKENS = 2000      # thinking is on by default and counts toward this limit
MAX_LOOP_STEPS = 5     # safety stop - a stuck loop can never keep calling the API

HIGH_THRESHOLD = 0.60    # > 60% similarity
MEDIUM_THRESHOLD = 0.35  # 35-60% similarity; below this is LOW
NO_AUTO_RESOLVE_PRIORITIES = {"P1"}   # guardrail: critical incidents always go to a human

# -- LOAD CHROMADB KB (built in Lab C1) ---------------------------------------

def chunk_article(text, title_fallback):
    """Split one markdown article at '## ' headings; each chunk keeps the article title."""
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), title_fallback)
    chunks, section, buffer = [], "Overview", []
    for line in lines + ["## __end__"]:
        if line.startswith("## "):
            body = "\n".join(buffer).strip()
            if body:
                chunks.append({"section": section, "text": f"{title}\n{section}\n{body}"})
            section, buffer = line[3:].strip(), []
        else:
            buffer.append(line)
    return chunks


def build_kb():
    """Rebuild the isdo_kb collection from data/kb/ using cosine distance."""
    db = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    kb = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md in sorted(KB_DIR.glob("*.md")):
        for i, chunk in enumerate(chunk_article(md.read_text(encoding="utf-8"), md.stem)):
            ids.append(f"{md.stem}::{i}")
            docs.append(chunk["text"])
            metas.append({"article": md.name, "section": chunk["section"]})
    if not docs:
        sys.exit(f"ERROR: no .md files in {KB_DIR}. Copy the KB articles there first.")
    kb.add(ids=ids, documents=docs, metadatas=metas)
    print(f"KB rebuilt: {len(docs)} chunks from {len(set(m['article'] for m in metas))} articles")
    return kb


def load_kb():
    """Reuse the Lab C1 collection if it's usable; otherwise rebuild it.

    1 - distance is only a sensible 0-1 confidence score with cosine distance,
    so a collection built with the default (L2) distance is rebuilt.
    """
    db = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        kb = db.get_collection(COLLECTION)
        if kb.count() > 0 and (kb.metadata or {}).get("hnsw:space") == "cosine":
            print(f"KB loaded from Lab C1: {kb.count()} chunks")
            return kb
    except Exception:
        pass
    return build_kb()


KB = load_kb()

# -- TOOL DEFINITIONS ---------------------------------------------------------

tools = [
    {
        "name": "search_kb",
        "description": "Search the knowledge base for articles matching the ticket. "
                       "Returns the top 2 articles with confidence scores and full article text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The ticket's short description and details, used as-is"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "draft_resolution",
        "description": "Draft the resolution for the requester, based on the KB article found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "resolution_text": {"type": "string",
                                    "description": "3-4 numbered steps taken from the KB article"},
                "auto_resolve": {"type": "boolean",
                                 "description": "True only if HIGH confidence and the KB article says L1 auto-resolvable"},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"],
                               "description": "Use the confidence_level returned by search_kb"},
                "kb_article_used": {"type": "string"},
            },
            "required": ["ticket_number", "resolution_text", "auto_resolve", "confidence", "kb_article_used"],
        },
    },
]

# -- TOOL IMPLEMENTATION ------------------------------------------------------

def confidence_level(score):
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def search_kb(query):
    """Return the top 2 distinct articles, scored by their best-matching chunk."""
    raw = KB.query(query_texts=[query], n_results=min(10, KB.count()))
    best = {}  # article -> smallest distance
    for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
        name = meta.get("article") or meta.get("filename", "unknown")
        best[name] = min(dist, best.get(name, 99))
    top = sorted(best.items(), key=lambda kv: kv[1])[:2]

    articles = []
    for name, dist in top:
        score = round(max(0.0, 1.0 - dist), 2)
        path = KB_DIR / name
        articles.append({
            "article": name,
            "confidence_score": score,
            # Whole article (~2 KB) so the model sees the actual Resolution Steps section
            "content": path.read_text(encoding="utf-8") if path.exists() else "",
        })
    top_score = articles[0]["confidence_score"] if articles else 0.0
    return {"query": query, "articles": articles,
            "top_score": top_score, "confidence_level": confidence_level(top_score)}


def apply_guardrails(draft, search_result, priority):
    """Code-level HITL rules - these hold even if the model gets it wrong.
    They can only turn auto_resolve OFF, never on."""
    draft = dict(draft)
    notes = []
    if search_result:
        level = search_result["confidence_level"]
        if draft.get("confidence") != level:
            notes.append(f"confidence corrected {draft.get('confidence')} -> {level} (score {search_result['top_score']:.0%})")
            draft["confidence"] = level
    else:
        draft["confidence"] = "LOW"
        notes.append("no KB search was run - forced LOW")
    if draft.get("auto_resolve") and draft["confidence"] != "HIGH":
        draft["auto_resolve"] = False
        notes.append("auto_resolve blocked: confidence is not HIGH")
    if draft.get("auto_resolve") and priority in NO_AUTO_RESOLVE_PRIORITIES:
        draft["auto_resolve"] = False
        notes.append(f"auto_resolve blocked: {priority} tickets always need a human")
    return draft, notes

# -- RESOLUTION AGENT ---------------------------------------------------------

SYSTEM_PROMPT = """You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Call search_kb ONCE, using the ticket's summary and details as the query. Do not retry
   with different wording - a LOW-confidence result is a valid, expected outcome.
2. Call draft_resolution:
   - confidence: use the confidence_level returned by search_kb, unchanged.
   - resolution_text: 3-4 numbered steps copied from the matched article's Resolution Steps.
     Be specific (exact menu paths, commands, URLs) - no generic advice.
   - auto_resolve: true ONLY if confidence is HIGH, the ticket is not P1, and the article's
     "Auto-Resolve Eligibility" section says the issue is L1 auto-resolvable. Otherwise false.
   - If confidence is LOW, say the ticket should be escalated to L2 and name the closest article.
3. After the draft_resolution result comes back, reply with one short confirmation line."""


def resolve_ticket(ticket_number, short_description, description, category, priority):
    print(f"\n{'=' * 55}")
    print(f"Resolving: {ticket_number} | Category: {category} | Priority: {priority}")
    print(f"{'=' * 55}")
    print(f"Issue: {short_description}")

    messages = [{
        "role": "user",
        "content": f"Find a resolution for this ticket:\n\nTicket: {ticket_number}\n"
                   f"Category: {category}\nPriority: {priority}\n"
                   f"Summary: {short_description}\nDetails: {description}",
    }]
    last_search, final = None, None

    steps = 0
    while steps < MAX_LOOP_STEPS:
        steps += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "search_kb":
                    result = search_kb(block.input["query"])
                    last_search = result
                    print(f"  -> KB search: '{block.input['query'][:70]}'")
                    for art in result["articles"]:
                        print(f"       [{art['confidence_score']:.0%}] {art['article']}")
                elif block.name == "draft_resolution":
                    result, notes = apply_guardrails(block.input, last_search, priority)
                    final = result
                    print(f"\n  -> Confidence: {result['confidence']}  |  Auto-resolve: {result['auto_resolve']}")
                    print(f"  -> KB Article: {result.get('kb_article_used')}")
                    for note in notes:
                        print(f"  -> Guardrail: {note}")
                    print("\n  RESOLUTION DRAFT:")
                    for line in str(result.get("resolution_text", "")).splitlines():
                        print(f"    {line}")
                    if not result["auto_resolve"]:
                        print(f"\n  WARNING  HITL FLAG: {result['confidence']} confidence / {priority} "
                              f"-- human review required before sending.")
                else:
                    result = {"error": f"Unknown tool: {block.name}"}
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": json.dumps(result)})
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n  Agent: {block.text.strip()}")
            break

        print(f"  !! Stopped early: stop_reason = {response.stop_reason}")
        break

    if final is None:
        print("\n  WARNING  HITL FLAG: no resolution drafted -- route to a human.")
    return final

# -- RUN ON SAMPLE TICKETS ----------------------------------------------------

if __name__ == "__main__":
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
         "Network", "P2"),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset.",
         "Access", "P2"),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple Finance users unable to login to SAP. Error code: DBCON_FAIL.",
         "Application", "P1"),
        # Step 5 - a ticket with no matching KB article
        ("TEST-0005", "Cisco Webex not launching on Mac M2",
         "Cisco Webex app bounces in the dock and closes immediately on a MacBook with M2 chip.",
         "Software", "P3"),
    ]

    results = [(t[0], t[4], resolve_ticket(*t)) for t in test_tickets]

    print(f"\n{'=' * 55}")
    print("RESOLUTION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  {'Ticket':<12}{'Priority':<10}{'Confidence':<12}{'Auto-resolve':<14}KB article")
    for number, prio, r in results:
        if r:
            print(f"  {number:<12}{prio:<10}{r['confidence']:<12}{str(r['auto_resolve']):<14}{r.get('kb_article_used')}")
        else:
            print(f"  {number:<12}{prio:<10}{'-':<12}{'False':<14}(none)")
