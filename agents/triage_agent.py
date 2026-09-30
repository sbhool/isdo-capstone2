"""
ISDO Lab C3 - Triage Agent
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling and an agentic (ReAct) loop.

Run from the project root:   python agents/triage_agent.py
Needs:  pip install anthropic python-dotenv
        a .env file in the project root containing  ANTHROPIC_API_KEY=sk-ant-...
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_CSV = PROJECT_ROOT / "data" / "incidents.csv"

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ERROR: ANTHROPIC_API_KEY not found. Add it to the .env file in your project root.")

client = anthropic.Anthropic()

MODEL = "claude-opus-5"   # model named in the lab
MAX_TOKENS = 2000         # thinking is on by default and counts toward this limit
MAX_LOOP_STEPS = 5        # safety stop so a stuck loop can never keep calling the API

# -- TOOL DEFINITIONS ---------------------------------------------------------

tools = [
    {
        "name": "classify_ticket",
        "description": "Classify an IT support ticket. Returns category, priority, "
                       "assignment_group, whether PII was detected, and one-sentence reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"],
                    "description": "The ticket category",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=Critical/many users, P2=High/some users, P3=Medium/single user, P4=Low/request",
                },
                "assignment_group": {
                    "type": "string",
                    "description": "Team to assign to, e.g. Network-Ops, App-Support, Desktop-Support, "
                                   "Service-Desk, Security-Ops, Server-Ops, Email-Support, DBA-Team",
                },
                "pii_detected": {
                    "type": "boolean",
                    "description": "True if the ticket contains names, email addresses, employee IDs, or IP addresses",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision",
                },
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
        },
    },
    {
        "name": "get_open_tickets",
        "description": "Get a count of currently open tickets by category from the incidents CSV.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# -- TOOL IMPLEMENTATION ------------------------------------------------------

def get_open_tickets():
    """Read incidents.csv and return the number of Open tickets per category."""
    counts = {}
    try:
        with open(INCIDENTS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if None in row:  # malformed row (unquoted comma) - skip rather than miscount
                    print(f"  Warning: skipping malformed CSV row {row.get('number')}")
                    continue
                if row.get("state", "").strip() == "Open":
                    cat = row.get("category", "Unknown").strip()
                    counts[cat] = counts.get(cat, 0) + 1
    except FileNotFoundError:
        return {"error": f"File not found: {INCIDENTS_CSV}"}
    return counts


def handle_tool_call(tool_name, tool_input):
    """Route a tool call from the model to its local implementation."""
    if tool_name == "get_open_tickets":
        return get_open_tickets()
    if tool_name == "classify_ticket":
        return tool_input  # the structured classification IS the tool's output
    return {"error": f"Unknown tool: {tool_name}"}

# -- TRIAGE AGENT -------------------------------------------------------------

SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

For every ticket, call the classify_ticket tool exactly once. Do not answer in plain text instead.
After the tool result comes back, reply with one short confirmation line.

Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)

PII: set pii_detected=true if the ticket contains a person's name, an email address,
an employee ID, or an IP address. Text shown as [REDACTED] is already masked and is not PII.

Be consistent: the same ticket must always get the same classification."""


def triage_ticket(ticket_number, short_description, description):
    """Run the agentic loop on one ticket. Returns the classification dict (or None)."""
    print(f"\n{'=' * 55}")
    print(f"Triaging: {ticket_number}")
    print(f"{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [{
        "role": "user",
        "content": f"Please triage this ticket:\n\nTicket: {ticket_number}\n"
                   f"Summary: {short_description}\nDetails: {description}",
    }]
    classification = None

    # Agentic while loop: Reason -> Act (tool call) -> Observe (tool result) -> Reason ...
    steps = 0
    while steps < MAX_LOOP_STEPS:
        steps += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "low"},  # low thinking effort: fast, cheap, consistent
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
                print(f"  -> Tool called: {block.name}")
                result = handle_tool_call(block.name, block.input)
                if block.name == "classify_ticket":
                    classification = result
                    print(f"  -> Category:    {result.get('category')}")
                    print(f"  -> Priority:    {result.get('priority')}")
                    print(f"  -> Assign To:   {result.get('assignment_group')}")
                    print(f"  -> PII Found:   {result.get('pii_detected')}")
                    print(f"  -> Reason:      {result.get('reasoning')}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            break

        # Anything else (e.g. max_tokens) - stop instead of looping forever
        print(f"  !! Stopped early: stop_reason = {response.stop_reason}")
        break

    if classification is None:
        print("  !! No classification produced for this ticket")
    return classification

# -- RUN ON SAMPLE TICKETS ----------------------------------------------------

if __name__ == "__main__":
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. Error: authentication failed."),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today."),
        ("INC0001008", "Network switch down - Building C",
         "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
        ("REQ-1002", "VPN access for new contractor joining project Phoenix",
         "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. Email: contractor@client.com"),
        # Step 5 - your own ticket
        ("TEST-0006", "Salesforce CRM access issue",
         "User cannot access Salesforce CRM from company laptop since this morning."),
        # Step 5 (part 2) - uncomment to see if priority changes when many users are affected:
        # ("TEST-0007", "Salesforce CRM access issue",
        #  "Entire Sales team (25 users) cannot access Salesforce CRM since this morning."),
    ]

    results = []
    for number, short_desc, desc in test_tickets:
        results.append((number, triage_ticket(number, short_desc, desc)))

    print(f"\n{'=' * 55}")
    print("TRIAGE SUMMARY")
    print(f"{'=' * 55}")
    print(f"  {'Ticket':<12}{'Category':<13}{'Priority':<10}{'Assign To':<17}PII")
    for number, c in results:
        if c:
            print(f"  {number:<12}{c['category']:<13}{c['priority']:<10}{c['assignment_group']:<17}{c['pii_detected']}")
        else:
            print(f"  {number:<12}(no classification)")

    print(f"\n{'=' * 55}")
    print("OPEN TICKET COUNTS BY CATEGORY")
    print(f"{'=' * 55}")
    counts = get_open_tickets()  # demo the get_open_tickets tool directly
    if "error" in counts:
        print(f"  {counts['error']}")
    else:
        for cat, count in sorted(counts.items()):
            print(f"  {cat:<20} {count} open")