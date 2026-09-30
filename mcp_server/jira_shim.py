"""
ISDO Lab C2 - Mock Jira Service Management REST API (Flask shim) - port 5002

  GET  /rest/agile/1.0/board/requests   list all (?request_type= ?priority= ?assignee= ?status=)
  GET  /rest/api/2/issue/<key>          get one request (Jira-style nested 'fields')
  PUT  /rest/api/2/issue/<key>          update a request in memory
  POST /rest/api/2/issue                create a request
  GET  /health                          health check

Run from the project root:  python mcp_server/jira_shim.py
"""
import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "requests.csv")
FILTERS = ["request_type", "priority", "assignee", "status"]

def load_requests():
    """Load requests.csv into a dict keyed by request key."""
    data = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if None in row:  # row had more columns than the header (unquoted comma)
                    print(f"Warning: malformed row {row.get('key')} - check for unquoted commas")
                    row.pop(None)
                data[row["key"]] = {k: (v or "").strip() for k, v in row.items()}
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return data

REQUESTS = load_requests()  # in-memory store for this session

def to_jira(key, r):
    """Wrap a flat CSV row in Jira's nested 'fields' structure."""
    return {"key": key, "fields": {
        "summary": r.get("summary"),
        "issuetype": {"name": r.get("request_type")},
        "priority": {"name": r.get("priority")},
        "status": {"name": r.get("status")},
        "assignee": {"displayName": r.get("assignee")},
        "customfield_sla": r.get("sla"),
    }}

@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    results = list(REQUESTS.values())
    for key in FILTERS:
        val = request.args.get(key)  # Flask already decodes '+' to a space
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"issues": results, "total": len(results)})

@app.get("/rest/api/2/issue/<key>")
def get_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(key, REQUESTS[key]))

@app.put("/rest/api/2/issue/<key>")
def update_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"errorMessages": ["No JSON update body provided"]}), 400
    fields = data.get("fields", data)  # accept Jira-style {"fields": {...}} or flat
    REQUESTS[key].update(fields)
    print(f"[Jira Mock] Updated {key}: {fields}")
    return jsonify({"key": key, "message": "Updated successfully"})

@app.post("/rest/api/2/issue")
def create_request():
    fields = (request.get_json(silent=True) or {}).get("fields", {})
    if not fields.get("summary"):
        return jsonify({"errorMessages": ["fields.summary is required"]}), 400
    key = f"REQ-{1001 + len(REQUESTS)}"
    REQUESTS[key] = {"key": key, "summary": fields["summary"],
                     "request_type": fields.get("issuetype", {}).get("name", ""),
                     "priority": fields.get("priority", {}).get("name", "Medium"),
                     "assignee": "", "sla": "", "status": "Open"}
    print(f"[Jira Mock] Created request: {key}")
    return jsonify({"key": key, "message": "Request created"}), 201

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})

if __name__ == "__main__":
    print("Jira Mock API starting on http://localhost:5002")
    print(f"Loaded {len(REQUESTS)} requests from data/requests.csv")
    app.run(port=5002, debug=True, use_reloader=False)