"""
ISDO Lab C2 - Mock ServiceNow Table API (Flask shim) - port 5001

  GET   /api/now/table/incident            list all (?category= ?priority= ?state= ?assignment_group=)
  GET   /api/now/table/incident/<number>   get one incident
  PATCH /api/now/table/incident/<number>   update fields in memory (e.g. {"state": "Escalated"})
  POST  /api/now/table/incident            create an incident
  GET   /health                            health check

Run from the project root:  python mcp_server/snow_shim.py
"""
import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "incidents.csv")
FILTERS = ["category", "priority", "state", "assignment_group"]


def load_incidents():
    """Load incidents.csv into a dict keyed by incident number."""
    incidents = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if None in row:  # row had more columns than the header (unquoted comma)
                    print(f"Warning: malformed row {row.get('number')} - check for unquoted commas")
                    row.pop(None)
                incidents[row["number"]] = {k: (v or "").strip() for k, v in row.items()}
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return incidents


INCIDENTS = load_incidents()  # in-memory store for this session


@app.get("/api/now/table/incident")
def list_incidents():
    results = list(INCIDENTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"result": results, "total": len(results)})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not updates:
        return jsonify({"error": "No JSON update body provided"}), 400
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.post("/api/now/table/incident")
def create_incident():
    data = request.get_json(silent=True)
    if not data or "number" not in data:
        return jsonify({"error": "Missing required field: number"}), 400
    INCIDENTS[data["number"]] = data
    print(f"[ServiceNow Mock] Created incident: {data['number']}")
    return jsonify({"result": data, "message": "Incident created"}), 201


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print("ServiceNow Mock API starting on http://localhost:5001")
    print(f"Loaded {len(INCIDENTS)} incidents from data/incidents.csv")
    # use_reloader=False: avoids loading twice and losing in-memory PATCH updates on restart
    app.run(port=5001, debug=True, use_reloader=False)