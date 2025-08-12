import json, threading
from pathlib import Path
from flask import Flask, jsonify, render_template_string
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


app = Flask(__name__)
EVENTS = []
LOG_FILE = Path("logs/processing_events.jsonl")
LOG_DIR = Path("logs")

# Load last 10 events at startup
def load_initial_events():
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()[-10:]
        EVENTS[:] = [json.loads(l) for l in lines]

class _Handler(FileSystemEventHandler):
    def on_modified(self, e):
        if e.src_path.endswith("processing_events.jsonl"):
            lines = LOG_FILE.read_text().splitlines()[-10:]
            EVENTS[:] = [json.loads(l) for l in lines]

def _start_watcher():
    o = Observer()
    o.schedule(_Handler(), str(LOG_DIR), recursive=False)
    o.start()

HTML = """
<!doctype html><title>Debug</title>
<h2>Last 10 events</h2><ul>
{% for ev in events %}
    <li>{{ev.timestamp}} – {{ev.event}}</li>
{% endfor %}
</ul>
"""

@app.route("/")
def index():
    return render_template_string(HTML, events=EVENTS)

@app.route("/api/events")
def api():
    return jsonify(EVENTS)

if __name__ == "__main__":
    load_initial_events()
    threading.Thread(target=_start_watcher, daemon=True).start()
    app.run(port=5000)
