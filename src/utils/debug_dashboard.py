import json, threading
from pathlib import Path
from flask import Flask, jsonify, render_template_string
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)
EVENTS = []

LOG_DIR = Path("logs")
class _Handler(FileSystemEventHandler):
    def on_modified(self, e):
        if e.src_path.endswith(".jsonl"):
            lines = LOG_DIR.read_text().splitlines()[-10:]
            EVENTS[:] = [json.loads(l) for l in lines]

def _start_watcher():
    o = Observer()
    o.schedule(_Handler(), str(LOG_DIR), recursive=False)
    o.start()

HTML = """
<!doctype html><title>Debug</title>
<h2>Last 10 events</h2><ul>
{% for ev in events %}
  <li>{{ev.timestamp}} – {{ev.module}}: {{'%.2f'|format(ev.latency_ms)}} ms</li>
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
    threading.Thread(target=_start_watcher, daemon=True).start()
    app.run(port=5000)

@app.route("/")
def index():
    return render_template_string(_HTML, events=_EVENTS)

@app.route("/api/events")
def api_events():
    return jsonify(_EVENTS)

if __name__ == "__main__":
    threading.Thread(target=start_watcher, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
