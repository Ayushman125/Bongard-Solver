"""
Debug Dashboard: real-time plotting of JSON event logs.
Version: 0.1.0
"""

__version__ = "0.1.0"

import json, time, threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
_EVENTS = []

_HTML = """
<!doctype html>
<title>Debug Dashboard</title>
<h1>Profiler Latencies (last 10 events)</h1>
<ul>
  {% for ev in events %}
    <li>{{ev.timestamp}} — {{ev.module}}: {{'%.2f'%ev.latency_ms}} ms</li>
  {% endfor %}
</ul>
"""

class LogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("profiler_events.jsonl"):
            with open(event.src_path) as f:
                lines = f.readlines()[-10:]
            _EVENTS.clear()
            for ln in lines:
                _EVENTS.append(json.loads(ln))

def start_watcher():
    observer = Observer()
    handler = LogHandler()
    observer.schedule(handler, path="logs/", recursive=False)
    observer.start()

@app.route("/")
def index():
    return render_template_string(_HTML, events=_EVENTS)

@app.route("/api/events")
def api_events():
    return jsonify(_EVENTS)

if __name__ == "__main__":
    threading.Thread(target=start_watcher, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
