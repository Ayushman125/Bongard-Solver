import threading
import time
import requests
from src.utils.debug_dashboard import app, _start_watcher

def test_dashboard_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"logs").mkdir()
    # start server
    threading.Thread(target=lambda: app.run(port=5001), daemon=True).start()
    time.sleep(1)
    resp = requests.get("http://127.0.0.1:5001/api/events")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
