"""
processing_monitor.py
Dashboard and logging for Bongard Solver pipeline health.
"""
import logging

class ProcessingMonitor:
    def __init__(self):
        self.events = []
        self.log_file = 'logs/processing_events.jsonl'

    def log_event(self, event, info=None):
        import json, datetime, os
        self.events.append((event, info))
        logging.info(f"[ProcessingMonitor] {event}: {info}")
        # Write event to .jsonl file for dashboard
        event_dict = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event': event,
            'info': info
        }
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_dict) + '\n')

    def get_events(self):
        return self.events

processing_monitor = ProcessingMonitor()
