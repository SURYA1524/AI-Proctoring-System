import time

class EventBuffer:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append({
            "event": event,
            "time": time.strftime("%H:%M:%S")
        })