from fastapi import FastAPI

app = FastAPI()

latest_data = {
    "score": 0,
    "emotion": "Initializing",
    "cognitive_load": 0
}

@app.get("/status")
def get_status():
    return latest_data

def update_status(data):
    global latest_data
    latest_data = data