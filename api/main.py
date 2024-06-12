import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import uuid
import time
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

# In-memory storage for experiments and their metrics
experiments: Dict[str, Dict] = {}
subscribers = []

class ExperimentCreate(BaseModel):
    name: str
    description: str = None

class MetricLog(BaseModel):
    key: str
    value: float

class Experiment(BaseModel):
    id: str
    name: str
    description: str
    metrics: Dict[str, List[float]] = {}

@app.post("/experiments/", response_model=Experiment)
def create_experiment(experiment: ExperimentCreate):
    experiment_id = str(uuid.uuid4())
    experiments[experiment_id] = {
        "id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "metrics": {
            "loss": [],
            "accuracy": [],
        }
    }
    return experiments[experiment_id]

@app.delete("/experiments/")
def clear_experiments():
    experiments.clear()
    return {"status": "success"}

@app.get("/experiments/", response_model=List[Experiment])
def list_experiments():
    return list(experiments.values())

@app.get("/experiments/{experiment_id}", response_model=Experiment)
def get_experiment(experiment_id: str):
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiments[experiment_id]

@app.post("/experiments/{experiment_id}/metrics/")
def log_metric(experiment_id: str, metric: MetricLog):
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if metric.key not in experiments[experiment_id]["metrics"]:
        experiments[experiment_id]["metrics"][metric.key] = []
    experiments[experiment_id]["metrics"][metric.key].append(metric.value)
    # Notify subscribers about the new metric
    for subscriber in subscribers:
        subscriber.put_nowait("update")
    return {"status": "success"}

@app.get("/events/")
async def message_stream():
    async def event_generator():
        queue = asyncio.Queue()
        subscribers.append(queue)
        try:
            while True:
                message = await queue.get()
                yield {"event": "update", "data": message}
        except asyncio.CancelledError:
            subscribers.remove(queue)
            raise
    return EventSourceResponse(event_generator())