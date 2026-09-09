import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import GateCNN, KilterCNN

MODEL_PATH = "models/kilter_cnn.pt"
GATE_MODEL_PATH = "models/kilter_gate.pt"
MAX_ANGLE = 70.0
ROLE_TO_CHANNEL = {12: 0, 13: 1, 14: 2, 15: 3}  # start, middle, finish, foot

# loaded once at startup, reused for every request
model = KilterCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# runs ahead of the grade model -- rejects climbs the grade model has no
# real basis for judging (e.g. too few holds, or a gap far beyond anything
# in the training data) instead of letting it silently extrapolate a grade
# with no support for it. see generate_gate_dataset.py for why this exists.
gate_model = GateCNN()
gate_model.load_state_dict(torch.load(GATE_MODEL_PATH, map_location="cpu"))
gate_model.eval()

app = FastAPI()

# the frontend runs on a different port during dev, browsers block that
# by default unless the server explicitly allows it -- matched by pattern
# rather than a fixed port, since next dev picks a different one whenever
# 3000 is already taken
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    cells: list[tuple[int, int, int]]  # (col, row, role_id)
    angle: float


class PredictResponse(BaseModel):
    probs: list[float]
    valid: bool
    valid_confidence: float  # gate's own confidence in that valid/invalid call


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    image = torch.zeros(1, 4, 38, 47)
    for col, row, role in req.cells:
        channel = ROLE_TO_CHANNEL.get(role)
        if channel is not None:
            image[0, channel, row, col] = 1.0

    angle_tensor = torch.tensor([req.angle / MAX_ANGLE], dtype=torch.float32)

    with torch.no_grad():
        gate_probs = F.softmax(gate_model(image, angle_tensor), dim=1)[0]
        valid = bool(gate_probs[1] >= gate_probs[0])
        valid_confidence = gate_probs[1 if valid else 0].item()

        # still run the grade model either way -- cheap, and the frontend
        # can choose whether to show a rejected climb's number at all. what
        # it must not do is present that number with the same confidence as
        # a climb the gate actually accepted.
        logits = model(image, angle_tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()

    return {"probs": probs, "valid": valid, "valid_confidence": valid_confidence}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
