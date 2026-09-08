import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import KilterCNN

MODEL_PATH = "models/kilter_cnn.pt"
MAX_ANGLE = 70.0
ROLE_TO_CHANNEL = {12: 0, 13: 1, 14: 2, 15: 3}  # start, middle, finish, foot

# loaded once at startup, reused for every request
model = KilterCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    image = torch.zeros(1, 4, 38, 47)
    for col, row, role in req.cells:
        channel = ROLE_TO_CHANNEL.get(role)
        if channel is not None:
            image[0, channel, row, col] = 1.0

    angle_tensor = torch.tensor([req.angle / MAX_ANGLE], dtype=torch.float32)

    with torch.no_grad():
        logits = model(image, angle_tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()

    return {"probs": probs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
