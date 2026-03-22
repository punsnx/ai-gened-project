from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.classifier import classify_image_bytes, load_model


class HealthResponse(BaseModel):
    status: str


class InferenceResponse(BaseModel):
    result: str
    percent: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Image Classification API",
    version="1.0.0",
    description="Classifies uploaded images into nothing, drinking_water, or food.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def read_root() -> str:
    html_path = Path(__file__).resolve().parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


async def _run_inference(file: UploadFile) -> InferenceResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        result, percent = classify_image_bytes(image_bytes=image_bytes)
        return InferenceResponse(result=result, percent=percent)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to classify image: {exc}") from exc


@app.post("/inference", response_model=InferenceResponse)
async def inference_post(file: UploadFile = File(...)) -> InferenceResponse:
    return await _run_inference(file)


@app.get("/inference", response_model=InferenceResponse)
async def inference_get(file: UploadFile = File(...)) -> InferenceResponse:
    return await _run_inference(file)
