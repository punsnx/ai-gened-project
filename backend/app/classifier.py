from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES_DEFAULT = ["drinking_water", "food", "nothing"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "food-detector-model.pth"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: torch.nn.Module | None = None
_class_names: list[str] = CLASS_NAMES_DEFAULT.copy()


def _build_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_features, num_classes))
    return model.to(_device)


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location=_device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=_device)


def _extract_state_dict_and_classes(payload) -> tuple[dict, list[str] | None]:
    class_names = None
    if isinstance(payload, dict):
        if isinstance(payload.get("class_names"), (list, tuple)):
            class_names = [str(x) for x in payload["class_names"]]

        if "model_state_dict" in payload:
            return payload["model_state_dict"], class_names
        if "state_dict" in payload:
            return payload["state_dict"], class_names

    return payload, class_names


def load_model() -> None:
    global _model, _class_names
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    payload = _load_checkpoint(MODEL_PATH)
    state_dict, class_names = _extract_state_dict_and_classes(payload)
    num_classes = int(state_dict["fc.1.weight"].shape[0]) if "fc.1.weight" in state_dict else len(CLASS_NAMES_DEFAULT)

    model = _build_model(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    if class_names and len(class_names) == num_classes:
        _class_names = class_names
    elif num_classes == len(CLASS_NAMES_DEFAULT):
        _class_names = CLASS_NAMES_DEFAULT.copy()
    else:
        _class_names = [f"class_{i}" for i in range(num_classes)]

    _model = model


def _preprocess(image: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0).to(_device)


def classify_image_bytes(image_bytes: bytes) -> tuple[str, float]:
    if _model is None:
        raise RuntimeError("Model is not loaded")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = _preprocess(image)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    idx = int(torch.argmax(probs).item())
    result = _class_names[idx]
    percent = round(float(probs[idx]) * 100.0, 2)
    return result, percent
