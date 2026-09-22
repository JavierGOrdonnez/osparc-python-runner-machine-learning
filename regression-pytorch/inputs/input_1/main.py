import json
import os
import time
from pathlib import Path

import torch


input_dir = Path(os.environ["INPUT_1"])
output_dir = Path(os.environ["OUTPUT_1"])
if not torch.cuda.is_available():
    raise RuntimeError("GPU validation requested but PyTorch CUDA is unavailable")

device = torch.device("cuda")
(output_dir / "ready.json").write_text(json.dumps({"device": str(device)}))

load_started = time.perf_counter()
model = torch.jit.load(input_dir / "model.pt", map_location=device)
model.eval()
load_seconds = time.perf_counter() - load_started
samples = torch.tensor(json.loads((input_dir / "samples.json").read_text()), device=device)
with torch.inference_mode():
    model(samples[:1])

inference_started = time.perf_counter()
with torch.inference_mode():
    probabilities_tensor = model(samples)
torch.cuda.synchronize()
inference_seconds = time.perf_counter() - inference_started
probabilities = probabilities_tensor.cpu().tolist()

(output_dir / "result.json").write_text(
    json.dumps(
        {
            "actual_device": "gpu" if probabilities_tensor.device.type == "cuda" else "cpu",
            "framework": "pytorch",
            "native_device": str(probabilities_tensor.device),
            "predictions": [max(range(len(row)), key=row.__getitem__) for row in probabilities],
            "probabilities": probabilities,
            "timings_seconds": {
                "model_load": load_seconds,
                "warmed_inference": inference_seconds,
            },
        },
        sort_keys=True,
    )
)
