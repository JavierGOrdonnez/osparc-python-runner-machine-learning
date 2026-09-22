import json
import os
import time
from pathlib import Path

import tensorflow as tf


input_dir = Path(os.environ["INPUT_1"])
output_dir = Path(os.environ["OUTPUT_1"])
if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("GPU validation requested but TensorFlow GPU is unavailable")

(output_dir / "ready.json").write_text(json.dumps({"device": "/GPU:0"}))
load_started = time.perf_counter()
with tf.device("/GPU:0"):
    model = tf.keras.models.load_model(input_dir / "model.keras")
load_seconds = time.perf_counter() - load_started
samples = tf.constant(json.loads((input_dir / "samples.json").read_text()), dtype=tf.float32)
with tf.device("/GPU:0"):
    model(samples[:1], training=False)

inference_started = time.perf_counter()
with tf.device("/GPU:0"):
    probabilities_tensor = model(samples, training=False)
    probabilities = probabilities_tensor.numpy().tolist()
inference_seconds = time.perf_counter() - inference_started

(output_dir / "result.json").write_text(
    json.dumps(
        {
            "actual_device": "gpu" if "GPU" in probabilities_tensor.device else "cpu",
            "framework": "tensorflow",
            "native_device": probabilities_tensor.device,
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
