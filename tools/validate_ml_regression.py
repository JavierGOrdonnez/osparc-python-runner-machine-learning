#!/usr/bin/env python3
"""Runs GPU model-regression fixtures through the uploaded-code runner."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / ".validation-runs"
SERVICE_IMAGES = {
    "pytorch": "osparc-python-runner-pytorch",
    "tensorflow": "osparc-python-runner-tensorflow",
}


def _run(command, *, env):
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)


def _image_facts(image):
    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        text=True,
        capture_output=True,
        check=True,
    )
    inspected = json.loads(completed.stdout)
    return {"id": inspected["Id"], "size_bytes": inspected["Size"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=SERVICE_IMAGES)
    parser.add_argument("--tag", default="1.1.2")
    args = parser.parse_args()

    fixture = ROOT / f"regression-{args.framework}"
    run_dir = RUNS_DIR / f"{args.framework}-{uuid.uuid4().hex}"
    subprocess.run(["docker", "run", "--rm", "-v", f"{fixture}:/fixture:ro", "-v", f"{run_dir}:/run", "alpine:3.20", "sh", "-c", "cp -a /fixture/inputs /run/ && mkdir -p /run/outputs && chown -R 8004:8004 /run && chmod -R u+rwX /run"], check=True)
    env = {
        **os.environ,
        "IMAGE_TO_RUN": SERVICE_IMAGES[args.framework],
        "TAG_TO_RUN": args.tag,
        "VALIDATION_DIR": str(run_dir.relative_to(ROOT)),
    }
    command = [
        "docker",
        "compose",
        "--file",
        "docker-compose-local.yml",
        "--file",
        "docker-compose-local-gpu.yml",
    ]
    project_name = f"ml-regression-{uuid.uuid4().hex}"
    project_command = command + ["--project-name", project_name]
    command = project_command + ["up", "--abort-on-container-exit", "--exit-code-from", "runner-ml"]

    started = time.perf_counter()
    ready = run_dir / "outputs" / "output_1" / "ready.json"
    ready_seconds = None
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while process.poll() is None:
            if ready_seconds is None and ready.is_file():
                ready_seconds = time.perf_counter() - started
            time.sleep(0.05)
        stdout, stderr = process.communicate()
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        _run(project_command + ["down", "--remove-orphans"], env=env)
    wall_seconds = time.perf_counter() - started
    logs = completed.stdout + completed.stderr
    archive = run_dir / "outputs" / "output_1.zip"
    if completed.returncode:
        raise RuntimeError(logs)
    if not archive.is_file():
        raise AssertionError(f"Missing declared output archive: {archive}\n{logs}")
    if ready_seconds is None:
        raise AssertionError(f"Service never reached model-ready state\n{logs}")

    collection_started = time.perf_counter()
    with zipfile.ZipFile(archive) as output:
        result = json.loads(output.read("result.json"))
    collection_seconds = time.perf_counter() - collection_started
    golden = json.loads((run_dir / "inputs" / "input_1" / "golden.json").read_text())
    if result["predictions"] != golden["predictions"]:
        raise AssertionError(f"Unexpected predictions: {result['predictions']}")
    for actual_row, expected_row in zip(result["probabilities"], golden["probabilities"], strict=True):
        for actual, expected in zip(actual_row, expected_row, strict=True):
            if abs(actual - expected) > golden["probability_tolerance"]:
                raise AssertionError(f"Unexpected probability: {actual} != {expected}")
    if result["actual_device"] != "gpu":
        raise AssertionError(f"Device fallback detected: {result}")

    measurement = {
        "framework": args.framework,
        "device": "gpu",
        "image": f"simcore/services/comp/{SERVICE_IMAGES[args.framework]}:{args.tag}",
        "image_facts": _image_facts(
            f"simcore/services/comp/{SERVICE_IMAGES[args.framework]}:{args.tag}"
        ),
        "machine": platform.node(),
        "platform": platform.platform(),
        "timings_seconds": {
            "container_wall": wall_seconds,
            "startup_to_ready": ready_seconds,
            "model_load": result["timings_seconds"]["model_load"],
            "warmed_inference": result["timings_seconds"]["warmed_inference"],
            "output_collection": collection_seconds,
        },
        "result": result,
    }
    print(json.dumps(measurement, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
