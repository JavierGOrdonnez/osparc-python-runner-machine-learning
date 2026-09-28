#!/usr/bin/env python3
"""Layer 2: container-contract tests for the built ML runner images.

Faster than a full regression run and needs no GPU: it inspects the built
image and the local compose file to confirm the contract the Layer 3 harness
relies on still holds -- the runner user id, the input/output folder env
contract, the entrypoint shape, and that the local compose requests the
nvidia runtime. Runs after `build`, before the full GPU regression.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVICE_IMAGES = {
    "pytorch": "osparc-python-runner-pytorch",
    "tensorflow": "osparc-python-runner-tensorflow",
}
SC_USER_ID = "8004"


def _image_inspect(image):
    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise AssertionError(
            f"image {image} not built. run `make build` first.\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _env_map(configured):
    mapping = {}
    for entry in configured or []:
        key, _, value = entry.partition("=")
        mapping[key] = value
    return mapping


def check_image(framework, tag):
    image = f"simcore/services/comp/{SERVICE_IMAGES[framework]}:{tag}"
    inspected = _image_inspect(image)
    config = inspected["Config"]
    env = _env_map(config.get("Env"))

    assert env.get("SC_USER_ID") == SC_USER_ID, (
        f"{image}: SC_USER_ID must be {SC_USER_ID}"
    )
    assert env.get("INPUT_FOLDER"), f"{image}: INPUT_FOLDER env contract missing"
    assert env.get("OUTPUT_FOLDER"), f"{image}: OUTPUT_FOLDER env contract missing"

    entry = config.get("Entrypoint") or []
    assert entry and entry[-1].endswith("entrypoint.sh"), (
        f"{image}: expected entrypoint to end in entrypoint.sh, got {entry}"
    )


def check_compose_runtime():
    compose_file = ROOT / "docker-compose-local.yml"
    text = compose_file.read_text()
    assert "runtime: nvidia" in text, (
        f"{compose_file}: must request the nvidia runtime for GPU regression"
    )
    assert "NVIDIA_VISIBLE_DEVICES" in text, (
        f"{compose_file}: must set NVIDIA_VISIBLE_DEVICES for GPU regression"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=SERVICE_IMAGES)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    check_image(args.framework, args.tag)
    check_compose_runtime()
    print(f"container contract OK: {args.framework}:{args.tag}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
