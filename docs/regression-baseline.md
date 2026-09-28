# Regression baseline

The regression targets run three layers, from cheapest to most comprehensive:

1. Pure validation-logic tests (no Docker).
2. Container contract checks (Docker inspection, no GPU).
3. The full GPU regression fixture.

The fixtures require Docker's NVIDIA runtime and a visible NVIDIA GPU. The local Compose configuration already requests the NVIDIA runtime. To validate a specific image tag, override the framework's tag variable, for example:

```shell
make validate-pytorch TAG_PYTORCH=1.1.3
```

The harness creates disposable mounts under `.validation-runs/`, executes through the normal uploaded-code and archive-collection path, and removes its Compose project afterward. It prints JSON with image identity, GPU evidence, and timing measurements. The `make run-*-local` targets remain separate local smoke checks.

## Initial local measurements

Measured on `ordonez-wkst` (WSL2, NVIDIA GeForce RTX 3050) with locally built `1.1.2` images. These are observational comparison values, not pass/fail thresholds.

| Framework | Image size | GPU evidence | Startup to ready | Model load | Warmed inference | Output collection | Container wall |
|---|---:|---|---:|---:|---:|---:|---:|
| PyTorch | 11,559,024,272 B | `cuda:0` | 94.072 s | 0.228 s | 0.008 s | 0.001 s | 97.980 s |
| TensorFlow | 9,444,168,492 B | `GPU:0` | 100.596 s | 0.356 s | 0.006 s | 0.001 s | 104.303 s |
