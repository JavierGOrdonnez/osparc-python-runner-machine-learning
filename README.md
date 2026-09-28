
# osparc-python-runner-pytorch and osparc-python-runner-tensorflow

This repository contains the source code for two o²S²PARC Services: osparc-python-runner-pytorch and osparc-python-runner-tensorflow

Building the docker images:

```shell
make build
```


Test the built images locally:


**pytorch**
```shell
make run-pytorch-local
```

**tensorflow**
```shell
make run-tensorflow-local
```

## Regression baseline

The machine-learning runners are GPU services. The deterministic regression fixtures require Docker's NVIDIA runtime and a visible NVIDIA GPU; they fail if either framework falls back to the CPU. The local `docker-compose-local.yml` already requests the `nvidia` runtime, so no extra GPU compose file is needed.

Run one fixture per framework. Each target builds the images and runs three layers, cheapest first: layer 1 (pure validation logic, no Docker), layer 2 (container contract, `docker inspect`, no GPU), then layer 3 (full GPU regression):

```shell
make validate-pytorch
make validate-tensorflow
```

Or run both in one go:

```shell
make validate-regression
```

The harness creates disposable uid-8004-owned mounts under `.validation-runs/`, executes through the normal uploaded-code and archive
collection path, and removes its Docker Compose project afterward. It prints a JSON measurement containing image identity/size, GPU evidence, startup to model-ready, model load, warmed inference, output collection, and total container time. The existing `make run-*-local` commands remain separate network smoke checks. To validate a specific tag, override the version variable, e.g. `make validate-pytorch TAG_PYTORCH=1.1.3`.

### Initial local baseline

Measured on `ordonez-wkst` (WSL2, NVIDIA GeForce RTX 3050) with locally built `1.1.2` images. Timings are observational comparison values, not pass/failthresholds; rerun the commands above after an intentional runtime change.

| Framework | Image size | GPU evidence | Startup to ready | Model load | Warmed inference | Output collection | Container wall |
|---|---:|---|---:|---:|---:|---:|---:|
| PyTorch | 11,559,024,272 B | `cuda:0` | 94.072 s | 0.228 s | 0.008 s | 0.001 s | 97.980 s |
| TensorFlow | 9,444,168,492 B | `GPU:0` | 100.596 s | 0.356 s | 0.006 s | 0.001 s | 104.303 s |


Raising the version can be achieved via one for three methods. The `major`,`minor` or `patch` can be bumped, for example:


**pytorch**
```shell
make version-pytorch-patch
```

**tensorflow**
```shell
make version-tensorflow-patch
```


If you already have a local copy of **o<sup>2</sup>S<sup>2</sup>PARC** running and wish to push data to the local registry:

```shell
make publish-local
```
