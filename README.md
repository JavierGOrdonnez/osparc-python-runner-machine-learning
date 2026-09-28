
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

## Regression checks

The GPU regression fixtures require Docker's NVIDIA runtime and a visible NVIDIA GPU:

```shell
make validate-pytorch
make validate-tensorflow
make validate-regression  # both frameworks
```

See [`docs/regression-baseline.md`](docs/regression-baseline.md) for the validation layers and the initial local measurements.


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
