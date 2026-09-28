# minimalistic utility to test and develop locally

SHELL = /bin/sh
.DEFAULT_GOAL := help

export IMAGE_PYTORCH=osparc-python-runner-pytorch
export IMAGE_TENSORFLOW=osparc-python-runner-tensorflow
export TAG_PYTORCH=1.1.2
export TAG_TENSORFLOW=1.1.2

# PYTHON ENVIRON ---------------------------------------------------------------------------------------
.PHONY: devenv
.venv:
	@python3 --version
	python3 -m venv $@
	# upgrading package managers
	$@/bin/pip install --upgrade uv

devenv: .venv  ## create a python virtual environment with tools to dev, run and tests cookie-cutter
	# installing extra tools
	@$</bin/uv pip install wheel setuptools
	# your dev environment contains
	@$</bin/uv pip list
	@echo "To activate the virtual environment, run 'source $</bin/activate'"

# Builds new service version ----------------------------------------------------------------------------
define _bumpversion
	# upgrades as $(subst $(1),,$@) version, commits and tags
	@docker run -i --rm -v $(PWD):/ml-runner \
		-u $(shell id -u):$(shell id -g) \
		itisfoundation/ci-service-integration-library:v2.2.1 \
		sh -c "cd /ml-runner && bump2version --verbose --list --config-file $(1) $(subst $(2),,$@)"
endef

.PHONY: version-tensorflow-patch version-tensorflow-minor version-tensorflow-major
version-tensorflow-patch version-tensorflow-minor version-tensorflow-major: .bumpversion-tensorflow.cfg ## increases tensroflow service's version
	@make compose-spec
	@$(call _bumpversion,$<,version-tensorflow-)
	@make compose-spec

.PHONY: version-pytorch-patch version-pytorch-minor version-pytorch-major
version-pytorch-patch version-pytorch-minor version-pytorch-major: .bumpversion-pytorch.cfg ## increases pytorchservice's version
	@make compose-spec
	@$(call _bumpversion,$<,version-pytorch-)
	@make compose-spec


define _create_run_script
	@docker run -i --rm -v $(PWD):/ml-runner \
		-u $(shell id -u):$(shell id -g) \
		itisfoundation/ci-service-integration-library:v2.2.1 \
		sh -c "cd /ml-runner && \
			ooil run-creator \
				--runscript $(1)/service.cli/run \
				--metadata .osparc/$(1)/metadata.yml"
endef

.PHONY: create-run-script
create-run-script: ## assembles run scrips for pytorch and tensorflow
	@$(call _create_run_script,osparc-python-runner-pytorch)
	@$(call _create_run_script,osparc-python-runner-tensorflow)
	# TODO: add check to make sure git state is not dirty in CI so it fails

.PHONY: compose-spec
compose-spec: ## runs ooil to assemble the docker-compose.yml file
	@docker run -i --rm -v $(PWD):/ml-runner \
		-u $(shell id -u):$(shell id -g) \
		itisfoundation/ci-service-integration-library:v2.2.1 \
		sh -c "cd /ml-runner && ooil compose"

build: | compose-spec	## build docker image
	docker compose build

# To test built service locally -------------------------------------------------------------------------
.PHONY: run-pytorch-local
run-pytorch-local: ## runs pytorch image with local configuration
	find validation-pytorch/outputs -type f ! -name '.gitkeep' -delete
	IMAGE_TO_RUN=${IMAGE_PYTORCH} \
	TAG_TO_RUN=${TAG_PYTORCH} \
	VALIDATION_DIR=validation-pytorch \
	docker compose --file docker-compose-local.yml up --abort-on-container-exit --exit-code-from runner-ml

.PHONY: run-tensorflow-local
run-tensorflow-local: ## runs tensorflow image with local configuration
	find validation-tensorflow/outputs -type f ! -name '.gitkeep' -delete
	IMAGE_TO_RUN=${IMAGE_TENSORFLOW} \
	TAG_TO_RUN=${TAG_TENSORFLOW} \
	VALIDATION_DIR=validation-tensorflow \
	docker compose --file docker-compose-local.yml up --abort-on-container-exit --exit-code-from runner-ml

# GPU regression baseline -------------------------------------------------------------------------------
# three layers, cheapest first:
#   L1 pure validation logic (no docker)  L2 container contract (docker inspect, no GPU)  L3 full GPU regression
define _validate_contract
	# layer 2: container contract for $(1):$(2)
	@uv run python tools/tests/test_container_contract.py $(1) --tag $(2)
endef

define _validate_regression
	# layer 3: deterministic GPU regression fixture for $(1)
	@uv run python tools/validate_ml_regression.py $(1) --tag $(2)
endef

.PHONY: test-unit
test-unit: ## layer 1: pure regression validation logic tests (no docker required)
	@uv run python tools/tests/test_regression_logic.py

.PHONY: validate-pytorch
validate-pytorch: test-unit build ## runs pytorch GPU regression (L1 logic, L2 contract, L3 GPU; override tag with TAG_PYTORCH=x.y.z)
	@$(call _validate_contract,pytorch,${TAG_PYTORCH})
	@$(call _validate_regression,pytorch,${TAG_PYTORCH})

.PHONY: validate-tensorflow
validate-tensorflow: test-unit build ## runs tensorflow GPU regression (L1 logic, L2 contract, L3 GPU; override tag with TAG_TENSORFLOW=x.y.z)
	@$(call _validate_contract,tensorflow,${TAG_TENSORFLOW})
	@$(call _validate_regression,tensorflow,${TAG_TENSORFLOW})

.PHONY: validate-regression
validate-regression: validate-pytorch validate-tensorflow ## runs both GPU regression fixtures

.PHONY: publish-local
publish-local: ## push to local throw away registry to test integration
	docker tag simcore/services/comp/${IMAGE_PYTORCH}:${TAG_PYTORCH} registry:5000/simcore/services/comp/${IMAGE_PYTORCH}:${TAG_PYTORCH}
	docker push registry:5000/simcore/services/comp/${IMAGE_PYTORCH}:${TAG_PYTORCH}
	docker tag simcore/services/comp/${IMAGE_TENSORFLOW}:${TAG_TENSORFLOW} registry:5000/simcore/services/comp/${IMAGE_TENSORFLOW}:${TAG_TENSORFLOW}
	docker push registry:5000/simcore/services/comp/${IMAGE_TENSORFLOW}:${TAG_TENSORFLOW}
	@curl registry:5000/v2/_catalog | jq

.PHONY: help
help: ## this colorful help
	@echo "Recipes for '$(notdir $(CURDIR))':"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[[:alpha:][:space:]_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
