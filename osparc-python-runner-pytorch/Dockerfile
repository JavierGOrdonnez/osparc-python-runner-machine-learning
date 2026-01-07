FROM simcore/services/comp/common:0.0.0 AS service-base

LABEL maintainer="Andrei Neagu <neagu@speag.swiss>"

COPY --chown=${SC_USER_NAME}:${SC_USER_NAME} service.cli/ service.cli/
COPY --chown=${SC_USER_NAME}:${SC_USER_NAME} osparc_python_runner_pytorch osparc_python_runner_pytorch
RUN mv main.py osparc_python_runner_pytorch/main.py

RUN uv pip install -r osparc_python_runner_pytorch/requirements.txt

# Make venv writable for runtime package installation
# The scu user needs to be able to install packages from user requirements.txt
RUN chmod -R 777 /opt/venv