FROM quay.io/fedora/python-313:latest
MAINTAINER Mario Vazquez <mavazque@redhat.com>

USER root
RUN mkdir -p /opt/ragapi

# Install uv
RUN pip3 install uv

# Copy dependency files
COPY ../pyproject.toml /opt/ragapi/pyproject.toml

WORKDIR /opt/ragapi

# Install dependencies using uv
RUN uv sync --frozen --no-dev

USER 1024
ADD ../src/ /opt/ragapi/

ENTRYPOINT ["uv", "run", "python3", "-u", "/opt/ragapi/cli.py"]