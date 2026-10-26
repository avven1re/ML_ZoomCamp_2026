FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app


COPY ["pyproject.toml", ".python-version", "uv.lock", "./"]

RUN uv sync --locked --no-dev

COPY predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

