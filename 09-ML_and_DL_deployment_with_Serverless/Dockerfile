FROM public.ecr.aws/lambda/python:3.14

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN uv export --locked --no-dev --no-hashes --no-emit-project -o requirements.txt && \
    uv pip install --system -r requirements.txt

COPY model.bin .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]