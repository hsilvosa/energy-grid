FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/fixtures ./data/fixtures
RUN pip install --upgrade pip && pip install ".[streaming]"

RUN addgroup --system energy && adduser --system --ingroup energy energy \
    && mkdir -p /app/data/runtime && chown -R energy:energy /app
USER energy

EXPOSE 8000
CMD ["uvicorn", "energy_grid.api:app", "--host", "0.0.0.0", "--port", "8000"]
