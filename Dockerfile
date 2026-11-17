FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RESEARCH_NOTEBOOK=/app/data/notebook.db
WORKDIR /app
COPY pyproject.toml README.md ./
COPY research_agent ./research_agent
RUN pip install --no-cache-dir . && useradd --create-home app && mkdir -p /app/data && chown app /app/data
USER app
VOLUME ["/app/data"]
ENTRYPOINT ["research-agent"]
CMD ["--help"]
