FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir '.[xgb]' -c configs/verified-constraints.txt
COPY api ./api
COPY dashboard ./dashboard
COPY docs ./docs
COPY .streamlit ./.streamlit
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
