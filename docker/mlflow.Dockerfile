FROM ghcr.io/mlflow/mlflow:v3.15.1
RUN pip install --no-cache-dir psycopg[binary] boto3
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "mlflow:5000,localhost:*,127.0.0.1:*", "--backend-store-uri", "postgresql+psycopg://energy:energy@postgres:5432/mlflow", "--default-artifact-root", "s3://mlflow-artifacts"]
