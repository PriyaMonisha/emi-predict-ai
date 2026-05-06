# filename: docker/airflow/Dockerfile
# purpose:  Airflow image with EMI Predict AI Python dependencies (Section 11)
# version:  1.1
#
# Extends apache/airflow:2.8.4-python3.11 and installs packages required by
# the EMI batch prediction DAG: MLflow, LightGBM, XGBoost, Evidently, Redis.

FROM apache/airflow:2.8.4-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY docker/airflow/requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt
