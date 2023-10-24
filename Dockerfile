FROM docker.stackable.tech/stackable/pyspark-k8s:3.3.0-stackable23.7.0

COPY /workspace/input/main_spark.py /stackable/spark/

RUN pip install onnxmltools