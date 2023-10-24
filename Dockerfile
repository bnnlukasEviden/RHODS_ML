FROM docker.stackable.tech/stackable/pyspark-k8s:3.3.0-stackable23.7.0

RUN pip install onnxmltools && ls && pwd && ls source