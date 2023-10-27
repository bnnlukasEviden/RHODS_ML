FROM docker.stackable.tech/stackable/pyspark-k8s:3.3.0-stackable23.7.0

COPY ./main_spark_lstm.py /stackable/spark

RUN pip install boto3 tensorflow openvino pandas numpy scikit-learn