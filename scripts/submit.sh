#!/bin/sh

export HDP_VERSION=2.5.3.0-37
export SPARK_MAJOR_VERSION=2
export SPARK_HOME=/usr/hdp/current/spark2-client
export PYSPARK_PYTHON="/usr/bin/python3.5"

cd venv/lib/python*/site-packages && \
ln -s geopyspark-0.3.0-*.egg/geopyspark && \
zip -9 -r ../../../../libs.zip . -x \*pandas\* -x \*numpy\* && \
cd ../../../..

appId=$(yarn application -list 2>&1 | grep OpenEO-GeoPySpark-test | awk '{print $1}')
yarn application -kill ${appId}

assembly="geopyspark/geopyspark/jars/geotrellis-backend-assembly-0.3.0.jar"

spark-submit \
 --master yarn --deploy-mode cluster \
 --conf spark.speculation=true \
 --conf spark.speculation.quantile=0.4 --conf spark.speculation.multiplier=1.1 \
 --conf spark.dynamicAllocation.minExecutors=30 \
 --conf spark.yarn.appMasterEnv.SPARK_HOME=/usr/hdp/current/spark2-client --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
 --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=/usr/bin/python3.5 --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.5 \
 --conf spark.locality.wait=300ms --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.extraClassPath=$(basename "${assembly}") \
 --driver-class-path $(basename "${assembly}") \
 --files ${assembly} --conf spark.hadoop.security.authentication=kerberos --conf spark.yarn.maxAppAttempts=1 \
 --py-files $(find openeo-python-client/dist -name 'openeo_api-*.egg'),$(find openeo-python-driver/dist -name 'openeo_driver*.egg'),$(find openeo-geopyspark-driver/dist -name 'openeo_geopyspark*.egg'),libs.zip \
 --name OpenEO-GeoPySpark-test openeo-geopyspark-driver/openeogeotrellis/deploy/probav-mep.py 2>&1 > /dev/null &
