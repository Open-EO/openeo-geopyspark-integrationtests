#!/bin/sh -e

jobName=$1

export SPARK_HOME=/usr/hdp/current/spark2-client
export PYSPARK_PYTHON="./python"
export PATH="$SPARK_HOME/bin:$PATH"

pushd venv/
zip -rq ../venv.zip *
popd

extensions=$(ls GeoPySparkExtensions-*.jar)
backend_assembly=$(find $VIRTUAL_ENV -name 'geotrellis-backend-assembly-*.jar')

appId=$(yarn application -list 2>&1 | grep ${jobName} | awk '{print $1}')
yarn application -kill ${appId} || true

spark-submit \
 --master yarn --deploy-mode cluster \
 --principal jenkins@VGT.VITO.BE --keytab ${HOME}/jenkins.keytab \
 --driver-memory 4G \
 --conf spark.executor.memory=8G \
 --conf spark.speculation=true \
 --conf spark.speculation.quantile=0.4 --conf spark.speculation.multiplier=1.1 \
 --conf spark.dynamicAllocation.minExecutors=20 \
 --conf "spark.yarn.appMasterEnv.SPARK_HOME=$SPARK_HOME" --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
 --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
 --conf spark.locality.wait=300ms --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true \
 --files python,$(ls typing-*-none-any.whl),openeo-geopyspark-driver/layercatalog.json,openeo-geopyspark-driver/scripts/submit_batch_job.sh,openeo-geopyspark-driver/scripts/log4j.properties,openeo-geopyspark-driver/openeogeotrellis/deploy/batch_job.py \
 --archives "venv.zip#venv" \
 --conf spark.hadoop.security.authentication=kerberos --conf spark.yarn.maxAppAttempts=1 \
 --jars ${extensions},${backend_assembly} \
 --name ${jobName} openeo-geopyspark-driver/openeogeotrellis/deploy/probav-mep.py no-zookeeper 2>&1 > /dev/null &
