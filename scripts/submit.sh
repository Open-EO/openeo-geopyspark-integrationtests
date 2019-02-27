#!/bin/sh -e

jobName=$1

export SPARK_MAJOR_VERSION=2
export SPARK_HOME=/usr/hdp/current/spark2-client
export PYSPARK_PYTHON="./python"
export PATH="$SPARK_HOME/bin:$PATH"

pushd venv/
zip -rq ../venv.zip *
popd

hdfsVenvDir=${jobName}

hadoop fs -mkdir -p ${hdfsVenvDir}
hadoop fs -put -f venv.zip ${hdfsVenvDir}

hdfsVenvZip=hdfs:/user/jenkins/${hdfsVenvDir}/venv.zip

extensions=$(ls geotrellis-extensions-*.jar)
backend_assembly=$(find $VIRTUAL_ENV -name 'geotrellis-backend-assembly-*.jar')

echo "Found backend assembly: ${backend_assembly}"

appId=$(yarn application -list 2>&1 | grep ${jobName} | awk '{print $1}')
if [ ! -z "$var" ]
then
    echo "Killing running intergration test service: ${appId}"
    yarn application -kill ${appId} || true
fi

echo "Submitting: ${jobName}"
spark-submit \
 --master yarn --deploy-mode cluster \
 --principal jenkins@VGT.VITO.BE --keytab ${HOME}/jenkins.keytab \
 --driver-memory 4G \
 --conf spark.executor.memory=8G \
 --conf spark.speculation=true \
 --conf spark.speculation.quantile=0.4 --conf spark.speculation.multiplier=1.1 \
 --conf spark.dynamicAllocation.minExecutors=20 \
 --conf spark.locality.wait=300ms --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true \
 --conf "spark.yarn.appMasterEnv.SPARK_HOME=$SPARK_HOME" --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
 --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
 --conf "spark.yarn.appMasterEnv.OPENEO_VENV_ZIP=$hdfsVenvZip" \
 --files python,$(ls typing-*-none-any.whl),openeo-geopyspark-driver/layercatalog.json,openeo-geopyspark-driver/scripts/submit_batch_job.sh,openeo-geopyspark-driver/scripts/log4j.properties,openeo-geopyspark-driver/openeogeotrellis/deploy/batch_job.py \
 --archives "${hdfsVenvZip}#venv" \
 --conf spark.hadoop.security.authentication=kerberos --conf spark.yarn.maxAppAttempts=1 \
 --jars ${extensions},${backend_assembly} \
 --name ${jobName} openeo-geopyspark-driver/openeogeotrellis/deploy/probav-mep.py no-zookeeper 2>&1 > /dev/null &
