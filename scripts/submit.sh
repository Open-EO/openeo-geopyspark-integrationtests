#!/usr/bin/env bash

set -eo pipefail

jobName=$1
version=$2
pysparkPython="venv/bin/python"

export HDP_VERSION=3.1.0.0-78
export SPARK_MAJOR_VERSION=2
export SPARK_HOME=/opt/spark3_0_0
export PATH="$SPARK_HOME/bin:$PATH"

export PYTHONPATH="venv/lib64/python3.8/site-packages:venv/lib/python3.8/site-packages"

hdfsVenvZip=https://artifactory.vgt.vito.be/auxdata-public/openeo/dev/openeo-"${version}".zip
extensions=https://artifactory.vgt.vito.be/auxdata-public/openeo/geotrellis-extensions-2.0.0_2.12-SNAPSHOT.jar
backend_assembly=https://artifactory.vgt.vito.be/auxdata-public/openeo/geotrellis-backend-assembly-0.4.6-openeo_2.12.jar

echo "Found backend assembly: ${backend_assembly}"

echo "Submitting Spark job ${jobName}"
date
${SPARK_HOME}/bin/spark-submit \
 --master yarn --deploy-mode cluster \
 --queue lowlatency \
 --driver-memory 2G \
 --principal jenkins@VGT.VITO.BE --keytab ${HOME}/jenkins.keytab \
 --conf spark.executor.cores=2 \
 --driver-java-options "-Dlog4j.debug=true -Dlog4j.configuration=file:log4j.properties" \
 --conf spark.driver.memoryOverhead=3g \
 --conf spark.executor.memoryOverhead=512m \
 --conf spark.executor.memory=2G \
 --conf spark.speculation=true \
 --conf spark.speculation.quantile=0.4 --conf spark.speculation.multiplier=1.1 \
 --conf spark.dynamicAllocation.minExecutors=5 \
 --conf spark.locality.wait=300ms --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true \
 --conf spark.yarn.submit.waitAppCompletion=false \
 --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
 --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$pysparkPython" \
 --conf "spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=$pysparkPython" \
 --conf spark.executorEnv.LD_LIBRARY_PATH=venv/lib64 --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=venv/lib64 \
 --conf "spark.yarn.appMasterEnv.OPENEO_VENV_ZIP=$hdfsVenvZip" \
 --conf spark.executorEnv.DRIVER_IMPLEMENTATION_PACKAGE=openeogeotrellis --conf spark.yarn.appMasterEnv.DRIVER_IMPLEMENTATION_PACKAGE=openeogeotrellis \
 --conf spark.yarn.appMasterEnv.WMTS_BASE_URL_PATTERN=http://tsviewer-rest-test.vgt.vito.be/openeo/services/%s \
 --conf spark.executorEnv.AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --conf spark.yarn.appMasterEnv.AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
 --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} --conf spark.yarn.appMasterEnv.AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
 --conf spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_DOCKER_MOUNTS=/var/lib/sss/pipes:/var/lib/sss/pipes:rw,/usr/hdp/current/:/usr/hdp/current/:ro,/etc/hadoop/conf/:/etc/hadoop/conf/:ro,/etc/krb5.conf:/etc/krb5.conf:ro,/data/MTDA:/data/MTDA:ro \
 --conf spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_TYPE=docker \
 --conf spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE=vito-docker-private-dev.artifactory.vgt.vito.be/python38-hadoop \
 --conf spark.executorEnv.YARN_CONTAINER_RUNTIME_TYPE=docker \
 --conf spark.executorEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE=vito-docker-private-dev.artifactory.vgt.vito.be/python38-hadoop \
 --conf spark.executorEnv.YARN_CONTAINER_RUNTIME_DOCKER_MOUNTS=/var/lib/sss/pipes:/var/lib/sss/pipes:rw,/usr/hdp/current/:/usr/hdp/current/:ro,/etc/hadoop/conf/:/etc/hadoop/conf/:ro,/etc/krb5.conf:/etc/krb5.conf:ro,/data/MTDA:/data/MTDA:ro \
 --files venv38/layercatalog.json,venv38/log4j.properties \
 --py-files tests/data/custom_processes.py \
 --archives "${hdfsVenvZip}#venv" \
 --conf spark.hadoop.security.authentication=kerberos --conf spark.yarn.maxAppAttempts=1 \
 --jars ${extensions},${backend_assembly} \
 --name ${jobName} openeogeotrellis.deploy.probav-mep.py

echo "Submitted"
date
