#!/usr/bin/env bash

set -eo pipefail

jobName=$1
version=$2
pysparkPython="venv/bin/python"

export HDP_VERSION=3.1.0.0-78
export SPARK_MAJOR_VERSION=2
export SPARK_HOME=/usr/hdp/$HDP_VERSION/spark2/
export PATH="$SPARK_HOME/bin:$PATH"

export PYTHONPATH="venv/lib64/python3.6/site-packages:venv/lib/python3.6/site-packages"

hdfsVenvZip=https://artifactory.vgt.vito.be/auxdata-public/openeo/dev/openeo-"${version}".zip
extensions=https://artifactory.vgt.vito.be/libs-snapshot-public/org/openeo/geotrellis-extensions/2.1.0-SNAPSHOT/geotrellis-extensions-2.1.0-SNAPSHOT.jar
backend_assembly=https://artifactory.vgt.vito.be/auxdata-public/openeo/geotrellis-backend-assembly-0.4.6-openeo.jar

echo "Found backend assembly: ${backend_assembly}"

echo "Submitting Spark job ${jobName}"
date
${SPARK_HOME}/bin/spark-submit \
 --master yarn --deploy-mode cluster \
 --queue lowlatency \
 --principal jenkins@VGT.VITO.BE --keytab ${HOME}/jenkins.keytab \
 --driver-memory 2G \
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
 --conf spark.executorEnv.LD_LIBRARY_PATH=venv/lib64:/tmp_epod/gdal --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=venv/lib64:/tmp_epod/gdal \
 --conf spark.executorEnv.PROJ_LIB=/tmp_epod/gdal/data --conf spark.yarn.appMasterEnv.PROJ_LIB=/tmp_epod/gdal/data \
 --conf "spark.yarn.appMasterEnv.OPENEO_VENV_ZIP=$hdfsVenvZip" \
 --conf spark.executorEnv.DRIVER_IMPLEMENTATION_PACKAGE=openeogeotrellis --conf spark.yarn.appMasterEnv.DRIVER_IMPLEMENTATION_PACKAGE=openeogeotrellis \
 --conf spark.yarn.appMasterEnv.WMTS_BASE_URL_PATTERN=http://tsviewer-rest-test.vgt.vito.be/openeo/services/%s \
 --conf spark.executorEnv.AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --conf spark.yarn.appMasterEnv.AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
 --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} --conf spark.yarn.appMasterEnv.AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
 --files venv36/layercatalog.json,venv36/log4j.properties \
 --py-files tests/data/custom_processes.py \
 --archives "${hdfsVenvZip}#venv" \
 --conf spark.hadoop.security.authentication=kerberos --conf spark.yarn.maxAppAttempts=1 \
 --jars ${extensions},${backend_assembly} \
 --name ${jobName} openeogeotrellis.deploy.probav-mep.py

echo "Submitted"
date