#!/bin/sh

export HDP_VERSION=2.5.3.0-37
export SPARK_MAJOR_VERSION=2
export SPARK_HOME=/usr/hdp/current/spark2-client
export PYSPARK_PYTHON="/usr/bin/python3.5"

readlink -f openeo-python-client/dist/openeo_api-0.0.1-py3.5.egg
readlink -f openeo-python-driver/dist/openeo_driver-0.0.0-py3.5.egg
readlink -f openeo-geopyspark-driver/dist/openeo_integration_tests-0.0.0-py3.5.egg
