#!/usr/bin/env groovy

node("jenkinsslave1.vgt.vito.be") {
  properties([disableConcurrentBuilds()])

  deleteDir()
  checkout scm

  stage('Build and test') {
    sh '''
      git clone https://github.com/Open-EO/openeo-python-client.git
      git clone https://github.com/Open-EO/openeo-python-driver.git
      git clone https://github.com/Open-EO/openeo-geopyspark-driver.git
      git clone https://github.com/locationtech-labs/geopyspark.git
    '''

    sh '''
      python3 -m venv venv
      . venv/bin/activate

      pip install wheel pytest

      cd openeo-python-client
      pip install travis-sphinx==2.1.0 "sphinx<1.7"
      pip install -r requirements-dev.txt
      pip install -r requirements.txt
      pytest
      python setup.py install bdist_egg

      cd ../openeo-python-driver
      pip install -r requirements-dev.txt
      pip install -r requirements.txt
      python setup.py install bdist_egg

      cd ../geopyspark
      make virtual-install

      cd ../openeo-geopyspark-driver
      pip install $(cat requirements.txt | tr '\\n' ' ' | sed -e 's/\\+openeo1//' | sed -e 's/openeo-api==0.0.1/openeo-api/') --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-packages-public/simple
      SPARK_HOME=$(find_spark_home.py) geopyspark install-jar
      echo SPARK_HOME=$(find_spark_home.py) pytest
      python setup.py install bdist_egg
    '''
  }

  stage('Deploy on Spark') {
    sh 'scripts/submit.sh'
  }

  sleep 120

  stage('Run integration tests') {
    sh '''
      . venv/bin/activate

      pip install setuptools nose2
      python setup.py install
      ENDPOINT=$(scripts/endpoint.sh) nose2 --plugin nose2.plugins.junitxml --junit-xml
    '''

    junit '**/nose2-junit.xml'
  }
}
