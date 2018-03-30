#!/usr/bin/env groovy

node("jenkinsslave1.vgt.vito.be") {
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
      python3 -m virtualenv venv
      . venv/bin/activate

      pip install pytest

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
      export PYTHONPATH=$VIRTUAL_ENV/lib/python3/site-packages
      make build
      pip install -e .

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

  stage('Run integration tests') {
    sh '''
      . venv/bin/activate

      pip install setuptools
      python setup.py install test
    '''
  }
}
