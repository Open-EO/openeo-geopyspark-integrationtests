#!/usr/bin/env groovy

node("jenkinsslave1.vgt.vito.be") {
  properties([disableConcurrentBuilds()])

  jobName = "OpenEO-GeoPySpark-${env.BRANCH_NAME}"

  deleteDir()
  checkout scm

  stage('Build and test') {
    sh '''
      git clone https://github.com/Open-EO/openeo-python-client.git
      git clone https://github.com/Open-EO/openeo-python-driver.git
      git clone https://github.com/Open-EO/openeo-geopyspark-driver.git
    '''

    try {
      sh '''
        python3 -m venv venv
        . venv/bin/activate

        pip install wheel pytest pytest-timeout

        cd openeo-python-client
        pip install travis-sphinx==2.1.0 "sphinx<1.7"
        pip install -r requirements-dev.txt
        pip install -r requirements.txt
        pytest --junit-xml=pytest-junit.xml
        python setup.py install bdist_egg

        cd ../openeo-python-driver
        pip install -r requirements-dev.txt
        pip install -r requirements.txt
        pytest --junit-xml=pytest-junit.xml
        python setup.py install bdist_egg

        cd ../openeo-geopyspark-driver
        pip install $(cat requirements.txt | tr '\\n' ' ' | sed -e 's/openeo-api==0.0.1/openeo-api/') --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-packages-public/simple
        SPARK_HOME=$(find_spark_home.py) TRAVIS=1 pytest --junit-xml=pytest-junit.xml
        python setup.py install bdist_egg
      '''
    } finally {
      junit '**/pytest-junit.xml'
    }
  }

  stage('Deploy on Spark') {
    withMavenEnv() {
      sh "mvn dependency:copy -Dartifact=be.vito.eodata:GeoPySparkExtensions:2.7.0-SNAPSHOT -DoutputDirectory=."
    }

    sh "scripts/submit.sh ${jobName}"
  }

  sleep 120

  stage('Run integration tests') {
    try {
      sh """
        . venv/bin/activate
        python setup.py install
        ENDPOINT=\$(scripts/endpoint.sh ${jobName}) pytest tests --timeout 120 --junit-xml=pytest-junit.xml
      """
    } finally {
      junit '**/pytest-junit.xml'
    }
  }
}

void withMavenEnv(List envVars = [], def body) {
    String mvntool = tool name: "Maven 3.5.0", type: 'hudson.tasks.Maven$MavenInstallation'
    String jdktool = tool name: "OpenJDK 8 Centos7", type: 'hudson.model.JDK'

    List mvnEnv = ["PATH+MVN=${mvntool}/bin", "PATH+JDK=${jdktool}/bin", "JAVA_HOME=${jdktool}", "MAVEN_HOME=${mvntool}"]

    mvnEnv.addAll(envVars)
    withEnv(mvnEnv) {
        body.call()
    }
}
