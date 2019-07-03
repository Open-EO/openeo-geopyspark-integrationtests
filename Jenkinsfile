#!/usr/bin/env groovy

@Library('lib')_

def config = [:]
def docker_registry = config.docker_registry ?: 'vito-docker-private.artifactory.vgt.vito.be'
def python_version = config.python_version ?: '3.5'
def run_tests = (config.run_tests == false) ? config.run_tests : true
def extra_container_volumes = config.extra_container_volumes ?: ''
def extra_env_variables = config.extra_env_variables ?: ''
def pre_test_script = config.pre_test_script ?: ''

pipeline {
    // Run job on any node with this label
    agent {
      label "devdmz"
    }
    // Set built-in environment variables
    environment {
      BRANCH_NAME  = "${env.BRANCH_NAME}"
      BUILD_NUMBER = "${env.BUILD_NUMBER}"
      BUILD_URL    = "${env.BUILD_URL}"
      JOB_NAME     = "${env.JOB_NAME}"
      JOB_URL      = "${env.JOB_URL}"
      WORKSPACE    = "${env.WORKSPACE}"
    }
    // Placeholder to be able to pass an email address to the job
    parameters {
      string(name: 'mail_address', defaultValue: 'Dummy')
    }
    // Disable default checkout to have more control over checkout step
    options {
      skipDefaultCheckout(true)
    }
    // Start of the pipeline
    stages {
      // Checkout the project code
      stage('Checkout') {
        steps {
          checkOut(false)
        }
      }
      // Prepare the virtual environment where the package will be built and tested
      stage('Prepare virtualenv') {
        steps {
          prepareVenv(docker_registry, python_version)
        }
      }
      stage('Package & Publish virtualenv'){
        steps{
            cd venv35
            sh 'zip -r ../venv.zip *'
            cd ..
        }
      }
      // Run the tests
      stage('Execute Tests') {
        when {
          expression {
            run_tests == true
          }
        }
        steps {
          executePythonTests(docker_registry, python_version, 'tests', true, extra_container_volumes, extra_env_variables, pre_test_script)
        }
      }
    }
  }


node("jenkinsslave1.vgt.vito.be") {
  properties([disableConcurrentBuilds()])

  jobName = "OpenEO-GeoPySpark-${env.BRANCH_NAME}"

  deleteDir()
  checkout scm

  stage('Build and test') {
    sh '''
      #git clone https://github.com/Open-EO/openeo-python-client.git
      #git clone https://github.com/Open-EO/openeo-python-driver.git
      git clone https://github.com/Open-EO/openeo-geopyspark-driver.git
    '''

    try {
      withMavenEnv() {
        sh '''
          export LD_LIBRARY_PATH=/opt/rh/rh-python35/root/usr/lib64:${LD_LIBRARY_PATH}

          python3.5 -m venv venv
          source venv/bin/activate

          pip install --upgrade --force-reinstall pip
          pip download typing==3.6.6
          pip download Fiona==1.7.13 && pip install Fiona-1.7.13-cp35-cp35m-manylinux1_x86_64.whl
          pip install wheel pytest pytest-timeout

          pip install --upgrade --force-reinstall openeo_api openeo_driver

          cd openeo-geopyspark-driver
          pip install $(cat requirements.txt | tr '\\n' ' ' | sed -e 's/openeo-api==0.0.1/openeo-api/') --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-openeo/simple
          SPARK_HOME=$(find_spark_home.py) geopyspark install-jar
          mkdir -p jars && mvn dependency:copy -Dartifact=org.openeo:geotrellis-extensions:1.1.0-SNAPSHOT -DoutputDirectory=jars
          SPARK_HOME=$(find_spark_home.py) TRAVIS=1 pytest --junit-xml=pytest-junit.xml
          python setup.py install bdist_egg
        '''
      }
    } finally {
      junit '**/pytest-junit.xml'
    }
  }

  stage('Deploy on Spark') {
    sh "scripts/submit.sh ${jobName}"
  }

  sleep 180

  stage('Run integration tests') {
    try {
      sh """
        export LD_LIBRARY_PATH=/opt/rh/rh-python35/root/usr/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}

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
