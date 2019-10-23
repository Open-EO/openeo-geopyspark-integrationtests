#!/usr/bin/env groovy

@Library('lib')_

def config = [:]
def docker_registry = config.docker_registry ?: 'vito-docker-private.artifactory.vgt.vito.be'
def python_version = config.python_version ?: '3.6'
def run_tests = (config.run_tests == false) ? config.run_tests : true
def extra_container_volumes = config.extra_container_volumes ?: ''
def extra_env_variables = config.extra_env_variables ?: ''
def pre_test_script = config.pre_test_script ?: ''
def pytest_results = 'pytest/pytest_results.xml'


jobName = "OpenEO-GeoPySpark-${env.BRANCH_NAME}"
appId = ""

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
      PYTEST_RESULTS= "${pytest_results}"
    }
    // Placeholder to be able to pass an email address to the job
    parameters {
      string(name: 'mail_address', defaultValue: 'Dummy')
    }
    // Disable default checkout to have more control over checkout step
    options {
      disableConcurrentBuilds()
      skipDefaultCheckout(true)
    }
    // Start of the pipeline
    stages {
      // Checkout the project code
      stage('Checkout') {
        steps {
        script {
            git.checkoutDefault(true)
          }
        }
      }
      stage('Sleep while artifactory refreshes'){
        steps{
            sleep 60
        }
      }
      // Prepare the virtual environment where the package will be built and tested
      stage('Prepare virtualenv') {
        steps {
          script{
            python.createVenv(docker_registry, python_version)
          }
        }
      }
      stage('Package & Publish virtualenv'){
        steps {
              sh 'cd venv36 && zip -r ../venv36.zip *'
              artifactory.uploadSpec("""
                  {
                     "files": [
                       {
                         "pattern": "venv36.zip",
                         "target": "auxdata-public/openeo/",
                         "regexp": "true"
                       }
                     ]
                  }
                """.stripIndent(), null)
        }
      }
      stage('Deploy on Spark') {
        steps{
            sh "scripts/submit.sh ${jobName}"
            script{
              appList = sh( returnStdout:true, script: "yarn application -list -appStates RUNNING,ACCEPTED 2>&1 | grep ${jobName}  || true")
              echo appList
              appId = appList.split("\n").collect { it.split()[0]}[0]

            }
            echo "Spark Job started: ${appId}"
        }
      }
      stage('Wait for Spark job'){
        steps{
            sleep 180
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
          script{
            endpoint = sh(returnStdout: true, script: "scripts/endpoint.sh ${jobName}").trim()
            echo "ENDPOINT=${endpoint}"
            python.test(docker_registry, python_version, 'tests', true, extra_container_volumes, ["ENDPOINT=${endpoint}"], pre_test_script)
          }

        }
      }
    }
    post {
      // Record the test results in Jenkins
      always {
        script{
            if( appId != "" ) {
                echo "Killing running Spark application: ${appId}"
                sh "yarn application -kill ${appId} || true"
            }
            python.recordTestResults(run_tests)
        }
      }
      // Send a mail notification on failure
      failure {
       script {
          notification.fail(mail_address)
        }
      }
    }
  }

