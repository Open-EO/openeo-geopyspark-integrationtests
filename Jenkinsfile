#!/usr/bin/env groovy

@Library('lib')_

def docker_registry = globalDefaults.docker_registry_prod()
def python_version = '3.6'
def pre_test_script = ''
def pylint_results = 'test_results/pylint.out'
def pytest_results = 'test_results/pytest_results.xml'
def jobName = "OpenEO-GeoPySpark-${env.BRANCH_NAME}"
def appId = ""
def deploy = ("${BRANCH_NAME}" == 'master') ? true : false
def pylint = false
def run_tests = true
def test_coverage = false

pipeline {
    // Run job on any node with this label
    agent {
      label "devdmz"
    }
    // Set built-in environment variables
    environment {
      BRANCH_NAME    = "${env.BRANCH_NAME}"
      BUILD_NUMBER   = "${env.BUILD_NUMBER}"
      BUILD_URL      = "${env.BUILD_URL}"
      DATE           = utils.getDate()
      JOB_NAME       = "${env.JOB_NAME}"
      JOB_URL        = "${env.JOB_URL}"
      WORKSPACE      = "${env.WORKSPACE}"
      PYTEST_RESULTS = "${pytest_results}"
      PYLINT         = "${pylint}"
      TEST_COVERAGE  = "${test_coverage}"
    }
    parameters {
      string(name: 'mail_address', defaultValue: 'dummy@vito.be')
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
            python.createVenv(docker_registry, python_version, '')
          }
        }
      }
      stage('Package virtualenv'){
        steps {
          script{
            dir('venv36') {
              sh "zip -r ../openeo-${DATE}-${BUILD_NUMBER}.zip *"
            }
          }
        }
      }
      stage('Upload archive') {
        steps {
          script {
            artifactory.uploadSpec(
            """
              {
                "files": [
                  {
                    "pattern": "openeo(.*).zip",
                    "target": "auxdata-public/openeo/dev/",
                    "regexp": "true"
                  }
                ]
              }
            """.stripIndent(), null)
          }
        }
      }
      stage('Deploy on Spark') {
        steps{
            sh "scripts/submit.sh ${jobName} ${DATE}-${BUILD_NUMBER}"
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
            sleep 900
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
            python.test(docker_registry, python_version, 'tests', true, '', ["ENDPOINT=${endpoint}"], pre_test_script)
          }
        }
      }
      stage('Trigger deploy job') {
        when {
          expression {
            deploy
          }
        }
        steps {
          script {
            utils.triggerJob('geo.openeo_deploy', ['version': "${DATE}-${BUILD_NUMBER}", 'openeo_env': 'dev', 'mail_address': mail_address])
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
            python.recordTestResults()
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
