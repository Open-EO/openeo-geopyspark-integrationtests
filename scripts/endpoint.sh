#!/bin/bash

jobName=$1

appId=$(yarn application -list 2>&1 | grep ${jobName} | awk '{print $1}')

endpointLine=$(yarn logs -applicationId ${appId} -log_files stdout | grep 'Listening at')
endpointGroup="Listening at: (.*) \("

if [[ ${endpointLine} =~ ${endpointGroup} ]]; then
    echo ${BASH_REMATCH[1]}
else
    >&2 echo "could not determine endpoint from Spark job name ${jobName}"
    exit 1
fi
