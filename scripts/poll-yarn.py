"""
Script to poll job status in YARN
"""

import logging
import subprocess
import sys
from typing import List
import time
import re

_log = logging.getLogger('poll-yarn')


def main(argv: List[str]):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    # Simple interface for now
    cmd, job_name = argv
    if cmd == 'wait-for-webapp':
        app = App.from_job_name(job_name=job_name, states=["SUBMITTED", "ACCEPTED", "RUNNING"])
        app.wait_for_listening_webapp()
    else:
        # TODO: define more commands?
        raise ValueError(cmd)


def run_command(command: List[str], timeout=60) -> subprocess.CompletedProcess:
    _log.info("Running {c!r}".format(c=command))
    return subprocess.run(command, stdout=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)


class Yarn:
    """Simple wrapper around YARN cli interface"""

    @staticmethod
    def list_applications(states: List[str] = None) -> List[List[str]]:
        states = states or ['RUNNING']
        command = ['yarn', 'application', '-list', '-appStates', ','.join(s.upper() for s in states)]
        p = run_command(command)
        app_lines = [line.split() for line in p.stdout.split('\n') if line.startswith('application_')]
        _log.info("Found {c} apps".format(c=len(app_lines)))
        # TODO: do a bit of parsing?
        return app_lines

    @staticmethod
    def get_application_status(app_id: str) -> str:
        stdout = run_command(['yarn', 'application', '-status', app_id]).stdout
        return re.search(r'\s+State\s*:\s*(\w+)', stdout).group(1)

    @staticmethod
    def read_application_logs(app_id: str) -> str:
        command = ['yarn', 'logs', '-applicationId', app_id, '-log_files', 'stdout']
        _log.info("Running {c!r}".format(c=command))
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                yield line
        assert p.returncode == 0


class App:
    """Inspect OpenEO webapp state."""

    def __init__(self, app_id: str, yarn: Yarn):
        self.app_id = app_id
        self.yarn = yarn

    @classmethod
    def from_job_name(cls, job_name: str, states=List[str]) -> 'App':
        yarn = Yarn()
        _log.info("Looking up app id for job name {j!r}".format(j=job_name))
        apps = [line for line in yarn.list_applications(states=states) if line[1] == job_name]
        if len(apps) == 0:
            raise Exception("No app found with name {j!r}".format(j=job_name))
        if len(apps) > 1:
            raise Exception("Multiple apps found with name {j!r}: {a}".format(j=job_name, a=apps))
        _log.info("Found this app: {a!r}".format(a=apps[0]))
        return cls(app_id=apps[0][0], yarn=yarn)

    def wait_for_listening_webapp(self, timeout=30 * 60, sleep=20) -> str:
        start = time.time()

        _log.info("Waiting for app {a!r} to reach state RUNNING".format(a=self.app_id))
        while True:
            status = self.yarn.get_application_status(app_id=self.app_id)
            if status == "RUNNING":
                _log.info("Reached status RUNNING (elapsed: {e}s)".format(e=time.time() - start))
                break
            elif status not in ["SUBMITTED", "ACCEPTED"]:
                raise ValueError(status)
            elif time.time() > start + timeout:
                raise TimeoutError("timeout")
            _log.info("Not RUNNING yet, instead: {s!r}. Will sleep {p}s.".format(s=status, p=sleep))
            time.sleep(sleep)

        _log.info("Waiting for webapp to start listening for connections")
        while True:
            try:
                url = self.get_webapp_url()
                _log.info("Found webapp url {u!r} (elapsed: {e}s)".format(u=url, e=time.time() - start))
                return url
            except ValueError:
                _log.info("No webapp url found yet")
            if time.time() > start + timeout:
                raise TimeoutError("timeout")
            _log.info("Will sleep {p}s.".format(p=sleep))
            time.sleep(sleep)

    def get_webapp_url(self) -> str:
        """Get 'listening' url of openeo web app from logs"""
        listen_logs = [line for line in self.yarn.read_application_logs(self.app_id) if 'Listening at' in line]
        if len(listen_logs) != 1:
            raise ValueError("Expected 1 'listening at' log line but found {l}".format(l=listen_logs))
        return re.search(r"Listening at: (\S+)", listen_logs[0]).group(1)


if __name__ == '__main__':
    main(sys.argv[1:])
