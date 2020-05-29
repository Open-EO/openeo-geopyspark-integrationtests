"""
Script to poll job status in YARN
"""

import logging
import re
import subprocess
import sys
import time
from typing import List, Iterator

_log = logging.getLogger('poll-yarn')


def main(argv: List[str]):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    # Simple interface for now
    _log.info("argv: {a!r}".format(a=argv))
    cmd, job_name = argv[1:]
    if cmd == 'get-app-id':
        app = App.from_job_name(job_name=job_name, states=["SUBMITTED", "ACCEPTED", "RUNNING"])
        print(app.app_id)
    elif cmd == 'wait-for-webapp':
        app = App.from_job_name(job_name=job_name, states=["SUBMITTED", "ACCEPTED", "RUNNING"])
        app.wait_for_listening_webapp()
    elif cmd == 'get-webapp-url':
        app = App.from_job_name(job_name=job_name, states=["RUNNING"])
        print(app.get_webapp_url())
    elif cmd == 'kill-when-running':
        try:
            App.from_job_name(job_name=job_name, states=["RUNNING"]).kill()
        except AppNotFoundException as e:
            _log.info("No running app {n!}: {e!}".format(n=job_name, e=e))
    else:
        raise ValueError(cmd)


def run_command(command: List[str], timeout=60) -> subprocess.CompletedProcess:
    _log.info("Running {c!r}".format(c=command))
    return subprocess.run(command, stdout=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)


class Yarn:
    """Simple wrapper around YARN cli interface"""

    @staticmethod
    def list_applications(states: List[str] = None) -> List[List[str]]:
        """List all YARN applications"""
        states = states or ['RUNNING']
        command = ['yarn', 'application', '-list', '-appStates', ','.join(s.upper() for s in states)]
        p = run_command(command)
        app_lines = [line.split() for line in p.stdout.split('\n') if line.startswith('application_')]
        _log.info("Found {c} apps with states {s!r}".format(c=len(app_lines), s=states))
        # TODO: do a bit of parsing?
        return app_lines

    @staticmethod
    def get_application_status(app_id: str) -> str:
        """Get YARN app status"""
        stdout = run_command(['yarn', 'application', '-status', app_id]).stdout
        return re.search(r'\s+State\s*:\s*(\w+)', stdout).group(1)

    @staticmethod
    def read_application_logs(app_id: str) -> Iterator[str]:
        """Stream the application logs"""
        command = ['yarn', 'logs', '-applicationId', app_id, '-log_files', 'stdout']
        _log.info("Running {c!r}".format(c=command))
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                yield line
        assert p.returncode == 0


class AppNotFoundException(Exception):
    pass


class App:
    """Inspect OpenEO webapp state."""

    def __init__(self, app_id: str, yarn: Yarn):
        self.app_id = app_id
        self.yarn = yarn

    @classmethod
    def from_job_name(cls, job_name: str, states=List[str]) -> 'App':
        """Get YARN app listing and get app id corresponding with given job name."""
        yarn = Yarn()
        _log.info("Looking up app id for job name {j!r}".format(j=job_name))
        apps = [line for line in yarn.list_applications(states=states) if line[1] == job_name]
        if len(apps) == 0:
            raise AppNotFoundException("No app found with name {j!r} and state in {s!r}".format(j=job_name, s=states))
        if len(apps) > 1:
            raise Exception("Multiple apps found with name {j!r}: {a!r}".format(j=job_name, a=apps))
        _log.info("Found this app: {a!r}".format(a=apps[0]))
        return cls(app_id=apps[0][0], yarn=yarn)

    def wait_for_listening_webapp(self, timeout=30 * 60, sleep=20) -> str:
        start = time.time()

        def elapsed():
            return time.time() - start

        _log.info("Waiting for app {a!r} to reach state RUNNING".format(a=self.app_id))
        while True:
            status = self.yarn.get_application_status(app_id=self.app_id)
            if status == "RUNNING":
                _log.info("App {a!r} reached status RUNNING (elapsed: {e:.1f}s)".format(a=self.app_id, e=elapsed()))
                break
            elif status not in ["SUBMITTED", "ACCEPTED"]:
                raise ValueError(status)
            elif elapsed() > timeout:
                raise TimeoutError("timeout")
            _log.info("App {a!r} not RUNNING yet after {e:.1f}s, instead: {s!r}. Will sleep {p}s.".format(
                a=self.app_id, s=status, e=elapsed(), p=sleep)
            )
            time.sleep(sleep)

        _log.info("Waiting for webapp to start listening for connections")
        while True:
            try:
                url = self.get_webapp_url()
                _log.info("Found webapp url {u!r} (elapsed: {e:.1f}s)".format(u=url, e=elapsed()))
                return url
            except ValueError:
                _log.info("No webapp url found yet after {e:.1f}s".format(e=elapsed()))
            if elapsed() > timeout:
                raise TimeoutError("timeout")
            _log.info("Will sleep {p}s.".format(p=sleep))
            time.sleep(sleep)

    def get_webapp_url(self) -> str:
        """Get 'listening' url of openeo web app from logs"""
        listen_logs = [line for line in self.yarn.read_application_logs(self.app_id) if 'Listening at' in line]
        if len(listen_logs) != 1:
            raise ValueError("Expected 1 'Listening at' log line in logs of {a!r} but found {l!r}".format(
                a=self.app_id, l=listen_logs)
            )
        url = re.search(r"Listening at: (\S+)", listen_logs[0]).group(1)
        _log.info("Found webapp url in logs of app {a!r}: {u!r}".format(a=self.app_id, u=url))
        return url

    def kill(self):
        stdout = run_command(['yarn', 'application', '-kill', self.app_id]).stdout
        _log.info("Kill stdout: {s!r}".format(s=stdout))


if __name__ == '__main__':
    main(sys.argv)
