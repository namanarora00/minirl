"""Run untrusted, model-written code in Docker containers.

A pool of long-lived containers, each locked down: no network, read-only
root filesystem, a small writable tmpfs at /w, capped memory, CPU and process
count. Each run gets a fresh directory under /w, the files arrive as a tar on
stdin, the command runs under a timeout, and the directory is deleted after.
One `docker exec` per run (~0.1 s) instead of one `docker run` (~0.5 s).

Known weakness, kept on purpose because it is the interesting part: runs share
a container, so code that escapes its directory (fills /w, leaves a process
behind) can disturb later runs. Real RL environment stacks pay for a fresh
sandbox per episode; making that cheap is exactly what VM fork/reset is for.
"""

import io
import queue
import subprocess
import tarfile
import uuid

IMAGE = "python:3.12-slim"


class DockerPool:
    def __init__(self, n=8, image=IMAGE):
        self.free = queue.Queue()
        self.names = []
        for _ in range(n):
            name = f"minirl-sbx-{uuid.uuid4().hex[:8]}"
            subprocess.run(
                ["docker", "run", "-d", "--rm", "--name", name, "--network", "none", "--read-only",
                 "--tmpfs", "/w:rw,exec,size=64m", "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
                 image, "sleep", "infinity"],
                check=True, capture_output=True,
            )
            self.names.append(name)
            self.free.put(name)

    def run(self, files: dict, cmd: str, timeout=10):
        """Write `files` into a fresh directory, run `cmd` there. Returns (exit_code, output)."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for path, text in files.items():
                data = text.encode()
                info = tarfile.TarInfo(path)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        d = f"/w/{uuid.uuid4().hex[:12]}"
        script = f"mkdir -p {d} && tar -x -C {d} && cd {d} && timeout {timeout} {cmd} 2>&1; code=$?; cd / && rm -rf {d}; exit $code"
        name = self.free.get()
        try:
            p = subprocess.run(["docker", "exec", "-i", name, "sh", "-c", script], input=buf.getvalue(), capture_output=True, timeout=timeout + 15)
        finally:
            self.free.put(name)
        out = p.stdout.decode(errors="replace")
        if p.returncode == 124:
            out += f"\n[timed out after {timeout}s]"
        return p.returncode, out

    def close(self):
        if self.names:
            subprocess.run(["docker", "rm", "-f", *self.names], capture_output=True)
            self.names = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
