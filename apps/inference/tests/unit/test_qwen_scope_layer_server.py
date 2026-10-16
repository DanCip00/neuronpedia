"""Exercise managed-server discovery and confirmation without loading a model."""

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from unittest import TestCase

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/qwen_scope_layer_server.sh"
FAKE_SERVER = """import socket
import sys
import signal
port = int(sys.argv[sys.argv.index('--port') + 1])
sock = socket.socket()
sock.bind(('127.0.0.1', port))
sock.listen()
print('ready', flush=True)
signal.pause()
"""


class TestQwenScopeLayerServer(TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="qwen-launcher-test-")
        self.root = Path(self.temp.name)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.script = self.root / "qwen_scope_layer_server.sh"
        self.script.write_text(
            SCRIPT.read_text().replace(
                'DEFAULT_LOG_DIR="/tmp/neuronpedia-qwen-scope"', f'DEFAULT_LOG_DIR="{self.log_dir}"'
            )
        )
        self.script.chmod(0o755)
        self.server_file = self.root / "start.py"
        self.server_file.write_text(FAKE_SERVER)
        self.processes: list[subprocess.Popen[str]] = []
        self.reapers: list[threading.Thread] = []
        self.managed_pidfiles: list[Path] = []

    def tearDown(self) -> None:
        for path in self.managed_pidfiles:
            if path.exists():
                self.call("--pidfile", str(path), "--yes", "--timeout-seconds", "3")
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for reaper in self.reapers:
            reaper.join(timeout=3)
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            if process.stdout is not None:
                process.stdout.close()
        self.temp.cleanup()

    def launch(self, layer: int = 52) -> tuple[subprocess.Popen[str], Path]:
        import socket

        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.server_file),
                "--model_id",
                "Qwen/Qwen3.5-27B",
                "--include_sae",
                f"^layer{layer}$",
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.processes.append(process)
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(), "ready")
        reaper = threading.Thread(target=process.wait, daemon=True)
        reaper.start()
        self.reapers.append(reaper)
        path = self.log_dir / f"layer{layer}.port{port}.pid"
        path.write_text(
            f"PID={process.pid}\nPGID={process.pid}\nLAYER=layer{layer}\n"
            f"PORT={port}\nHOST=127.0.0.1\nCUDA_DEVICES=test\n"
        )
        return process, path

    def call(self, *args: str, reply: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script), "stop", *args],
            input=reply,
            text=True,
            capture_output=True,
            timeout=12,
            check=False,
        )

    def test_bare_stop_discovers_and_waits_for_confirmation(self) -> None:
        process, path = self.launch()
        result = self.call(reply="n\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "cancelled")
        self.assertIn("Stop this server? [y/N]", result.stderr)
        self.assertIn("layer52", result.stderr)
        self.assertIsNone(process.poll())
        self.assertTrue(path.exists())

    def test_eof_cancels_without_signalling(self) -> None:
        process, _ = self.launch()
        result = self.call()
        self.assertEqual(json.loads(result.stdout)["status"], "cancelled")
        self.assertIsNone(process.poll())

    def test_confirmed_discovery_stops_server_and_removes_pidfile(self) -> None:
        process, path = self.launch()
        result = self.call(reply="yes\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "stopped")
        self.assertIsNotNone(process.poll())
        self.assertFalse(path.exists())

    def test_explicit_pidfile_still_prompts_and_yes_supports_automation(self) -> None:
        process, path = self.launch()
        result = self.call("--pidfile", str(path), reply="no\n")
        self.assertEqual(json.loads(result.stdout)["status"], "cancelled")
        self.assertIsNone(process.poll())
        result = self.call("--pidfile", str(path), "--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "stopped")
        self.assertNotIn("Stop this server?", result.stderr)

    def test_multiple_servers_require_selection_even_with_yes(self) -> None:
        first, _ = self.launch(31)
        second, second_path = self.launch(52)
        refused = self.call("--yes")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("multiple servers", refused.stderr)
        self.assertIsNone(first.poll())
        self.assertIsNone(second.poll())
        selected = self.call(reply="2\ny\n")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertIsNone(first.poll())
        self.assertIsNotNone(second.poll())
        self.assertFalse(second_path.exists())

    def test_layer_filter_selects_only_the_requested_server(self) -> None:
        first, _ = self.launch(31)
        second, _ = self.launch(52)
        result = self.call("--layer", "52", "--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(first.poll())
        self.assertIsNotNone(second.poll())

    def test_stale_pidfile_does_not_target_an_unrelated_process(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], text=True, start_new_session=True
        )
        self.processes.append(process)
        reaper = threading.Thread(target=process.wait, daemon=True)
        reaper.start()
        self.reapers.append(reaper)
        path = self.log_dir / "layer52.port5002.pid"
        path.write_text(f"PID={process.pid}\nPGID={process.pid}\nLAYER=layer52\nPORT=5002\n")
        result = self.call("--yes")
        self.assertEqual(json.loads(result.stdout)["status"], "not_running")
        explicit = self.call("--pidfile", str(path), "--yes")
        self.assertNotEqual(explicit.returncode, 0)
        self.assertIn("refusing to signal", explicit.stderr)
        self.assertIsNone(process.poll())

    def test_recorded_process_start_must_match(self) -> None:
        process, path = self.launch()
        path.write_text(path.read_text() + "PID_START_TICKS=1\n")
        result = self.call("--yes")
        self.assertEqual(json.loads(result.stdout)["status"], "not_running")
        self.assertIsNone(process.poll())

    def test_empty_discovery_is_a_noop(self) -> None:
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "not_running")

    def fake_start_environment(self) -> tuple[Path, dict[str, str]]:
        """Use tiny HTTP servers instead of GPU/model processes."""
        app = self.root / "repo/apps/inference"
        app.mkdir(parents=True)
        app.joinpath("start.py").write_text("""import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

def setting(flag, env):
    return os.environ.get(env, sys.argv[sys.argv.index(flag) + 1])

port = int(setting('--port', 'SERVER_PORT'))
if str(port) == os.environ.get('FAKE_FAIL_PORT'):
    sys.exit(1)
identity = {
    'port': port,
    'cuda': os.environ['CUDA_VISIBLE_DEVICES'],
    'model': setting('--model_id', 'MODEL_ID'),
    'source': setting('--include_sae', 'INCLUDE_SAE'),
}
Path(f'identity-{port}.json').write_text(json.dumps(identity))
expected = os.environ.get('FAKE_EXPECT_PORTS', '')
deadline = time.monotonic() + 4
while expected and not all(Path(f'identity-{p}.json').exists() for p in expected.split(',')):
    if time.monotonic() > deadline:
        raise RuntimeError('the other replica was not started concurrently')
    time.sleep(0.02)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        data = json.dumps({'status': 'healthy'}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"results": [{}]}')

HTTPServer(('127.0.0.1', port), Handler).serve_forever()
""")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(f'#!/bin/sh\nshift 2\nexec {shlex.quote(sys.executable)} "$@"\n')
        uv.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SERVER_PORT": "1",
            "MODEL_ID": "wrong",
            "INCLUDE_SAE": "wrong",
        }
        return app, env

    def start_pool(self, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
        """Record every pidfile for cleanup, including after a failed start."""
        try:
            return subprocess.run(
                [
                    "bash",
                    str(self.script),
                    "start",
                    "--repo",
                    str(self.root / "repo"),
                    "--layer",
                    "52",
                    "--timeout-seconds",
                    "10",
                    *args,
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        finally:
            self.managed_pidfiles.extend(self.log_dir.glob("*.pid"))

    def free_ports(self) -> tuple[str, str]:
        """Reserve two different ephemeral ports, then release them together."""
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            second.bind(("127.0.0.1", 0))
            return str(first.getsockname()[1]), str(second.getsockname()[1])

    def test_independent_replicas_start_and_stop_through_the_launcher(self) -> None:
        """CLI settings win over inherited settings; stopping one leaves the other alive."""
        app, env = self.fake_start_environment()
        replicas: list[dict[str, object]] = []
        for gpu in (2, 3):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            result = subprocess.run(
                [
                    "bash",
                    str(self.script),
                    "start",
                    "--repo",
                    str(self.root / "repo"),
                    "--layer",
                    "52",
                    "--port",
                    str(port),
                    "--cuda-devices",
                    str(gpu),
                    "--timeout-seconds",
                    "10",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            info = json.loads(result.stdout)
            self.managed_pidfiles.append(Path(info["pidfile"]))
            replicas.append(info)
            self.assertTrue(info["probed"])
            identity = json.loads(app.joinpath(f"identity-{port}.json").read_text())
            self.assertEqual(
                identity, {"port": port, "cuda": str(gpu), "model": "Qwen/Qwen3.5-27B", "source": "^layer52$"}
            )
        self.assertNotEqual(replicas[0]["pgid"], replicas[1]["pgid"])
        self.assertNotEqual(replicas[0]["log"], replicas[1]["log"])
        stopped = self.call("--pidfile", str(replicas[0]["pidfile"]), reply="y\n")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertFalse(Path(str(replicas[0]["pidfile"])).exists())
        status = subprocess.run(
            ["bash", str(self.script), "status", "--pidfile", str(replicas[1]["pidfile"])],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["status"], "running")

    def test_pool_pairs_devices_and_ports_and_starts_concurrently(self) -> None:
        app, env = self.fake_start_environment()
        first, second = self.free_ports()
        env["FAKE_EXPECT_PORTS"] = f"{first},{second}"
        result = self.start_pool(
            env,
            "--ports",
            first,
            second,
            "--cuda-devices",
            "2",
            "3",
            "--max-concurrent-requests",
            "48",
            "--extra-arg",
            "--fake-option",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        replicas = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(replicas), 2)
        self.assertEqual(
            {info["base_url"] for info in replicas},
            {
                f"http://127.0.0.1:{first}",
                f"http://127.0.0.1:{second}",
            },
        )
        self.assertEqual(len({info["pgid"] for info in replicas}), 2)
        self.assertEqual(len({info["pidfile"] for info in replicas}), 2)
        self.assertEqual(len({info["log"] for info in replicas}), 2)
        for port, gpu in ((first, "2"), (second, "3")):
            identity = json.loads((app / f"identity-{port}.json").read_text())
            self.assertEqual(identity["cuda"], gpu)
            self.assertEqual(identity["source"], "^layer52$")
        for info in replicas:
            self.assertTrue(info["probed"])
            self.assertEqual(info["max_concurrent_requests"], 48)
        stopped = self.call("--port", first, "--yes")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        status = subprocess.run(
            ["bash", str(self.script), "status", "--pidfile", str(self.log_dir / f"layer52.port{second}.pid")],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(json.loads(status.stdout)["status"], "running")

    def test_invalid_pools_do_not_start_any_server(self) -> None:
        app, env = self.fake_start_environment()
        cases = [
            (["--ports", "5002", "5003", "--cuda-devices", "2"], "same number"),
            (["--ports", "5002", "5003"], "same number"),
            (["--ports", "5002", "5002", "--cuda-devices", "2", "3"], "duplicate port"),
            (["--ports", "5002", "5003", "--cuda-devices", "2", "2"], "duplicate CUDA device"),
            (["--ports", "5002", "--cuda-devices", "auto"], "explicit GPU"),
            (["--port", "5002", "--ports", "5003", "--cuda-devices", "2"], "cannot be combined"),
            (["--ports", "5002", "65536", "--cuda-devices", "2", "3"], "1..65535"),
            (["--cuda-devices", "2", "3"], "multiple devices require --ports"),
            (["--ports"], "requires at least one port"),
        ]
        for args, error in cases:
            with self.subTest(args=args):
                result = self.start_pool(env, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse(list(app.glob("identity-*.json")))
                self.assertFalse(list(self.log_dir.iterdir()))

    def test_occupied_port_is_checked_before_starting_other_replicas(self) -> None:
        app, env = self.fake_start_environment()
        first, _ = self.free_ports()
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            result = self.start_pool(
                env,
                "--ports",
                first,
                str(occupied.getsockname()[1]),
                "--cuda-devices",
                "2",
                "3",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no replicas started", result.stderr)
        self.assertFalse(list(app.glob("identity-*.json")))

    def test_failed_replica_returns_failure_and_leaves_successful_copy_independent(self) -> None:
        app, env = self.fake_start_environment()
        first, second = self.free_ports()
        env["FAKE_FAIL_PORT"] = first
        result = self.start_pool(env, "--ports", first, second, "--cuda-devices", "2", "3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"replica on port {first}", result.stderr)
        replicas = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(replicas), 1)
        self.assertEqual(replicas[0]["base_url"], f"http://127.0.0.1:{second}")
        self.assertTrue((app / f"identity-{second}.json").exists())
