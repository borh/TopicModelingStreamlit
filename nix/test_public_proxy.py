import http.client
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


class Upstream(BaseHTTPRequestHandler):
    def do_GET(self):
        upgrade = self.headers.get("Upgrade") == "websocket"
        self.send_response(101 if upgrade else 200)
        self.send_header("X-Backend", self.server.backend)
        self.send_header("X-Received-IP", self.headers.get("X-Topic-Client-IP", ""))
        if upgrade:
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
        self.end_headers()

    do_POST = do_GET
    do_DELETE = do_GET

    def log_message(self, *args):
        pass


servers = []
for port, backend in [(18082, "campus"), (18083, "public")]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Upstream)
    server.backend = backend
    Thread(target=server.serve_forever, daemon=True).start()
    servers.append(server)
prefix = "/topic-modeling-gensim/"


def request(method, path, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", 18081, timeout=5)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        status, returned_headers = response.status, dict(response.getheaders())
        if status != 101:
            response.read()
        return status, returned_headers
    finally:
        connection.close()


try:
    for configuration, campus in zip(sys.argv[1:], [False, True]):
        process = subprocess.Popen(
            ["caddy", "run", "--config", configuration, "--adapter", "caddyfile"]
        )
        try:
            for attempt in range(100):
                try:
                    request("GET", prefix)
                    break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError("Caddy failed to start")
                    time.sleep(0.05)
            else:
                raise TimeoutError("Caddy failed to start")
            forged = {"X-Topic-Client-IP": "133.1.2.3", "X-Forwarded-For": "133.1.2.3"}
            status, headers = request("GET", prefix, forged)
            assert status == 200
            assert headers["X-Received-Ip"] == "127.0.0.1", headers
            assert headers["X-Backend"] == ("campus" if campus else "public")
            _, media_headers = request(
                "GET", prefix + "media/private-download.csv", forged
            )
            assert media_headers["X-Backend"] == ("campus" if campus else "public")
            assert headers["Content-Security-Policy"] == "frame-ancestors 'self'"
            assert headers["X-Content-Type-Options"] == "nosniff"
            for method in ["POST", "DELETE"]:
                status, _ = request(
                    method, prefix + "_stcore/upload_file/session/file", forged
                )
                assert status == (200 if campus else 403), (campus, method, status)
            status, headers = request(
                "GET",
                prefix + "_stcore/stream",
                {**forged, "Upgrade": "websocket", "Connection": "Upgrade"},
            )
            assert status == 101
            assert headers["X-Received-Ip"] == "127.0.0.1"
            assert request("GET", prefix.rstrip("/"))[0] == 302
        finally:
            process.terminate()
            process.wait(timeout=10)
finally:
    for server in servers:
        server.shutdown()
        server.server_close()
