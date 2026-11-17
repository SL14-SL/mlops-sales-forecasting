"""Open the token-protected AWS MLflow UI through a loopback-only proxy."""

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
import httpx


def local_request_allowed(
    *, host: str, origin: str | None, fetch_site: str | None, port: int
) -> bool:
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{value}" for value in hosts}
    return host in hosts and (origin is None or origin in origins) and fetch_site != "cross-site"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    raw = subprocess.check_output(
        [
            "terraform",
            "-chdir=infrastructure/terraform-aws",
            "output",
            "-json",
        ]
    )
    outputs = {key: value["value"] for key, value in json.loads(raw).items()}
    endpoint = outputs["mlflow_tracking_uri"].rstrip("/")
    token = boto3.client("secretsmanager").get_secret_value(
        SecretId=outputs["mlflow_token_secret_arn"],
    )["SecretString"]
    ignored_headers = {
        "host",
        "authorization",
        "connection",
        "content-length",
        "transfer-encoding",
        "accept-encoding",
        "origin",
        "referer",
    }
    client = httpx.Client(timeout=60, follow_redirects=False)

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not log query strings or credential-bearing headers.

        def forward(self):
            if (
                not local_request_allowed(
                    host=self.headers.get("Host", ""),
                    origin=self.headers.get("Origin"),
                    fetch_site=self.headers.get("Sec-Fetch-Site"),
                    port=args.port,
                )
                or not self.path.startswith("/")
                or self.path.startswith("//")
            ):
                self.send_error(403)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in ignored_headers
            }
            headers["Authorization"] = f"Bearer {token}"
            try:
                response = client.request(
                    self.command, endpoint + self.path, headers=headers, content=body
                )
            except httpx.HTTPError:
                self.send_error(502, "MLflow endpoint could not be reached")
                return
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() in {
                    "connection",
                    "content-length",
                    "transfer-encoding",
                    "content-encoding",
                }:
                    continue
                if key.lower() == "location" and value.startswith(endpoint + "/"):
                    value = value[len(endpoint) :]
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.content)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    print(f"MLflow UI: http://127.0.0.1:{args.port} (Ctrl+C to close)")
    with ThreadingHTTPServer(("127.0.0.1", args.port), Proxy) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            client.close()


if __name__ == "__main__":
    main()
