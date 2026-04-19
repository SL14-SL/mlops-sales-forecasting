from __future__ import annotations

import argparse
import os
import sys

from src.deployment.gcp import (
    GCPDeploymentError,
    build_and_push_image,
    deploy_cloud_run_service,
    validate_service_exists,
)


def _parse_env_vars(values: list[str] | None) -> dict[str, str]:
    """
    Parse repeated KEY=VALUE CLI arguments into a dict.

    Example:
        --env APP_ENV=prod --env API_KEY=abc123
    """
    env_vars: dict[str, str] = {}

    if not values:
        return env_vars

    for item in values:
        if "=" not in item:
            raise GCPDeploymentError(
                f"Invalid env var '{item}'. Expected format: KEY=VALUE"
            )

        key, value = item.split("=", 1)
        key = key.strip()

        if not key:
            raise GCPDeploymentError(
                f"Invalid env var '{item}'. Key must not be empty."
            )

        env_vars[key] = value

    return env_vars


def _collect_env_vars_from_names(names: list[str] | None) -> dict[str, str]:
    """
    Read selected environment variable names from os.environ.

    Example:
        --env-from APP_ENV --env-from API_KEY
    """
    env_vars: dict[str, str] = {}

    if not names:
        return env_vars

    for name in names:
        value = os.getenv(name)
        if value is None:
            raise GCPDeploymentError(
                f"Environment variable '{name}' is not set in the current environment."
            )
        env_vars[name] = value

    return env_vars


def _merge_env_vars(
    direct_env: dict[str, str],
    inherited_env: dict[str, str],
) -> dict[str, str]:
    merged = dict(inherited_env)
    merged.update(direct_env)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.deployment.cli",
        description="GCP deployment CLI for Cloud Run services.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate that a configured service exists in configs/gcp.yaml.",
    )
    validate_parser.add_argument(
        "service_key",
        help="Configured service key, e.g. prediction_api or mlflow.",
    )

    build_parser_cmd = subparsers.add_parser(
        "build",
        help="Build and push a service image to Artifact Registry.",
    )
    build_parser_cmd.add_argument("service_key")
    build_parser_cmd.add_argument("--tag", required=True)
    build_parser_cmd.add_argument(
        "--project-dir",
        default=".",
        help="Docker build context directory. Defaults to current directory.",
    )

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy a configured service to Cloud Run.",
    )
    deploy_parser.add_argument("service_key")
    deploy_parser.add_argument("--tag", required=True)
    deploy_parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Pass environment variables as KEY=VALUE. Can be used multiple times.",
    )
    deploy_parser.add_argument(
        "--env-from",
        action="append",
        default=[],
        help="Read environment variable by name from the current shell environment. Can be used multiple times.",
    )
    deploy_parser.add_argument(
        "--project-id",
        default=None,
        help="Override GCP project id from config.",
    )
    deploy_parser.add_argument(
        "--region",
        default=None,
        help="Override GCP region from config.",
    )
    deploy_parser.add_argument(
        "--service-account",
        default=None,
        help='Override Cloud Run service account',
    )

    build_deploy_parser = subparsers.add_parser(
        "build-deploy",
        help="Build, push, and deploy a configured service.",
    )
    build_deploy_parser.add_argument("service_key")
    build_deploy_parser.add_argument("--tag", required=True)
    build_deploy_parser.add_argument(
        "--project-dir",
        default=".",
        help="Docker build context directory. Defaults to current directory.",
    )
    build_deploy_parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Pass environment variables as KEY=VALUE. Can be used multiple times.",
    )
    build_deploy_parser.add_argument(
        "--env-from",
        action="append",
        default=[],
        help="Read environment variable by name from the current shell environment. Can be used multiple times.",
    )
    build_deploy_parser.add_argument(
        "--project-id",
        default=None,
        help="Override GCP project id from config.",
    )
    build_deploy_parser.add_argument(
        "--region",
        default=None,
        help="Override GCP region from config.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "validate":
            validate_service_exists(args.service_key)
            print(f"Service '{args.service_key}' is valid.")
            return 0

        if args.command == "build":
            validate_service_exists(args.service_key)
            image_uri = build_and_push_image(
                service_key=args.service_key,
                image_tag=args.tag,
                project_dir=args.project_dir,
            )
            print(f"Built and pushed image: {image_uri}")
            return 0

        if args.command == "deploy":
            validate_service_exists(args.service_key)

            direct_env = _parse_env_vars(args.env)
            inherited_env = _collect_env_vars_from_names(args.env_from)
            env_vars = _merge_env_vars(direct_env, inherited_env)

            image_uri = deploy_cloud_run_service(
                service_key=args.service_key,
                image_tag=args.tag,
                env_vars=env_vars,
                project_id=args.project_id,
                region=args.region,
                service_account=args.service_account,
            )
            print(f"Deployed service '{args.service_key}' with image: {image_uri}")
            return 0

        if args.command == "build-deploy":
            validate_service_exists(args.service_key)

            direct_env = _parse_env_vars(args.env)
            inherited_env = _collect_env_vars_from_names(args.env_from)
            env_vars = _merge_env_vars(direct_env, inherited_env)

            image_uri = build_and_push_image(
                service_key=args.service_key,
                image_tag=args.tag,
                project_dir=args.project_dir,
            )
            deploy_cloud_run_service(
                service_key=args.service_key,
                image_tag=args.tag,
                env_vars=env_vars,
                project_id=args.project_id,
                region=args.region,
            )
            print(
                f"Built, pushed, and deployed service '{args.service_key}' with image: {image_uri}"
            )
            return 0

        parser.print_help()
        return 1

    except GCPDeploymentError as exc:
        print(f"Deployment error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())