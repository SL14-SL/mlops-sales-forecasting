import sys
import json
import subprocess
from pathlib import Path


def get_current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def is_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return False


def main(manifest_path: str):
    path = Path(manifest_path)

    if not path.exists():
        raise SystemExit(f"❌ Manifest not found: {manifest_path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))

    dataset_version = manifest.get("dataset_version")
    git_commit = manifest.get("git_commit")
    config_name = manifest.get("config_name")

    current_commit = get_current_git_commit()
    dirty = is_git_dirty()

    print("\n📦 Manifest Inspection")
    print("-" * 40)
    print(f"Dataset version : {dataset_version}")
    print(f"Config          : {config_name}")
    print(f"Manifest commit : {git_commit}")
    print()

    print("🧠 Current Environment")
    print("-" * 40)
    print(f"Current commit  : {current_commit}")
    print(f"Git dirty       : {dirty}")
    print()

    # ---- Checks ----
    if git_commit is None:
        print("⚠️  No git_commit stored in manifest.")
    elif current_commit != git_commit:
        print("⚠️  Current code does NOT match manifest commit!")
    else:
        print("✅ Code matches manifest commit.")

    if dirty:
        print("⚠️  Working directory has uncommitted changes.")

    print("\n🚀 Reproduction Instructions")
    print("-" * 40)

    if git_commit and current_commit != git_commit:
        print(f"git checkout {git_commit}")
        print("uv sync")

    print(f"uv run python scripts/retrain_from_manifest.py {manifest_path}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/inspect_manifest_repro.py <manifest_path>"
        )

    main(sys.argv[1])