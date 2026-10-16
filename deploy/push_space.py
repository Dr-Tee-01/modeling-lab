"""Publish the app to a Hugging Face Space (Docker SDK).

Usage:  HF_TOKEN=... python deploy/push_space.py <hf-username>/<space-name>

Creates the Space if it does not exist, then uploads exactly what the Docker image
needs, with a README whose front matter tells Spaces how to run it. The upload
goes through the Hub API, which stores binary data files (meshes, .npy, .mat)
the way the Hub requires, so no git-lfs setup is needed.
"""
import os
import pathlib
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONT_MATTER = """---
title: Modeling Lab
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Epidemics, 3D shapes and markets, computed live
---

"""
SHIP = ["Dockerfile", "requirements.txt", "pyproject.toml", "src", "data", "deploy/gunicorn.conf.py"]


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy/push_space.py <hf-username>/<space-name>")
    repo_id, token = sys.argv[1], os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face access token with write permission.")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = pathlib.Path(tmp)
        for item in SHIP:
            src, dst = ROOT / item, stage / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
            else:
                shutil.copy2(src, dst)
        (stage / "README.md").write_text(FRONT_MATTER + (ROOT / "README.md").read_text())
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="space",
                          commit_message=f"Deploy {sha}", delete_patterns=["src/**", "data/**", "deploy/**"])
    print(f"Pushed. The Space builds in a few minutes: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
