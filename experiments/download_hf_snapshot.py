from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a HuggingFace repository snapshot for SPICE experiments")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        token=args.token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print({"repo_id": args.repo_id, "local_dir": path})


if __name__ == "__main__":
    main()
