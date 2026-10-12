"""Check access to pinned checkpoint files; optionally populate the HF cache."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/validation-models.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

    report = {"time": datetime.now(UTC).isoformat(), "download": args.download, "models": []}
    for row in json.loads(args.manifest.read_text())["models"]:
        entry = {"model": row, "status": "fail"}
        report["models"].append(entry)
        try:
            for revision in (row["revision"], row["tokenizer_revision"]):
                if not re.fullmatch("[0-9a-f]{40}", revision):
                    raise ValueError("checkpoint revisions must be immutable SHAs")
            entry["files"] = []
            for name, revision in (
                ("config.json", row["revision"]),
                ("model.safetensors.index.json", row["revision"]),
                ("tokenizer_config.json", row["tokenizer_revision"]),
                ("tokenizer.json", row["tokenizer_revision"]),
            ):
                metadata = get_hf_file_metadata(hf_hub_url(row["id"], name, revision=revision))
                if metadata.commit_hash != revision:
                    raise ValueError(f"{name}: server revision differs from requested SHA")
                entry["files"].append(
                    {
                        "name": name,
                        "revision": metadata.commit_hash,
                        "etag": metadata.etag,
                        "size": metadata.size,
                    }
                )
            if args.download:
                for revision in dict.fromkeys((row["revision"], row["tokenizer_revision"])):
                    snapshot_download(
                        row["id"],
                        revision=revision,
                        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"],
                    )
            entry["status"] = "pass"
        except Exception as error:
            entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "guidance": "Authenticate with hf auth login, obtain repository "
                "access if gated, and check network/cache storage.",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return int(any(entry["status"] != "pass" for entry in report["models"]))


if __name__ == "__main__":
    raise SystemExit(main())
