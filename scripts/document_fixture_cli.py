#!/usr/bin/env python3
"""Credential-free JSON document CLI used by the manual MCP smoke test."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--canary", default="virtual-document-cli-ok")
    arguments = parser.parse_args()
    print(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "cli/answer.md",
                        "title": "CLI fixture answer",
                        "media_type": "text/markdown",
                        "content": (
                            "# CLI document source\n\n"
                            f"The configured research query was: {arguments.query}\n"
                            f"Canary: {arguments.canary}\n"
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
