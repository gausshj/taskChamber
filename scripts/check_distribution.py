"""Audit release archives before they can be uploaded to a package index."""

from __future__ import annotations

import argparse
import configparser
import email
import hashlib
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EXPECTED_NAME = "taskchamber"
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_ENTRY_POINT = "taskchamber.cli:main"
EXPECTED_CLAUDE_REQUIREMENT = "claude-agent-sdk==0.2.115; extra == 'claude'"
EXPECTED_PROJECT_URLS = {
    "Documentation, https://github.com/gausshj/taskChamber/tree/main/docs",
    "Homepage, https://github.com/gausshj/taskChamber",
    "Issues, https://github.com/gausshj/taskChamber/issues",
    "Repository, https://github.com/gausshj/taskChamber",
}
FORBIDDEN_NAMES = {
    ".mcp.json",
    ".netrc",
    "agents.md",
    "claude.md",
    "id_ed25519",
    "id_rsa",
    "settings.json",
}
FORBIDDEN_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
FORBIDDEN_SOURCE_PATHS = {
    ".env.example",
    "docs",
    "scripts",
    "tests",
    "uv.lock",
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "Anthropic API key": re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    "GitHub legacy token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained token": re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "populated API key assignment": re.compile(
        rb"(?im)^(?:ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN|DEEPSEEK_API_KEY|"
        rb"GLM_API_KEY|OPENAI_API_KEY|TASKCHAMBER_PROFILE__[A-Z0-9_]+__API_KEY|"
        rb"Z_AI_API_KEY)\s*=\s*(?!\.\.\.$|replace|example)[^\s#]{16,}\s*$"
    ),
}


@dataclass(frozen=True)
class Distribution:
    path: Path
    members: dict[str, bytes]


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name}")


def _read_wheel(path: Path) -> Distribution:
    with zipfile.ZipFile(path) as archive:
        members: dict[str, bytes] = {}
        for info in archive.infolist():
            _validate_member_name(info.filename)
            if not info.is_dir():
                members[info.filename] = archive.read(info)
    return Distribution(path=path, members=members)


def _read_sdist(path: Path) -> Distribution:
    with tarfile.open(path, mode="r:gz") as archive:
        members: dict[str, bytes] = {}
        for member in archive.getmembers():
            _validate_member_name(member.name)
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {member.name}")
            members[member.name] = extracted.read()
    return Distribution(path=path, members=members)


def _single_member(distribution: Distribution, suffix: str) -> tuple[str, bytes]:
    matches = [
        (name, content) for name, content in distribution.members.items() if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{distribution.path.name}: expected one {suffix!r} member, found {len(matches)}"
        )
    return matches[0]


def _audit_member_paths(distribution: Distribution) -> None:
    for name in distribution.members:
        path = PurePosixPath(name)
        for part in path.parts:
            lowered = part.lower()
            if lowered == ".env" or lowered.startswith(".env."):
                raise ValueError(f"{distribution.path.name}: dotenv file was packaged: {name}")
            if lowered in FORBIDDEN_NAMES or PurePosixPath(lowered).suffix in FORBIDDEN_SUFFIXES:
                raise ValueError(f"{distribution.path.name}: sensitive path was packaged: {name}")
            if lowered in {".git", ".venv", "__pycache__"}:
                raise ValueError(f"{distribution.path.name}: local state was packaged: {name}")


def _audit_secrets(distribution: Distribution) -> None:
    for name, content in distribution.members.items():
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                raise ValueError(f"{distribution.path.name}: possible {label} in {name}")


def _project_version() -> str:
    import tomllib

    with Path("pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    return str(document["project"]["version"])


def _audit_wheel(distribution: Distribution, version: str) -> None:
    names = set(distribution.members)
    required_package_members = {
        "taskchamber/__init__.py",
        "taskchamber/cli.py",
        "taskchamber/py.typed",
        "taskchamber/transport/mcp.py",
    }
    missing = required_package_members - names
    if missing:
        raise ValueError(f"{distribution.path.name}: missing package members: {sorted(missing)}")

    _, metadata_bytes = _single_member(distribution, ".dist-info/METADATA")
    metadata = email.message_from_bytes(metadata_bytes)
    expected_metadata = {
        "Name": EXPECTED_NAME,
        "Version": version,
        "Requires-Python": ">=3.11",
        "License-Expression": EXPECTED_LICENSE,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{distribution.path.name}: {field} is {metadata.get(field)!r}, "
                f"expected {expected!r}"
            )

    extras = set(metadata.get_all("Provides-Extra", []))
    if "claude" not in extras:
        raise ValueError(f"{distribution.path.name}: missing the claude optional dependency")
    requirements = set(metadata.get_all("Requires-Dist", []))
    if EXPECTED_CLAUDE_REQUIREMENT not in requirements:
        raise ValueError(
            f"{distribution.path.name}: Claude SDK is not pinned to the tested version"
        )
    project_urls = set(metadata.get_all("Project-URL", []))
    if project_urls != EXPECTED_PROJECT_URLS:
        raise ValueError(f"{distribution.path.name}: project URLs are incomplete or unexpected")

    _, entry_points_bytes = _single_member(distribution, ".dist-info/entry_points.txt")
    entry_points = configparser.ConfigParser()
    entry_points.read_string(entry_points_bytes.decode("utf-8"))
    if entry_points.get("console_scripts", "taskchamber", fallback=None) != EXPECTED_ENTRY_POINT:
        raise ValueError(f"{distribution.path.name}: taskchamber console entry point is missing")

    license_members = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
    if len(license_members) != 1:
        raise ValueError(f"{distribution.path.name}: Apache-2.0 license file is missing")


def _audit_sdist(distribution: Distribution, version: str) -> None:
    root = f"{EXPECTED_NAME}-{version}"
    required = {
        f"{root}/LICENSE",
        f"{root}/README.md",
        f"{root}/pyproject.toml",
        f"{root}/taskchamber/__init__.py",
        f"{root}/taskchamber/py.typed",
    }
    missing = required - set(distribution.members)
    if missing:
        raise ValueError(f"{distribution.path.name}: missing source members: {sorted(missing)}")

    for name in distribution.members:
        relative = PurePosixPath(name).relative_to(root)
        if relative.parts and relative.parts[0] in FORBIDDEN_SOURCE_PATHS:
            raise ValueError(f"{distribution.path.name}: non-release source was packaged: {name}")


def _load_distributions(directory: Path) -> tuple[Distribution, Distribution]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"{directory}: expected exactly one wheel and one source distribution; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} source distribution(s)"
        )
    return _read_wheel(wheels[0]), _read_sdist(sdists[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing one wheel and one sdist."
    )
    args = parser.parse_args()

    version = _project_version()
    wheel, sdist = _load_distributions(args.directory)
    for distribution in (wheel, sdist):
        _audit_member_paths(distribution)
        _audit_secrets(distribution)
    _audit_wheel(wheel, version)
    _audit_sdist(sdist, version)

    for distribution in (wheel, sdist):
        digest = hashlib.sha256(distribution.path.read_bytes()).hexdigest()
        print(f"PASS {distribution.path.name} sha256={digest}")


if __name__ == "__main__":
    main()
