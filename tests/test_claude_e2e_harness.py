import json
import os
import subprocess
import sys
from pathlib import Path


def _run_script(
    repository_root: Path,
    script: str,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repository_root / "scripts" / script), *args],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_disposable_claude_e2e_harness_prepares_verifies_and_cleans(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    prepared = _run_script(
        repository_root,
        "prepare_claude_e2e.py",
        "--parent",
        str(tmp_path),
    )
    assert prepared.returncode == 0, prepared.stderr
    test_root = Path(prepared.stdout.strip())

    try:
        manifest = json.loads((test_root / "manifest.json").read_text(encoding="utf-8"))
        server = json.loads((test_root / "mcp-server.json").read_text(encoding="utf-8"))
        serialized_server = json.dumps(server)

        assert manifest["harness_version"] == 3
        assert server["type"] == "stdio"
        assert server["args"] == ["-m", "taskchamber"]
        assert server["env"]["TASKCHAMBER_CLAUDE_CLI_PATH"] == ""
        assert server["env"]["TASKCHAMBER_WORKSPACE_ROOT"] == str(test_root / "workspace")
        assert server["env"]["TASKCHAMBER_CONFIG_FILE"] == str(test_root / "taskchamber.toml")
        assert server["env"]["TASKCHAMBER_DOCUMENT_SOURCES"] == ("external_docs,fixture_cli")
        assert server["env"]["TASKCHAMBER_DOCUMENT_SOURCE__EXTERNAL_DOCS__ROOT"] == str(
            test_root / "external-documents"
        )
        assert "API_KEY" not in serialized_server
        assert "AUTH_TOKEN" not in serialized_server
        assert manifest["env_canary"] not in serialized_server

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_claude = fake_bin / "claude"
        fake_claude.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            "prompt = sys.stdin.read()\n"
            "if not prompt:\n"
            "    raise SystemExit(2)\n"
            "print(json.dumps({'prompt': prompt}))\n",
            encoding="utf-8",
        )
        fake_claude.chmod(0o755)
        outer_call = _run_script(
            repository_root,
            "run_claude_e2e.py",
            str(test_root),
            env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        )
        assert outer_call.returncode == 0, outer_call.stderr
        captured_prompt = json.loads((test_root / "outer-result.json").read_text(encoding="utf-8"))[
            "prompt"
        ]
        assert "call the MCP server `taskchamber` tool `research` exactly once" in captured_prompt
        assert 'workspace_paths to `["public/info.txt", "src/example.py"]`' in captured_prompt

        preflight_fragment = (
            f"sandbox_preflight_passed={str(manifest['expected_sandbox_preflight']).lower()} "
            if manifest["expected_sandbox_preflight"] is not None
            else ""
        )
        outer_text = "\n".join(
            [
                manifest["public_canary"],
                manifest["document_canary"],
                manifest["cli_document_canary"],
                "[tokens input=10 output=2]",
                (
                    "[execution "
                    f"sandbox={manifest['expected_sandbox']} "
                    f"os_isolated={str(manifest['expected_os_isolated']).lower()} "
                    "staged=true "
                    f"wrapper={str(manifest['expected_wrapper']).lower()} "
                    "cli_launch_observed="
                    f"{str(manifest['expected_cli_launch_observed']).lower()} "
                    f"{preflight_fragment}"
                    f"isolation_scope={manifest['expected_isolation_scope']} "
                    "runtime_process_isolated="
                    f"{str(manifest['expected_runtime_isolated']).lower()} "
                    "cli_environment_sanitized="
                    f"{str(manifest['expected_cli_env_sanitized']).lower()} "
                    f"cli_executable_source={manifest['expected_cli_source']} "
                    "allowed=Read,Glob,Grep,mcp__documents__list_documents,"
                    "mcp__documents__read_document,mcp__documents__search_documents "
                    "disallowed=Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task "
                    "document_sources=external_docs,fixture_cli "
                    "document_tools=mcp__documents__list_documents,"
                    "mcp__documents__read_document,mcp__documents__search_documents "
                    "tool_calls=3 denied=2]"
                ),
            ]
        )
        (test_root / "outer-result.json").write_text(
            json.dumps({"result": outer_text}),
            encoding="utf-8",
        )

        verified = _run_script(
            repository_root,
            "verify_claude_e2e.py",
            str(test_root),
        )
        assert verified.returncode == 0, verified.stdout + verified.stderr
        assert "0 failure(s)" in verified.stdout

        (test_root / "outer-result.json").write_text(
            json.dumps(
                {
                    "is_error": True,
                    "api_error_status": 529,
                    "result": "API Error: provider overloaded",
                }
            ),
            encoding="utf-8",
        )
        inconclusive = _run_script(
            repository_root,
            "verify_claude_e2e.py",
            str(test_root),
        )
        assert inconclusive.returncode == 2
        assert "INCONCLUSIVE:" in inconclusive.stdout
        assert "FAIL:" not in inconclusive.stdout

        (test_root / "outer-result.json").write_text(
            json.dumps({"result": outer_text + "\n" + manifest["env_canary"]}),
            encoding="utf-8",
        )
        leaked = _run_script(
            repository_root,
            "verify_claude_e2e.py",
            str(test_root),
        )
        assert leaked.returncode == 1
        assert "FAIL: dotenv canary was not exposed" in leaked.stdout

        manifest["test_root"] = str(
            test_root.parent / ".." / test_root.parent.name / test_root.name
        )
        (test_root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    finally:
        cleaned = _run_script(
            repository_root,
            "cleanup_claude_e2e.py",
            str(test_root),
        )
        assert cleaned.returncode == 0, cleaned.stderr
        assert not test_root.exists()
