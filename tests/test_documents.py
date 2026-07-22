import json
import sys
from pathlib import Path

import pytest

from taskchamber.application.documents import (
    CommandDocumentSource,
    DirectoryDocumentSource,
    DocumentSourceRegistry,
    build_document_source_registry,
)
from taskchamber.config import (
    CommandDocumentSourceConfig,
    DirectoryDocumentSourceConfig,
    DocumentParameterConfig,
    load_configuration,
)
from taskchamber.core.documents import DocumentRequestError, DocumentSourceError


@pytest.mark.anyio
async def test_directory_sources_unify_multiple_roots_without_copying(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    docs = tmp_path / "external-docs"
    logs = tmp_path / "external-logs"
    workspace.mkdir()
    docs.mkdir()
    logs.mkdir()
    (docs / "guide.md").write_text("alpha\nshared fact\n", encoding="utf-8")
    (docs / ".env").write_text("SECRET=not-readable\n", encoding="utf-8")
    (logs / "service.log").write_text("boot\nshared fact\n", encoding="utf-8")
    (logs / "private.pem").write_text("not-readable\n", encoding="utf-8")

    registry = DocumentSourceRegistry(
        {
            "docs": DirectoryDocumentSource(DirectoryDocumentSourceConfig(name="docs", root=docs)),
            "logs": DirectoryDocumentSource(DirectoryDocumentSourceConfig(name="logs", root=logs)),
        }
    )
    catalog = await registry.open(("docs", "logs"), query="shared fact")

    listed = await catalog.list_documents()
    assert {(item.source, item.document_id) for item in listed} == {
        ("docs", "guide.md"),
        ("logs", "service.log"),
    }
    hits = await catalog.search_documents(query="shared fact")
    assert {(hit.source, hit.document_id, hit.line) for hit in hits} == {
        ("docs", "guide.md", 2),
        ("logs", "service.log", 2),
    }
    page = await catalog.read_document(source="docs", document_id="guide.md")
    assert page.content == "alpha\nshared fact"

    # The catalog reads the configured source in place. No external file appears
    # in the project workspace or under a synthetic source folder.
    assert list(workspace.rglob("*")) == []
    assert not (workspace / "docs" / "guide.md").exists()


@pytest.mark.anyio
async def test_directory_source_patterns_are_anchored_to_source_root(tmp_path: Path) -> None:
    nested = tmp_path / ".pytest_cache"
    nested.mkdir()
    (tmp_path / "README.md").write_text("project readme", encoding="utf-8")
    (nested / "README.md").write_text("cache readme", encoding="utf-8")

    source = DirectoryDocumentSource(
        DirectoryDocumentSourceConfig(
            name="readme",
            root=tmp_path,
            include=("README.md",),
        )
    )
    prepared = await source.prepare(query="ignored")

    assert [item.document_id for item in await prepared.list_documents(pattern=None, limit=10)] == [
        "README.md"
    ]


@pytest.mark.anyio
async def test_directory_source_recursive_glob_matches_deep_documents(tmp_path: Path) -> None:
    deep = tmp_path / "one" / "two" / "three"
    deep.mkdir(parents=True)
    (deep / "guide.md").write_text("deep guide", encoding="utf-8")

    source = DirectoryDocumentSource(
        DirectoryDocumentSourceConfig(
            name="docs",
            root=tmp_path,
            include=("**/*.md",),
        )
    )
    prepared = await source.prepare(query="ignored")

    assert [item.document_id for item in await prepared.list_documents(pattern=None, limit=10)] == [
        "one/two/three/guide.md"
    ]


@pytest.mark.anyio
async def test_directory_source_recursive_excludes_match_at_any_depth(tmp_path: Path) -> None:
    files = {
        "x/y/guide.md": "public",
        "x/y/one/two/deep.md": "public",
        "x/y/private/immediate.md": "private",
        "x/y/one/private/deep.md": "private",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    source = DirectoryDocumentSource(
        DirectoryDocumentSourceConfig(
            name="docs",
            root=tmp_path,
            include=("x/y/**/*.md",),
            exclude=("**/private/**",),
        )
    )
    prepared = await source.prepare(query="ignored")

    assert [item.document_id for item in await prepared.list_documents(pattern=None, limit=10)] == [
        "x/y/guide.md",
        "x/y/one/two/deep.md",
    ]


@pytest.mark.anyio
async def test_directory_source_skips_symlinks_and_rechecks_the_file(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    target = root / "safe.txt"
    target.write_text("safe", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    source = DirectoryDocumentSource(DirectoryDocumentSourceConfig(name="docs", root=root))
    prepared = await source.prepare(query="safe")

    assert [item.document_id for item in await prepared.list_documents(pattern=None, limit=10)] == [
        "safe.txt"
    ]
    target.unlink()
    target.symlink_to(tmp_path / "outside.txt")
    with pytest.raises(DocumentSourceError, match="safe"):
        await prepared.read_document("safe.txt", start_line=1, max_lines=10)


@pytest.mark.anyio
async def test_command_source_uses_fixed_argv_and_parses_json_documents(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    query = f"topic; touch {marker}"
    script = (
        "import json,sys; "
        "print(json.dumps({'documents':[{'id':'api/result.md',"
        "'content':'query='+sys.argv[1]}]}))"
    )
    source = CommandDocumentSource(
        CommandDocumentSourceConfig(
            name="cli_docs",
            argv=(sys.executable, "-c", script, "{query}"),
            output_format="json",
        ),
        environment={},
    )

    prepared = await source.prepare(query=query)
    page = await prepared.read_document("api/result.md", start_line=1, max_lines=20)

    assert page.content == f"query={query}"
    assert not marker.exists()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "标题": "记录详情",
                "values": [1, "二", {"nested": True}],
                "metadata": {"owner": "测试"},
            },
            id="object",
        ),
        pytest.param([1, "二", {"nested": True}], id="array"),
        pytest.param("纯文本", id="string"),
        pytest.param(42, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
    ],
)
@pytest.mark.anyio
async def test_command_source_exposes_arbitrary_json_as_one_document(payload: object) -> None:
    script = f"print({json.dumps(json.dumps(payload, ensure_ascii=False))})"
    source = CommandDocumentSource(
        CommandDocumentSourceConfig(
            name="json_detail",
            argv=(sys.executable, "-c", script),
            output_format="json_document",
            document_id="detail.json",
        ),
        environment={},
    )

    prepared = await source.prepare(query="ignored")
    listed = await prepared.list_documents(pattern=None, limit=10)
    page = await prepared.read_document("detail.json", start_line=1, max_lines=100)

    assert [item.document_id for item in listed] == ["detail.json"]
    assert json.loads(page.content) == payload
    assert "\\u" not in page.content


@pytest.mark.anyio
async def test_command_source_forwards_only_allowlisted_environment(tmp_path: Path) -> None:
    configuration = load_configuration(
        environment={
            "ALLOWED_TOKEN": "visible-to-command",
            "HOST_SECRET": "must-not-be-forwarded",
        },
        working_directory=tmp_path,
    )
    script = (
        "import json,os; print(json.dumps({'allowed':os.getenv('ALLOWED_TOKEN'),"
        "'host_present':str('HOST_SECRET' in os.environ)}))"
    )
    config = CommandDocumentSourceConfig(
        name="api_cli",
        argv=(sys.executable, "-c", script),
        env_allow=("ALLOWED_TOKEN",),
        output_format="json",
    )
    registry = build_document_source_registry({"api_cli": config}, configuration)

    catalog = await registry.open(("api_cli",), query="ignored")
    allowed = await catalog.read_document(source="api_cli", document_id="allowed")
    host = await catalog.read_document(source="api_cli", document_id="host_present")

    assert allowed.content == "visible-to-command"
    assert host.content == "False"


@pytest.mark.anyio
async def test_command_source_enforces_timeout_and_output_limit() -> None:
    timeout_source = CommandDocumentSource(
        CommandDocumentSourceConfig(
            name="slow",
            argv=(sys.executable, "-c", "import time; time.sleep(2)"),
            timeout_seconds=0.05,
        ),
        environment={},
    )
    output_source = CommandDocumentSource(
        CommandDocumentSourceConfig(
            name="large",
            argv=(sys.executable, "-c", "print('x' * 1000)"),
            max_output_bytes=20,
        ),
        environment={},
    )
    failing_source = CommandDocumentSource(
        CommandDocumentSourceConfig(
            name="failed",
            argv=(
                sys.executable,
                "-c",
                "import sys; print('secret-stderr', file=sys.stderr); raise SystemExit(7)",
            ),
        ),
        environment={},
    )

    with pytest.raises(DocumentSourceError, match="time limit"):
        await timeout_source.prepare(query="ignored")
    with pytest.raises(DocumentSourceError, match="output limit"):
        await output_source.prepare(query="ignored")
    with pytest.raises(DocumentSourceError, match="exit code 7") as error:
        await failing_source.prepare(query="ignored")
    assert "secret-stderr" not in str(error.value)


@pytest.mark.anyio
async def test_registry_rejects_unknown_and_duplicate_sources(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    registry = DocumentSourceRegistry(
        {"docs": DirectoryDocumentSource(DirectoryDocumentSourceConfig(name="docs", root=root))}
    )

    with pytest.raises(DocumentRequestError, match="not configured"):
        await registry.open(("missing",), query="question")
    with pytest.raises(DocumentRequestError, match="duplicate"):
        await registry.open(("docs", "docs"), query="question")


@pytest.mark.anyio
async def test_command_source_accepts_only_declared_structured_parameter(tmp_path: Path) -> None:
    script = (
        "import json,sys; "
        "print(json.dumps({'documents':[{'id':'record.json','content':sys.argv[1]}]}))"
    )
    source = CommandDocumentSource(
        CommandDocumentSourceConfig(
            name="record_detail",
            aliases=("record detail", "记录详情"),
            argv=(sys.executable, "-c", script, "{record_id}"),
            parameters=(
                DocumentParameterConfig(
                    name="record_id",
                    aliases=("record", "记录编号"),
                    pattern=r"^rec_[A-Za-z0-9_-]+$",
                    example="rec_demo",
                ),
            ),
            output_format="json",
        ),
        environment={},
    )
    registry = DocumentSourceRegistry({"record_detail": source})

    catalog = await registry.open(
        ("记录详情",),
        query="analyze",
        parameters={"记录详情": {"记录编号": "rec_demo"}},
    )
    page = await catalog.read_document(
        source="record_detail",
        document_id="record.json",
    )

    assert page.content == "rec_demo"
    assert catalog.source_names == ("record_detail",)

    with pytest.raises(DocumentRequestError, match="may not begin"):
        await registry.open(
            ("record_detail",),
            query="analyze",
            parameters={"record_detail": {"record_id": "--delete"}},
        )


@pytest.mark.anyio
async def test_document_source_typo_returns_canonical_suggestion(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    registry = DocumentSourceRegistry(
        {
            "record_detail": DirectoryDocumentSource(
                DirectoryDocumentSourceConfig(
                    name="record_detail",
                    root=root,
                    aliases=("record detail",),
                )
            )
        }
    )

    with pytest.raises(DocumentRequestError) as error:
        await registry.open(("recrod detial",), query="question")

    message = str(error.value)
    assert "record_detail" in message
    assert "do not fall back to shell" in message
