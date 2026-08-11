# Releasing TaskChamber

TaskChamber is distributed as a local stdio MCP executable through PyPI. A
release installs the server and selected runtime adapter into a persistent uv
tool environment; it does not deploy a remote MCP service or register the
server in an MCP client.

## One-time publisher configuration

The PyPI project uses GitHub Actions Trusted Publishing. The pending publisher
and GitHub environment must use these exact values:

| Setting | Value |
| --- | --- |
| PyPI project | `taskchamber` |
| GitHub owner | `gausshj` |
| GitHub repository | `taskChamber` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Do not create a long-lived PyPI API token or add one to GitHub Secrets. The
publish job has only the permissions needed to retrieve the verified artifacts
and request a short-lived OIDC publishing credential.

## Preparing a release

1. Make the release changes through a pull request and wait for CI to pass.
2. Update `pyproject.toml` to the intended PEP 440 version and refresh
   `uv.lock`.
3. Run the complete local verification:

   ```bash
   uv sync --locked --all-groups
   uv run --no-sync --no-build pre-commit run --all-files --show-diff-on-failure
   uv run --no-sync --no-build pytest -q
   uv build --no-sources
   uv run --no-sync --no-build python scripts/check_distribution.py dist
   uv run --no-sync --no-build python scripts/test_uv_tool_install.py dist/*.whl
   ```

   The isolated PEP 517 build backend (`hatchling`), which `uv.lock` does not
   cover, is pinned by `tool.uv build-constraint-dependencies` in
   `pyproject.toml`. uv applies that pin to `sync`, `run`, and `build` alike;
   update it deliberately whenever `build-system.requires` changes.

   The bundled runtime extras pin their adapter SDK versions. Updating one is a
   compatibility change that requires the focused adapter tests and a new
   TaskChamber release.

4. Review the wheel and source-distribution hashes printed by the audit.
5. Merge the pull request into `main`.

## Publishing

Create and push an annotated version tag from the reviewed `main` commit:

```bash
git switch main
git pull --ff-only
git tag -a v0.1.0 -m "TaskChamber 0.1.0"
git push origin v0.1.0
```

`.github/workflows/release.yml` refuses to publish unless the tagged commit is
contained in `main` and the tag is exactly `v` followed by the package version.
It rebuilds and rechecks both archives, performs two consecutive
`uv tool install` operations in one disposable environment, and completes a
real credential-free stdio MCP call before the protected `pypi` environment can
publish.

PyPI release files and version numbers are immutable. Fixes require a new
version rather than rebuilding or replacing an existing release.

## Consumer verification

After publication, test from outside the source checkout:

```bash
uv tool install --python 3.11 'taskchamber[claude]==0.1.0'
uv tool list
```

Register the resulting `taskchamber` executable separately in the chosen MCP
client. Runtime/provider configuration belongs to the host environment and
must not be embedded in package metadata or a committed client configuration.
