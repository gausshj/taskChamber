# SonarCloud analysis

TaskChamber uses CI-based SonarCloud analysis. SonarQube Cloud automatic
analysis cannot import test coverage, so the `Coverage` job in
`.github/workflows/ci.yml` generates `coverage.xml` and runs the pinned
`SonarSource/sonarqube-scan-action` on every same-repository pull request and
every `main` push. Fork pull requests cannot read `SONAR_TOKEN` and skip the
scan step.

`sonar-project.properties` scopes the analysis: `taskchamber/` is source,
`tests/` is test code, the supported Python versions are 3.11-3.14, and the
Python coverage report is imported from `coverage.xml`.

## Coverage without SonarCloud credentials

The coverage report does not require any token:

```bash
uv sync --locked --all-groups
uv run --no-sync --no-build pytest -q \
  --cov=taskchamber --cov-branch --cov-report=term-missing --cov-report=xml
```

`coverage.xml`, `.coverage*`, `htmlcov/`, and `.scannerwork/` are git-ignored;
never commit reports, scanner caches, or tokens.

## Administrator setup (one-time)

1. In the SonarCloud project (`gausshj_taskChamber`), open
   Administration > Analysis Method and **disable automatic analysis** so each
   change produces exactly one analysis from CI.
2. Create a SonarCloud token for the project and store it as the
   `SONAR_TOKEN` GitHub Actions secret (Settings > Secrets and variables >
   Actions). No other Sonar credential belongs in the repository. Until the
   token exists, the workflow detects its absence and skips the scan step, so
   CI stays green during the transition.
3. Keep the quality gate computed on new code (the default "Sonar way" gate
   is selected). The `SonarCloud Code Analysis` check is the reported gate
   result on each pull request.
4. After the first CI-based analysis, review the findings imported under the
   scoped configuration and record false-positive or accepted decisions in
   SonarCloud instead of changing safe behavior only to reduce the count.
