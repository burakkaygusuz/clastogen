# Contributing

## Setup

```bash
git clone https://github.com/burakkaygusuz/clastogen.git
cd clastogen
uv sync --dev
```

## Checks

CI runs these on every pull request; run them locally first:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests scripts
uv run pytest
uv run pytest --clastogen -n 2 examples/
```

## Pull Requests

- Open pull requests against `main`. Direct pushes are blocked.
- Use [Conventional Commits](https://www.conventionalcommits.org/) with a scope, e.g. `fix(stats): ...`. Version bumps and GitHub release notes are generated from commit messages.
- For larger changes, open an issue first to agree on the approach.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues as described in [SECURITY.md](SECURITY.md).
