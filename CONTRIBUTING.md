# Contributing

Changes must preserve the benchmark's content-addressed inputs, closed artifact schemas, and
deterministic tie policies. A method change that can alter ranking bytes requires a new method
identity or version and an update to [the method contracts](docs/method-contracts.md).

Set up the locked development environment and run the local gates:

```console
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Do not commit official benchmark inputs, patient or trial text from a Local Run, model weights,
dense indexes, credentials, provider traces, or machine-local paths. Use the packaged synthetic
fixture for tests. A pull request should state which contract or reference System changes and how
the change affects existing artifact identities.
