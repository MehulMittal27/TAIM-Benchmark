# TrialGPT-TAIM-Luna-v1 external baseline

This System is a provider-backed TAIM adaptation, not an exact reproduction of the upstream GPT-4
paper System. The release does not include an upstream checkout, provider credentials, generated
outputs, a retrieval index, or the historical recipe-v1 retrieval artifact.

Execution requires a fresh 2,000-deep frozen retrieval artifact and matching lock created against
the amended immutable release, followed by a clean-commit publication preflight. The lock binds
the Track, exact topic set, complete Task Input trial set, and top-500 hashes. The release will not
accept historical campaign artifacts or a retrieval artifact from another Track. The paper
execution freezes 48 generation workers. After that preflight, add these options to the common
command:

The release ships the producer commands and their complete instructions in
`docs/trialgpt-retrieval-producer.md`. Producer locks bind protocol v11 and carry the
query-plan receipt that records their query plans; preflight refuses any other lock. The primary
System keeps 500 candidates and permits up to 1,500 logical provider calls per topic; a 500-call
configuration would be a separately named ablation.

```console
  --system TrialGPT-TAIM-Luna-v1 \
  --trialgpt-frozen-retrieval /path/to/fresh-retrieval.jsonl \
  --trialgpt-frozen-retrieval-lock /path/to/fresh-retrieval-lock.json \
  --trialgpt-publication-contract /path/to/all75-contract.json \
  --trialgpt-workspace /path/to/empty-or-in-place-resume-workspace \
  --trialgpt-generation-workers 48 \
  --top-k 20
```

Provider calls require separate explicit authorization. A timeout, provider failure, or missing
output is not a clinical judgment and cannot be imputed into the ranking.

The SIGIR rows use all declared topics under their respective TAIM query contracts. They are not
reproductions of TrialGPT's 58-patient subset. Do not use this System's retrieved top 500 as a
neutral corpus because that set depends on System scores.

## Preflight

For TrialGPT preflight, select the same Track and Profile used by the run:

```console
trial-benchmark patient-to-trial trialgpt preflight \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --data-dir prepared/trec-ct-2022 \
  --clinical-as-of 2022-01-01T00:00:00Z \
  --pool-count <count-from-receipt> \
  --pool-ids-sha256 sha256:<ordered-digest-from-receipt> \
  --pool-receipt-id sha256:<receipt-id> \
  --protocol-approval-id "$EXTERNAL_PROTOCOL_ID" \
  --frozen-retrieval /path/to/track-specific-retrieval.jsonl \
  --frozen-retrieval-lock /path/to/track-specific-retrieval-lock.json \
  --generation-workers 48 \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.2.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --output /path/to/trialgpt-contract.json
```


## Shared configured generation

Runners construct `taim.models.Model(model_id=..., backend="codex", parameters=...)` and inject
it into the TrialGPT stages. `taim.generation` owns the existing three-attempt runner and optional
SQLite response cache; `taim.model_codex` owns the constrained CLI transport. Construction and
configuration reads launch no processes. Explicit preflight still probes CLI and catalogue
identity. Generation may transmit input, spend quota and persist sensitive output. CLI execution
does not imply local inference or establish stronger tool/instruction isolation.

The `model-v1` requested configuration now supplies cache identity. Publication and private paper
runners use `model-generation-v1-cache.sqlite3`; explicit query and hybrid cache arguments receive
a `.model-v1` suffix. Old caches and evidence retain their original bytes and identities. Empty
eligibility stages return deterministic no-call traces and need no response-cache row. Successful
cache reuse preserves old usage as evidence, without counting it as new inference.

Codex attempt metadata uses `trace_version="model-generation-v1"`. Its logical-call ID is the
actual runner identity and its attempt number includes resumed transport attempts. Requested and
reported model identities remain separate. Catalogue fallback retains
`catalog_pinned_provider_revision_unavailable`; it does not identify immutable remote weights.
Source moves and consumer changes produce new implementation hashes. Prompt, normalization,
semantic validation, retry and abstention contracts remain unchanged.

The transport receives the output schema unless a Codex Model sets `output_schema_transport` to
`none`. Local acceptance always relies on each caller's validator. New consumers must validate
shape and domain constraints and include their contract versions in request identity. Validator
code is not automatically hashed. The response cache is not an inference-engine KV cache. It
supplies neither an in-flight dispatch journal nor a durable attempt budget. Alias
reproducibility and agent orchestration remain separate work.

A request with `output_schema=None` asks for free text through the same Model, runner and
backends. No schema reaches the transport: Codex gets no `--output-schema`, HTTP no
`response_format` and no schema instruction, OpenCode no schema instruction. The output is the
model's text, never parsed as JSON, and any normalizer or validator receives that string. An empty
text is repaired like invalid JSON. Retry, provenance and caching are unchanged, and a structured
request's identity is the same as before free text existed.

A Codex Model with `parameters={"output_schema_transport": "none"}` keeps a structured request's
schema off the CLI: Codex gets no `--output-schema`. The output is still parsed as JSON, normalized,
validated and repaired, and repair prompts still embed the schema. The configuration records
`"output_schema_transport": "none"` under `transport`, so it enters request identity; the default
`strict` is not recorded, and a strict configuration's identity is unchanged.

Since 2026-10-07 `TrialGPT-TAIM-Luna-v1` uses `none` for matching and aggregation, as upstream
TrialGPT sends no response format. `codex exec` sends every schema with `strict: true`, and OpenAI's
server began refusing the matching schema, which was never inside the strict subset. The method
contract, including `matching_schema_sha256`, is unchanged. The transport enters the provider
configuration and the publication contract, so these runs are distinguishable from earlier ones,
which sent both schemas. The deviation list in the private source repository records the change
as D9.


### Additional Model backends

The same `Generates.generate(GenerationRequest) -> GenerationResult` seam accepts explicitly
configured HTTP and OpenCode Models. No backend fallback occurs. These options do not change
frozen benchmark pipeline configurations or establish clinical or deployment readiness.

| Backend | Required options | Supported `parameters` |
| --- | --- | --- |
| `codex` | `model_id` | `reasoning_effort`, as above; `output_schema_transport` as `strict`, the default, or `none` |
| `http` | `model_id`, `endpoint` as the complete chat-completions URL | `temperature` from 0 to 2, `max_tokens` as a positive integer, `response_format` as `json_schema` or `json_object`, for structured requests |
| `opencode` | `model_id` in `provider/model` form | `variant` as a nonempty provider-specific string |

Install the `http` extra for HTTP generation. HTTP defaults are temperature 0, 4096 output tokens
and strict `json_schema`. Use `json_object` explicitly for servers without JSON-schema support;
it still receives the schema and must pass local validation. `system_prompt` supplies caller-owned
instructions and participates in configuration/cache identity. Clinical policy, language and
stopping decisions belong to the caller.

```python
from taim.models import Model

model = Model(
    backend="http",
    model_id="explicit-server-model",
    endpoint="https://model.example/v1/chat/completions",
    system_prompt="Explain supplied evidence in German.",
    parameters={"temperature": 0, "max_tokens": 2048, "response_format": "json_object"},
    auth_headers=lambda: {"Authorization": "Bearer " + runtime_token()},
    request_timeout_seconds=60,
    max_request_bytes=1_048_576,
    max_response_bytes=1_048_576,
)
```

`runtime_token` is an application function, not a TAIM credential loader. Only an Authorization
header is accepted. Endpoints cannot contain user information, query parameters or fragments.
No environment proxy, redirect, compressed response, tool call, refusal, duplicate JSON field,
non-object answer or incomplete choice is accepted. The request cap measures the entire serialized
UTF-8 body, including instructions and schema; retain any application-specific prompt cap too.

For HTTP and OpenCode, `request_timeout_seconds` defaults to 600 and covers all attempts in one
`generate` invocation. Both byte caps default to 1,048,576. An optional runtime `deadline` callable
returns an absolute `time.monotonic()` deadline; the earlier deadline wins. This lets a caller bound
several generation stages by one assessment deadline. HTTP uses cancellable asynchronous I/O
behind the synchronous interface. Callers should dispatch blocking `generate` off their event loop;
invocation on an already running loop also works through a separate worker.

Runtime `auth_headers`, `http_transport`, `deadline` and `opencode_environment` are excluded from
configuration, cache keys and provenance. `http_transport` accepts an `httpx.AsyncBaseTransport`
for offline tests; each attempt closes its client and transport. Authorization and cache access
remain caller responsibilities. Use a separate cache or bind a caller-owned processing identity
in `input_payload` when credentials or authorization scope change. TAIM must not derive that
identity from credentials. Failure attempts expose fixed `metadata["failure_code"]` values for
timeout, availability, HTTP status, byte ceilings and malformed responses. Error bodies and
headers are never persisted. HTTP 401/403 stops retries immediately.

OpenCode uses `opencode_executable="opencode"`, with an optional `opencode_environment` callable
that returns provider environment variables ending in `_API_KEY` or `_AUTH_TOKEN`. Other variables
and OpenCode/XDG overrides are refused. Existing user credentials and configuration are not loaded:
each attempt gets a temporary HOME and XDG directories, a dedicated generation agent, disabled
sharing and updates, denied tool permissions, and `--pure` to disable external plugins. No session
is resumed. The stdin prompt is not placed in process arguments. POSIX process groups are required;
timeout and combined stdout/stderr limits kill and reap the group, including leftover descendants.
This is process cleanup, not an operating-system security sandbox.

The invocation follows installed OpenCode 1.18.32 `run --help` and the
[versioned CLI source](https://github.com/anomalyco/opencode/blob/v1.18.32/packages/opencode/src/cli/cmd/run.ts).
It uses `run --format json --model ... --agent ... --pure --title ... --dir ...` and optional
`--variant`. The CLI has no schema flag, so the schema is included with the prompt. Completed text
and a `step_finish` with reason `stop` must belong to one session/message; tools, errors and
incomplete/mixed streams remain unresolved. Local request validators still own domain acceptance.

HTTP provenance records requested and reported model strings separately, including mismatches;
a server-reported alias is not an immutable model revision. OpenCode's JSON stream does not expose
the model identity, so `reported_model` stays null and the observed CLI version is recorded. Neither
transport claims pinned weights. Numeric token usage is retained when the response supplies it.
Offline fixtures qualify the wire/CLI behavior, not live-model quality or provider compatibility.
