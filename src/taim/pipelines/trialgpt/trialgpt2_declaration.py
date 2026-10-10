"""The declaration of ``TrialGPT2-TAIM-Luna-v1``: what it runs, and the condition it carries.

TrialGPT 2.0's own scorer, unchanged, re-ranks the top 500 of ``TrialGPT-TAIM-Luna-v1``'s frozen
three-query consensus retrieval, served by ``gpt-5.6-luna`` through a Codex bridge. It is TAIM's
composition by design. The owner ruled on 2026-10-06 "Record that through Codex it is comparable
only to itself", so the condition lives here as data the System carries, and the Default-Track
Compare gate reads it. Protocol v14 Section 7 freezes the statement and the caption note.

This module imports only the standard library, so the comparison gate can read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID = "TrialGPT2-TAIM-Luna-v1"

UPSTREAM_REPOSITORY = "https://github.com/TrialGPT2/TrialGPT2.0.git"
UPSTREAM_COMMIT = "1c5f9ebc3b98d1f780ceea68043efdaa2daba880"
UPSTREAM_TREE = "14538051649a86f92f74cb273e6b459f11686850"
UPSTREAM_LICENSE_EXPRESSION = "LicenseRef-NLM-Public-Domain"
UPSTREAM_LICENSE_SHA256 = "sha256:bb4da61d60a340334fd9dd5c71920bc803ae13b5080a51320aba4aba070480ad"
#: Every file of theirs the scorer executes, at the pin. Nothing else of the checkout is read.
UPSTREAM_EXECUTED_FILES = MappingProxyType(
    {
        "llm_pair_metrics.py": (
            "sha256:0c85e2a885cb43b3b65db47560de1d081386e1c5b3433543e73a2f8e07ed9b01"
        ),
        "src/__init__.py": (
            "sha256:01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
        ),
        "src/llm/__init__.py": (
            "sha256:01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
        ),
        "src/llm/client.py": (
            "sha256:4f202f759816e929b1311b983ea25324c2fef14cd6d22ed20b3690ccba235a64"
        ),
        "src/trial_matching/__init__.py": (
            "sha256:01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
        ),
        "src/trial_matching/matcher_with_usage_logging.py": (
            "sha256:032cf06de11ab88fefab9fd954d2fab5de821b3103545debf7869317131a80e0"
        ),
        "src/trial_matching/prompts/__init__.py": (
            "sha256:01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
        ),
        "src/trial_matching/prompts/trial_matcher.py": (
            "sha256:83a1d1ed1e63b8b625f613e9825b114fefa94f7e5bc820da2c6ac27e31564ad0"
        ),
        "src/trial_matching/run_matcher_with_usage_logging.py": (
            "sha256:ccb43e1e505e91aba6fd9c432c616a8c67408f7bf6d6a40f9863a2d254674e51"
        ),
    }
)
UPSTREAM_PROMPT_MODULE = "src/trial_matching/prompts/trial_matcher.py"
UPSTREAM_RUNNER_MODULE = "src.trial_matching.run_matcher_with_usage_logging"
#: SHA-256 of ``SYSTEM_PROMPT`` in their prompt module, 3,808 characters: the one prompt shipped,
#: not established as the paper's general matching policy.
UPSTREAM_SYSTEM_PROMPT_SHA256 = (
    "sha256:0e348a1805ae4a560432f679257d87c4ad1473a47f9eb8f95a2c6ca549149c79"
)
#: Canonical SHA-256 of the strict JSON schema their client sends as ``response_format``.
UPSTREAM_OUTPUT_SCHEMA_SHA256 = (
    "sha256:d97386208975f468ead13eae4895a2f838a194a1f5f036b88cf9f6cc9c561572"
)

MODEL_SERVED = "gpt-5.6-luna"
#: The model label their client sends, through their own ``--model`` option.
MODEL_LABEL_SENT = MODEL_SERVED
CANDIDATE_DEPTH = 500
GENERATION_WORKERS = 48


@dataclass(frozen=True, slots=True)
class BridgeDistortion:
    """One way the Codex bridge changes what TrialGPT 2.0's code asks its model."""

    name: str
    what: str
    consequence: str


@dataclass(frozen=True, slots=True)
class DeclaredComparability:
    """The Systems this one may stand beside, and what would widen that."""

    comparable_to: frozenset[str]
    runs_measured: int
    statement: str
    caption_note: str
    evidence_that_would_change_it: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Declaration:
    """Everything a reader of this System's output must carry, declared on the System."""

    system_id: str
    first_stage: str
    upstream_repository: str
    upstream_commit: str
    license_expression: str
    model_served: str
    bridge_distortions: tuple[BridgeDistortion, ...]
    bridge_integration: tuple[str, ...]
    comparability: DeclaredComparability


BRIDGE_DISTORTIONS = (
    BridgeDistortion(
        name="joined-messages",
        what=(
            "Every call's system and user messages are joined into one prompt, the system message "
            "becoming a 'SYSTEM INSTRUCTIONS:' preamble, because codex exec takes one prompt."
        ),
        consequence=(
            "TrialGPT 2.0 puts its whole matching policy in the system message, so joining the "
            "two changes the instruction boundary on every call. Its effect is not zero by "
            "assumption and has not been measured."
        ),
    ),
    BridgeDistortion(
        name="sampling-not-applied",
        what=(
            "Their client sends temperature 0.0 on every call; it cannot reach the Codex CLI, and "
            "every call records it as sampling_not_applied instead."
        ),
        consequence=(
            "The authors' decoding is not applied, so a repeated call can answer differently "
            "where theirs was asked to answer deterministically."
        ),
    ),
    BridgeDistortion(
        name="harness-and-backbone",
        what=(
            "Every call runs inside Codex's own harness and is served by gpt-5.6-luna, not the "
            "GPT-4.1 of the paper's headline or the GPT-5.4 of the repository's documented "
            "workflow."
        ),
        consequence=(
            "The model and the context it answers in differ from both of the authors' "
            "configurations."
        ),
    ),
)

BRIDGE_INTEGRATION = (
    "Their strict JSON schema reaches Codex as --output-schema.",
    "Every answer returns its token usage in the chat-completion body.",
    "The bridge makes exactly one Codex execution per request and retries nothing; their client "
    "retries as it always does.",
    "Their client's timeout is set above the bridge's through their own LLM_TIMEOUT_S variable.",
    "A response cache serves a resumed run: an answer is cached only when it conforms to their "
    "schema, and served again only for a request identical in messages, schema and model "
    "configuration.",
    "The bridge answers only the requests the run's own candidates produce, and records every "
    "request in a ledger with its pair, outcome and digests.",
)

COMPARABILITY_STATEMENT = (
    "Through Codex, TrialGPT2-TAIM-Luna-v1 is comparable only to itself: the Codex bridge sends "
    "TrialGPT 2.0's system and user messages as one prompt, cannot apply its temperature of 0.0, "
    "wraps every call in Codex's own harness, and is served by gpt-5.6-luna rather than the "
    "GPT-4.1 or GPT-5.4 its authors ran, so its numbers enter no paired comparison, difference or "
    "significance test with the TrialGPT 2.0 paper's, with TrialGPT-TAIM-Luna-v1's or with any "
    "other System's, and stand in no table whose caption implies a shared model condition."
)

CAPTION_NOTE = (
    "TrialGPT2-TAIM-Luna-v1 runs TrialGPT 2.0's scorer through Codex on gpt-5.6-luna, re-ranking "
    "TrialGPT-TAIM-Luna-v1's top 500. It is comparable only to itself: it enters no paired "
    "comparison, and its row does not imply a model condition shared with any other row."
)

TRIALGPT2_TAIM_LUNA_V1 = Declaration(
    system_id=TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID,
    first_stage=(
        "The top 500 per topic of TrialGPT-TAIM-Luna-v1's frozen three-query consensus "
        "retrieval, read from its lock unchanged; this System makes no query generation."
    ),
    upstream_repository=UPSTREAM_REPOSITORY,
    upstream_commit=UPSTREAM_COMMIT,
    license_expression=UPSTREAM_LICENSE_EXPRESSION,
    model_served=MODEL_SERVED,
    bridge_distortions=BRIDGE_DISTORTIONS,
    bridge_integration=BRIDGE_INTEGRATION,
    comparability=DeclaredComparability(
        comparable_to=frozenset({TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID}),
        runs_measured=0,
        statement=COMPARABILITY_STATEMENT,
        caption_note=CAPTION_NOTE,
        evidence_that_would_change_it=(
            "The same pairs through the bridge and through a real chat-completions endpoint at "
            "temperature 0, with the outputs compared. That needs a key and a spend approval, "
            "which are not given.",
        ),
    ),
)


__all__ = [
    "BRIDGE_DISTORTIONS",
    "BRIDGE_INTEGRATION",
    "CANDIDATE_DEPTH",
    "CAPTION_NOTE",
    "COMPARABILITY_STATEMENT",
    "GENERATION_WORKERS",
    "MODEL_LABEL_SENT",
    "MODEL_SERVED",
    "TRIALGPT2_TAIM_LUNA_V1",
    "TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID",
    "UPSTREAM_COMMIT",
    "UPSTREAM_EXECUTED_FILES",
    "UPSTREAM_LICENSE_EXPRESSION",
    "UPSTREAM_LICENSE_SHA256",
    "UPSTREAM_OUTPUT_SCHEMA_SHA256",
    "UPSTREAM_PROMPT_MODULE",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_RUNNER_MODULE",
    "UPSTREAM_SYSTEM_PROMPT_SHA256",
    "UPSTREAM_TREE",
    "BridgeDistortion",
    "Declaration",
    "DeclaredComparability",
]
