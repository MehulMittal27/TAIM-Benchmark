"""Observable, qrel-free TrialGPT algorithm boundaries.

The pinned upstream implementation couples retrieval orchestration to BEIR and qrels and assumes
CUDA.  This module preserves its observable inputs, ranking math, prompts, and score arithmetic
while exposing the boundaries through TAIM-owned Snapshot and provider interfaces.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from taim.contracts import content_sha256
from taim.eligibility import eligibility_criterion_trial
from taim.snapshot import DerivedView, TrialDocument

TRIALGPT_RRF_K = 20
TRIALGPT_CANDIDATE_DEPTH = 2_000
TRIALGPT_LLM_CANDIDATE_DEPTH = 500
TAIM_CONTROLLED_BM25_VARIANT_ID = "taim-controlled-bm25-v1"
TRIALGPT_EPSILON = 1e-9
TRIALGPT_TOKENIZER_ID = "taim-trialgpt-regex-tokenizer-v1"
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class TrialGPTPaperError(ValueError):
    """Raised when a paper-faithful TrialGPT input or output is invalid."""


INCLUSION_LABELS = frozenset(
    {"not applicable", "not enough information", "included", "not included"}
)
EXCLUSION_LABELS = frozenset(
    {"not applicable", "not enough information", "excluded", "not excluded"}
)


@dataclass(frozen=True, slots=True)
class TrialGPTTrialView:
    """The upstream TrialGPT trial fields reconstructed from a Snapshot trial."""

    trial_id: str
    brief_title: str
    diseases: tuple[str, ...]
    interventions: tuple[str, ...]
    brief_summary: str
    inclusion_criteria: str
    exclusion_criteria: str
    retrieval_text: str
    inclusion_criterion_ids: tuple[str, ...] = ()
    exclusion_criterion_ids: tuple[str, ...] = ()
    eligibility_view_sha256: str | None = None
    eligibility_source_sha256: str | None = None
    eligibility_criterion_inventory_sha256: str | None = None

    @property
    def inclusion(self) -> tuple[str, ...]:
        return split_criteria(self.inclusion_criteria)

    @property
    def exclusion(self) -> tuple[str, ...]:
        return split_criteria(self.exclusion_criteria)


def _section_texts(trial: TrialDocument, role: str) -> tuple[str, ...]:
    return tuple(section.text.strip() for section in trial.sections if section.role == role)


def _section_source_texts(trial: TrialDocument, role: str) -> tuple[str, ...]:
    """Prefer bounded source text when reconstructing upstream criterion boundaries."""

    values: list[str] = []
    for section in trial.sections:
        if section.role != role:
            continue
        source = section.provenance[0].raw_value if section.provenance else ""
        values.append(source.strip() or section.text.strip())
    return tuple(values)


def _split_eligibility_text(text: str) -> tuple[str, str]:
    marker = re.compile(r"(?i)\b(inclusion criteria|exclusion criteria)\s*:?\s*")
    matches = list(marker.finditer(text))
    if not matches:
        return text, ""
    parts: dict[str, list[str]] = {"inclusion criteria": [], "exclusion criteria": []}
    for index, match in enumerate(matches):
        kind = match.group(1).casefold()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts[kind].append(text[match.end() : end].strip())
    return "\n\n".join(parts["inclusion criteria"]), "\n\n".join(parts["exclusion criteria"])


def trial_view(
    trial: TrialDocument,
    *,
    eligibility_criterion_view: DerivedView | None = None,
) -> TrialGPTTrialView:
    """Map a Snapshot trial to the exact fields used by TrialGPT prompts/retrieval."""

    titles = _section_texts(trial, "brief_title")
    summaries = _section_texts(trial, "summary")
    conditions = _section_texts(trial, "condition")
    interventions = _section_texts(trial, "intervention")
    eligibility_source_sections = tuple(
        (section.source_text or section.provenance[0].raw_value).strip()
        for section in trial.sections
        if section.role == "eligibility"
    )
    missing = [name for name, values in (("brief_title", titles),) if not values]
    if missing:
        raise TrialGPTPaperError(
            f"trial {trial.trial_id} is missing TrialGPT fields: {', '.join(missing)}"
        )
    inclusion, exclusion = _split_eligibility_text("\n\n".join(eligibility_source_sections))
    inclusion_ids: tuple[str, ...] = ()
    exclusion_ids: tuple[str, ...] = ()
    eligibility_view_sha256: str | None = None
    eligibility_source_sha256: str | None = None
    eligibility_criterion_inventory_sha256: str | None = None
    eligibility = next(
        (section for section in trial.sections if section.role == "eligibility"),
        None,
    )
    if eligibility is not None and eligibility.source_text is not None:
        if eligibility_criterion_view is None:
            raise TrialGPTPaperError(
                f"trial {trial.trial_id} complete eligibility requires its supplied Derived View"
            )
        record = eligibility_criterion_trial(eligibility_criterion_view, trial.trial_id)
        if record is None:
            raise TrialGPTPaperError(
                f"trial {trial.trial_id} eligibility criterion Derived View record is missing"
            )
        if record.source_text_sha256 != eligibility.source_text_sha256:
            raise TrialGPTPaperError(
                f"trial {trial.trial_id} eligibility source hash does not match its Derived View"
            )
        provenance_sha256 = content_sha256([item.to_dict() for item in eligibility.provenance])
        if record.source_provenance_sha256 != provenance_sha256:
            raise TrialGPTPaperError(
                f"trial {trial.trial_id} eligibility provenance does not match its Derived View"
            )
        if any(
            eligibility.source_text[item.source_start : item.source_end] != item.source_span_text
            for item in record.criteria
        ):
            raise TrialGPTPaperError(
                f"trial {trial.trial_id} eligibility source spans do not match its Derived View"
            )
        prompt_items = tuple(
            item for item in record.criteria if split_criteria(item.text) == (item.text.strip(),)
        )
        inclusion_items = tuple(item for item in prompt_items if item.polarity == "inclusion")
        exclusion_items = tuple(item for item in prompt_items if item.polarity == "exclusion")
        inclusion = "\n\n".join(item.text for item in inclusion_items)
        exclusion = "\n\n".join(item.text for item in exclusion_items)
        inclusion_ids = tuple(item.identifier or "" for item in inclusion_items)
        exclusion_ids = tuple(item.identifier or "" for item in exclusion_items)
        eligibility_source_sha256 = eligibility.source_text_sha256
        eligibility_view_sha256 = record.view_sha256
        eligibility_criterion_inventory_sha256 = record.criterion_inventory_sha256
    elif eligibility is not None and eligibility.criteria:
        inclusion_texts = tuple(
            item.text for item in eligibility.criteria if item.polarity == "inclusion"
        )
        exclusion_texts = tuple(
            item.text for item in eligibility.criteria if item.polarity == "exclusion"
        )
        if inclusion_texts or exclusion_texts:
            inclusion = "\n\n".join(inclusion_texts)
            exclusion = "\n\n".join(exclusion_texts)
    return TrialGPTTrialView(
        trial_id=trial.trial_id,
        brief_title=titles[0],
        diseases=conditions,
        interventions=interventions,
        brief_summary=summaries[0] if summaries else "",
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        retrieval_text=_complete_retrieval_text(trial),
        inclusion_criterion_ids=inclusion_ids,
        exclusion_criterion_ids=exclusion_ids,
        eligibility_view_sha256=eligibility_view_sha256,
        eligibility_source_sha256=eligibility_source_sha256,
        eligibility_criterion_inventory_sha256=eligibility_criterion_inventory_sha256,
    )


def _complete_retrieval_text(trial: TrialDocument) -> str:
    text = trial.canonical_text
    for section in trial.sections:
        if section.role != "eligibility" or section.source_text is None:
            continue
        normalized = f"[ELIGIBILITY] {section.text}"
        complete = f"[ELIGIBILITY] {section.source_text}"
        if normalized not in text:
            raise TrialGPTPaperError(
                f"trial {trial.trial_id} canonical eligibility section cannot be replaced"
            )
        text = text.replace(normalized, complete, 1)
    return text


def split_criteria(criteria: str) -> tuple[str, ...]:
    """Apply TrialGPT's duplicated criterion splitting rule exactly."""

    return tuple(
        normalized
        for raw in criteria.split("\n\n")
        if (normalized := raw.strip())
        and "inclusion criteria" not in normalized.casefold()
        and "exclusion criteria" not in normalized.casefold()
        and len(normalized) >= 5
    )


def word_tokenize(text: str) -> tuple[str, ...]:
    """Tokenize with TAIM's dependency-independent publication tokenizer."""

    return tuple(_WORD.findall(text.casefold()))


def patient_sentences(text: str) -> tuple[str, ...]:
    """Number patient sentences as upstream does, including its synthetic final sentence."""

    sentences = tuple(part.strip() for part in _SENTENCE.split(text) if part.strip())
    return (
        *sentences,
        "The patient will provide informed consent, and will comply with the trial protocol "
        "without any practical issues.",
    )


def numbered_patient(text: str) -> str:
    return "\n".join(
        f"{index}. {sentence}" for index, sentence in enumerate(patient_sentences(text))
    )


def matching_score(matching: Mapping[str, object]) -> float:
    included = not_included = no_information = excluded = 0
    inclusion = matching.get("inclusion")
    exclusion = matching.get("exclusion")
    if not isinstance(inclusion, Mapping) or not isinstance(exclusion, Mapping):
        raise TrialGPTPaperError("matching output must contain inclusion and exclusion objects")
    for value in inclusion.values():
        if not isinstance(value, list) or len(value) != 3:
            continue
        label = value[2]
        if label == "included":
            included += 1
        elif label == "not included":
            not_included += 1
        elif label == "not enough information":
            no_information += 1
    for value in exclusion.values():
        if isinstance(value, list) and len(value) == 3 and value[2] == "excluded":
            excluded += 1
    score = included / (included + not_included + no_information + TRIALGPT_EPSILON)
    if not_included > 0:
        score -= 1
    if excluded > 0:
        score -= 1
    return score


def aggregation_score(aggregation: Mapping[str, object]) -> float:
    try:
        relevance_value = aggregation["relevance_score_R"]
        eligibility_value = aggregation["eligibility_score_E"]
        relevance = float(relevance_value)  # type: ignore[arg-type]
        eligibility = float(eligibility_value)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise TrialGPTPaperError("aggregation output must contain numeric R and E scores") from exc
    if not math.isfinite(relevance) or not math.isfinite(eligibility):
        raise TrialGPTPaperError("aggregation scores must be finite")
    if not 0 <= relevance <= 100 or not -relevance <= eligibility <= relevance:
        raise TrialGPTPaperError("aggregation scores violate TrialGPT's R/E bounds")
    return (relevance + eligibility) / 100


def final_score(matching: Mapping[str, object], aggregation: Mapping[str, object]) -> float:
    return matching_score(matching) + aggregation_score(aggregation)


def validate_matching_output(
    value: object,
    *,
    kind: str,
    patient_sentence_count: int | None = None,
) -> tuple[str, ...]:
    if patient_sentence_count is not None and (
        isinstance(patient_sentence_count, bool) or patient_sentence_count < 1
    ):
        raise ValueError("patient_sentence_count must be a positive integer or None")
    if not isinstance(value, Mapping):
        return (f"{kind} matching output must be an object",)
    labels = INCLUSION_LABELS if kind == "inclusion" else EXCLUSION_LABELS
    errors: list[str] = []
    for criterion_id, raw in value.items():
        if not isinstance(criterion_id, str) or not criterion_id.isdigit():
            errors.append("criterion keys must be decimal strings")
            continue
        if not isinstance(raw, list) or len(raw) != 3:
            errors.append(f"criterion {criterion_id} must be [reasoning, sentence_ids, label]")
            continue
        if not isinstance(raw[0], str) or not raw[0].strip():
            errors.append(f"criterion {criterion_id} reasoning must be a non-empty string")
        if not isinstance(raw[1], list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw[1]
        ):
            errors.append(f"criterion {criterion_id} sentence_ids must be non-negative integers")
        elif patient_sentence_count is not None and any(
            item >= patient_sentence_count for item in raw[1]
        ):
            errors.append(
                f"criterion {criterion_id} sentence_ids must be less than {patient_sentence_count}"
            )
        if not isinstance(raw[2], str) or raw[2] not in labels:
            errors.append(f"criterion {criterion_id} label is invalid for {kind}")
    return tuple(errors)


def _normalize_sentence_ids(value: object) -> object:
    if isinstance(value, list):
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return value
        if all(isinstance(item, str) and item.isascii() and item.isdigit() for item in value):
            return [int(item) for item in value]
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return value
        return _normalize_sentence_ids(decoded)
    tokens = [token.strip() for token in text.split(",")]
    if tokens and all(token.isascii() and token.isdigit() for token in tokens):
        return [int(token) for token in tokens]
    return value


def normalize_matching_output(value: object, *, kind: Literal["inclusion", "exclusion"]) -> object:
    """Canonicalize only unambiguous legacy criterion tuple encodings.

    TrialGPT's canonical tuple is ``[reasoning, sentence_ids, label]``. Some providers return the
    last two fields in the opposite order even when the prompt and schema are explicit. Recover
    that transposition only when the second value is an allowed label and the third value is an
    unambiguous sentence-ID encoding. Ambiguous values remain unchanged for semantic validation.
    """

    if not isinstance(value, Mapping):
        return value
    criteria_candidate = value.get("criteria")
    selected: Mapping[object, object] = (
        criteria_candidate if isinstance(criteria_candidate, Mapping) else value
    )
    normalized_criteria: dict[object, object] = {}
    allowed_labels = INCLUSION_LABELS if kind == "inclusion" else EXCLUSION_LABELS
    for criterion_id, raw in selected.items():
        if isinstance(raw, list) and len(raw) == 3:
            normalized_raw = list(raw)
            transposed_sentence_ids = _normalize_sentence_ids(raw[2])
            if (
                isinstance(raw[1], str)
                and raw[1] in allowed_labels
                and isinstance(transposed_sentence_ids, list)
                and all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in transposed_sentence_ids
                )
            ):
                normalized_raw[1] = transposed_sentence_ids
                normalized_raw[2] = raw[1]
                normalized_criteria[criterion_id] = normalized_raw
                continue
            normalized_raw[1] = _normalize_sentence_ids(raw[1])
            normalized_criteria[criterion_id] = normalized_raw
        else:
            normalized_criteria[criterion_id] = raw
    if selected is value:
        return normalized_criteria
    normalized = dict(value)
    normalized["criteria"] = normalized_criteria
    return normalized


def validate_aggregation_output(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("aggregation output must be an object",)
    errors: list[str] = []
    for name in ("relevance_explanation", "eligibility_explanation"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            errors.append(f"{name} must be a non-empty string")
    try:
        relevance = float(value["relevance_score_R"])
        eligibility = float(value["eligibility_score_E"])
    except (KeyError, TypeError, ValueError):
        errors.append("relevance_score_R and eligibility_score_E must be numbers")
    else:
        if not math.isfinite(relevance) or not math.isfinite(eligibility):
            errors.append("aggregation scores must be finite")
        elif not 0 <= relevance <= 100 or not -relevance <= eligibility <= relevance:
            errors.append("aggregation scores violate R/E bounds")
    return tuple(errors)


def matching_prompt(view: TrialGPTTrialView, kind: str, patient: str) -> str:
    if kind not in {"inclusion", "exclusion"}:
        raise TrialGPTPaperError("matching kind must be inclusion or exclusion")
    criteria = view.inclusion if kind == "inclusion" else view.exclusion
    criterion_ids = ", ".join(str(index) for index in range(len(criteria)))
    system = (
        "You are a helpful assistant for clinical trial recruitment. Your task is to compare a "
        f"given patient note and the {kind} criteria of a clinical trial to determine the "
        "patient's eligibility at the criterion level.\n"
    )
    if kind == "inclusion":
        system += (
            "The factors that allow someone to participate in a clinical study are called "
            "inclusion criteria. They are based on characteristics such as age, gender, the "
            "type and stage of a disease, previous treatment history, and other medical "
            "conditions.\n"
        )
    else:
        system += (
            "The factors that disqualify someone from participating are called exclusion "
            "criteria. They are based on characteristics such as age, gender, the type and "
            "stage of a disease, previous treatment history, and other medical conditions.\n"
        )
    system += (
        f"You should check the {kind} criteria one-by-one, and output the following three "
        "elements for each criterion:\n"
        f"There are exactly {len(criteria)} {kind} criteria. Return exactly one object key for "
        f"each criterion ID in {{{criterion_ids}}}; do not invent any other criterion IDs.\n"
        f"\tElement 1. For each {kind} criterion, briefly generate your reasoning process: "
        "First, judge whether the criterion is not applicable (not very common), where the "
        "patient does not meet the premise of the criterion. Then, check if the patient note "
        "contains direct evidence. If so, judge whether the patient meets or does not meet the "
        "criterion. If there is no direct evidence, try to infer from existing evidence, and "
        "answer one question: If the criterion is true, is it possible that a good patient note "
        "will miss such information? If impossible, then you can assume that the criterion is "
        "not true. Otherwise, there is not enough information.\n"
        "\tElement 2. If there is relevant information, you must generate a list of relevant "
        "sentence IDs in the patient note. The sentence_ids value must be a JSON array of "
        "non-negative integers, for example [0, 12]. Do not return quoted numbers, sentence "
        "text, comma-separated strings, or prose in this value. If there is no relevant "
        "information, you must annotate an empty list.\n"
        f"\tElement 3. Classify the patient eligibility for this specific {kind} criterion: "
    )
    if kind == "inclusion":
        system += (
            'the label must be chosen from {"not applicable", "not enough information", '
            '"included", "not included"}. "not applicable" should only be used for criteria '
            "that are not applicable to the patient. "
            '"not enough information" should be used where the patient note does not contain '
            "sufficient information for making the classification. Try to use as less "
            '"not enough information" as possible because if the note does not mention a '
            "medically important fact, you can assume that the fact is not true for the patient. "
            '"included" denotes that the patient meets the inclusion criterion, while '
            '"not included" means the reverse.\n'
        )
    else:
        system += (
            'the label must be chosen from {"not applicable", "not enough information", '
            '"excluded", "not excluded"}. "not applicable" should only be used for criteria '
            "that are not applicable to the patient. "
            '"not enough information" should be used where the patient note does not contain '
            "sufficient information for making the classification. Try to use as less "
            '"not enough information" as possible because if the note does not mention a '
            "medically important fact, you can assume that the fact is not true for the patient. "
            '"excluded" denotes that the patient meets the exclusion criterion and should be '
            'excluded in the trial, while "not excluded" means the reverse.\n'
        )
    system += (
        "You should output only a JSON dict exactly formatted as: "
        "dict{str(criterion_number): list[str(element_1_brief_reasoning), "
        "list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."
    )
    trial = (
        f"Title: {view.brief_title}\n"
        f"Target diseases: {', '.join(view.diseases)}\n"
        f"Interventions: {', '.join(view.interventions)}\n"
        f"Summary: {view.brief_summary}\n"
        f"{kind.title()} criteria:\n "
        + "".join(f"{index}. {criterion}\n" for index, criterion in enumerate(criteria))
        + "\n"
    )
    user = (
        "Here is the patient note, each sentence is led by a sentence_id:\n"
        f"{patient}\n\nHere is the clinical trial:\n{trial}\n\nPlain JSON output:"
    )
    return f"{system}\n\n{user}"


def aggregation_prompt(
    view: TrialGPTTrialView, patient: str, matching: Mapping[str, object]
) -> str:
    def criteria_text(kind: str, predictions: object) -> str:
        if not isinstance(predictions, Mapping):
            raise TrialGPTPaperError(f"{kind} matching output must be an object")
        criteria = view.inclusion if kind == "inclusion" else view.exclusion
        output = ""
        for criterion_id, prediction in predictions.items():
            if not isinstance(criterion_id, str) or not criterion_id.isdigit():
                continue
            index = int(criterion_id)
            if index >= len(criteria) or not isinstance(prediction, list) or len(prediction) != 3:
                continue
            output += f"{kind} criterion {index}: {criteria[index]}\n"
            output += f"\tPatient relevance: {prediction[0]}\n"
            if isinstance(prediction[1], list) and prediction[1]:
                output += f"\tEvident sentences: {prediction[1]}\n"
            output += f"\tPatient eligibility: {prediction[2]}\n"
        return output

    trial = (
        f"Title: {view.brief_title}\n"
        f"Target conditions: {', '.join(view.diseases)}\n"
        f"Summary: {view.brief_summary}"
    )
    prediction_text = criteria_text("inclusion", matching.get("inclusion")) + criteria_text(
        "exclusion", matching.get("exclusion")
    )
    system = (
        "You are a helpful assistant for clinical trial recruitment. You will be given a patient "
        "note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
        "Your task is to output two scores, a relevance score (R) and an eligibility score (E), "
        "between the patient and the clinical trial.\n"
        "First explain the consideration for determining patient-trial relevance. Predict the "
        "relevance score R (0~100), which represents the overall relevance between the patient "
        "and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical "
        "trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
        "Then explain the consideration for determining patient-trial eligibility. Predict the "
        "eligibility score E (-R~R), which represents the patient's eligibility to the clinical "
        "trial. Note that -R <= E <= R, where E=-R denotes that the patient is ineligible, "
        "E=R denotes that the patient is eligible, and E=0 denotes that the patient is neutral.\n"
        'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, '
        '"relevance_score_R": Float, "eligibility_explanation": Str, '
        '"eligibility_score_E": Float}. '
    )
    user = (
        f"Here is the patient note:\n{patient}\n\n"
        f"Here is the clinical trial description:\n{trial}\n\n"
        "Here are the criterion-level eligibility prediction:\n"
        f"{prediction_text}\n\nPlain JSON output:"
    )
    return f"{system}\n\n{user}"


__all__ = [
    "TRIALGPT_LLM_CANDIDATE_DEPTH",
    "TRIALGPT_TOKENIZER_ID",
    "TrialGPTPaperError",
    "TrialGPTTrialView",
    "aggregation_prompt",
    "final_score",
    "matching_prompt",
    "matching_score",
    "normalize_matching_output",
    "numbered_patient",
    "patient_sentences",
    "trial_view",
    "validate_aggregation_output",
    "validate_matching_output",
]
