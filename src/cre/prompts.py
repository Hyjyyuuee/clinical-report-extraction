"""Prompt templates shared by API inference, local inference and SFT data building."""

import json

from .schema import FindingType

SYSTEM = (
    "You are a clinical information extraction system. Read a chest X-ray report and "
    "return every radiological finding as JSON. Mark each finding as present, absent "
    "(explicitly negated) or uncertain (hedged language). Copy the supporting text "
    "exactly into 'evidence'. Do not infer findings that the report does not mention. "
    "Rules:\n"
    "- Extract a finding only when the report names it or an unambiguous synonym "
    "(e.g. 'effusion' = pleural_effusion, 'heart size is normal' = cardiomegaly absent).\n"
    "- Do not expand general statements such as 'lungs are clear' or 'no acute "
    "cardiopulmonary abnormality' into separate absent findings.\n"
    "- A normal heart or cardiomediastinal silhouette ('within normal limits', 'unremarkable', "
    "'upper limits of normal') is cardiomegaly absent; 'stable' alone is not; 'borderline' is uncertain.\n"
    "- 'Airspace disease' and 'airspace opacity' are opacity; use consolidation only when that word is used.\n"
    "- Calcified granulomas and prior granulomatous disease are granuloma.\n"
    "- Hyperinflated lungs are emphysema only if emphysema or COPD is named. Low lung volumes "
    "or bronchovascular crowding are not atelectasis.\n"
    "- Differentials ('A versus B', 'A or B', 'may represent', 'cannot be excluded') are uncertain. "
    "If a finding is stated and only its cause is hedged, the finding is present.\n"
    "- Set laterality whenever the side is stated for that finding; 'bibasilar' and 'bilateral' "
    "are bilateral, the lingula is left.\n"
    "- Use 'other' only for lung, pleural or cardiac findings not in the list; ignore "
    "bones, devices and anything outside the chest.\n"
    "Return only JSON with the shape {\"findings\": [...]}.\n"
    f"Allowed finding values: {', '.join(f.value for f in FindingType)}."
)

FEW_SHOT = [
    (
        "FINDINGS: Patchy opacity in the right lower lobe. No pleural effusion or pneumothorax. "
        "IMPRESSION: Right lower lobe opacity, pneumonia cannot be excluded.",
        {
            "findings": [
                {"finding": "opacity", "status": "present", "laterality": "right",
                 "location": "right lower lobe", "evidence": "Patchy opacity in the right lower lobe"},
                {"finding": "pleural_effusion", "status": "absent", "laterality": None,
                 "location": None, "evidence": "No pleural effusion"},
                {"finding": "pneumothorax", "status": "absent", "laterality": None,
                 "location": None, "evidence": "or pneumothorax"},
                {"finding": "pneumonia", "status": "uncertain", "laterality": "right",
                 "location": "right lower lobe", "evidence": "pneumonia cannot be excluded"},
            ]
        },
    ),
]


def format_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings or 'N/A'}\nIMPRESSION: {impression or 'N/A'}"


def build_messages(report: str, few_shot: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for example_report, example_output in FEW_SHOT:
            messages.append({"role": "user", "content": example_report})
            messages.append({"role": "assistant", "content": json.dumps(example_output)})
    messages.append({"role": "user", "content": report})
    return messages
