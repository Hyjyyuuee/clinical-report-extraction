"""Label schema for structured findings extracted from chest X-ray reports."""

from enum import Enum

from pydantic import BaseModel, Field


class FindingType(str, Enum):
    consolidation = "consolidation"
    infiltrate = "infiltrate"
    opacity = "opacity"
    pleural_effusion = "pleural_effusion"
    atelectasis = "atelectasis"
    pneumothorax = "pneumothorax"
    edema = "edema"
    cardiomegaly = "cardiomegaly"
    nodule = "nodule"
    granuloma = "granuloma"  # calcified granulomas are benign and very common in IU X-Ray
    emphysema = "emphysema"
    pneumonia = "pneumonia"
    other = "other"


class Status(str, Enum):
    present = "present"
    absent = "absent"        # explicitly negated, e.g. "no focal consolidation"
    uncertain = "uncertain"  # hedged, e.g. "cannot exclude pneumonia"


class Laterality(str, Enum):
    left = "left"
    right = "right"
    bilateral = "bilateral"


class Finding(BaseModel):
    finding: FindingType
    status: Status
    laterality: Laterality | None = None
    location: str | None = Field(None, description="Free-text anatomical location, e.g. 'right lower lobe'.")
    evidence: str = Field(description="Exact span from the report supporting this finding.")


class ReportExtraction(BaseModel):
    findings: list[Finding]


def json_schema() -> dict:
    return ReportExtraction.model_json_schema()
