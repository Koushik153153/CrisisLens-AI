"""Structured future NLP output; no extraction or scoring logic.

None means not yet detected, including counts, scores and action_required.
Timestamp is an optional ISO 8601 string describing the report time.
Use dataclasses.asdict(report), then json.dumps(...) for serialization.
"""

from dataclasses import dataclass, field
from typing import Generic, Literal, Optional, TypeVar

from emergency_taxonomy import (
    Actionability, ActionType, IncidentType, PriorityLevel, ResourceNeed,
    Severity, Urgency, VulnerableGroup,
)


AssessmentLabel = TypeVar("AssessmentLabel", Urgency, Severity, Actionability)


@dataclass
class DimensionAssessment(Generic[AssessmentLabel]):
    label: Optional[AssessmentLabel]
    confidence: float
    evidence: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    semantic_scores: dict[str, float] = field(default_factory=dict)
    cue_label: Optional[str] = None


@dataclass
class AssessmentResult:
    urgency: DimensionAssessment[Urgency]
    severity: DimensionAssessment[Severity]
    actionability: DimensionAssessment[Actionability]


@dataclass(frozen=True)
class ResourcePrediction:
    """Resource evidence; confidence is a rule score or semantic similarity."""

    resource: ResourceNeed
    confidence: float
    inference_type: Literal["EXPLICIT", "IMPLICIT"]
    evidence: str
    reason: str


@dataclass
class PriorityComponent:
    label: Optional[str]
    raw_value: float
    weight: float
    points: float
    details: list[str] = field(default_factory=list)


@dataclass
class PriorityResult:
    priority_score: Optional[float]
    priority_level: Optional[PriorityLevel]
    component_scores: dict[str, PriorityComponent] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    available: bool = False


@dataclass
class EmergencyReport:
    """One report and its optional analysis; list defaults are per instance."""

    report_id: str = ""
    source_file: str = ""
    raw_text: str = ""
    timestamp: Optional[str] = None

    incident_type: Optional[IncidentType] = None
    locations: list[str] = field(default_factory=list)
    affected_people: Optional[int] = None
    vulnerable_groups: list[VulnerableGroup] = field(default_factory=list)

    resource_needs: list[ResourceNeed] = field(default_factory=list)
    action_required: Optional[bool] = None

    urgency: Optional[Urgency] = None
    severity: Optional[Severity] = None
    actionability: Optional[Actionability] = None
    priority_score: Optional[float] = None
    priority_level: Optional[PriorityLevel] = None
    priority_reasons: list[str] = field(default_factory=list)
    summary: Optional[str] = None

    # Kept separate from the Step-1 boolean; multiple actions can be needed.
    actions: list[ActionType] = field(default_factory=list)

    # Step 3 diagnostics; similarity is not a calibrated probability.
    incident_confidence: Optional[float] = None
    incident_category_scores: dict[str, float] = field(default_factory=dict)
    # affected_people remains individuals only; no conversion of family counts.
    affected_count: Optional[int] = None
    affected_unit: Optional[Literal["PERSONS", "FAMILIES", "HOUSEHOLDS"]] = None
    affected_count_approximate: Optional[bool] = None
    resource_predictions: list[ResourcePrediction] = field(default_factory=list)
    assessment: Optional[AssessmentResult] = None
    priority_explanation: Optional[PriorityResult] = None
