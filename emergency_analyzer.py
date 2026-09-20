"""Step 3: semantic incident classification and constrained entity extraction.

Optional injected resource analysis; no triage, priority, API calls or model construction.
The caller owns the existing Embedder and injects it into EmergencyAnalyzer.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resource_intelligence import ResourceIntelligence
    from emergency_assessment import EmergencyAssessmentEngine
    from priority_engine import EmergencyPriorityEngine

from config import INCIDENT_CONFIDENCE_THRESHOLD
from embedder import Embedder
from emergency_schema import EmergencyReport
from emergency_taxonomy import IncidentType, VulnerableGroup


# General descriptions, not copies of corpus reports. OTHER is a rejection class.
INCIDENT_PROTOTYPES = {
    IncidentType.FLOOD: (
        "Flood water has inundated homes and streets, stranding residents.",
        "A river overflow has cut off a village and people need evacuation by boat.",
        "Families displaced by flooding are staying in a relief camp.",
    ),
    IncidentType.FIRE: (
        "A residential building is burning with flames and thick smoke.",
        "An electrical fire has broken out inside a house and occupants are trapped.",
        "Firefighters have extinguished a small fire and are checking the building.",
    ),
    IncidentType.MEDICAL_EMERGENCY: (
        "A person has collapsed unconscious and needs immediate medical attention.",
        "A seriously ill patient is struggling to breathe and needs hospital care.",
        "A patient has missed essential treatment and needs urgent medical help.",
    ),
    IncidentType.ROAD_ACCIDENT: (
        "Vehicles collided in a road crash and passengers were injured.",
        "A motorcycle collision has left riders hurt beside the road.",
        "A bus accident has injured travellers who need medical transport.",
    ),
    IncidentType.BUILDING_COLLAPSE: (
        "A building has collapsed and workers are trapped under rubble.",
        "The walls and roof of a house fell down, displacing its residents.",
        "A structural failure has left people beneath fallen concrete and debris.",
    ),
    IncidentType.LANDSLIDE: (
        "A landslide sent rocks and soil down a hillside onto a road.",
        "A slope has collapsed, blocking access to a mountain settlement.",
        "A mudslide has buried the hillside route beneath earth and stones.",
    ),
    IncidentType.CYCLONE_STORM: (
        "A cyclone warning forecasts damaging winds along the coast.",
        "A violent storm has torn roofs from homes and displaced families.",
        "Strong squall winds are damaging buildings in a coastal town.",
    ),
    IncidentType.POWER_OUTAGE: (
        "A power outage has interrupted electricity to homes and services.",
        "Mains electricity has failed and backup power for medical equipment is running out.",
        "Electricity has been restored after a feeder fault caused a blackout.",
    ),
    IncidentType.ROAD_BLOCKAGE: (
        "A fallen tree is blocking a road and vehicles must take a diversion.",
        "Construction debris obstructs the road and prevents vehicle access.",
        "An obstruction has closed the access route and requires clearance.",
    ),
}

# Exact place strings observed in the synthetic TXT corpus. Bare place names
# are also recognized as themselves, without inventing a district/city suffix.
CORPUS_LOCATIONS = (
    "Velachery, Chennai", "Mudichur, Chengalpattu", "Perumbakkam, Chennai",
    "Tondiarpet, Chennai", "Tambaram, Chennai", "Thiruvanmiyur, Chennai",
    "Cuddalore, Tamil Nadu", "Guindy, Chennai", "Porur, Chennai",
    "Saidapet, Chennai", "Kanchipuram, Tamil Nadu", "Coonoor, Nilgiris",
    "Nagapattinam, Tamil Nadu", "Rameswaram, Ramanathapuram",
    "Royapuram, Chennai", "Madurai, Tamil Nadu", "Adyar, Chennai",
    "Tiruvallur, Tamil Nadu", "Marina Beach, Chennai", "Poondi, Tiruvallur",
)
LOCATION_GAZETTEER = tuple(dict.fromkeys(
    list(CORPUS_LOCATIONS) + [part for place in CORPUS_LOCATIONS for part in place.split(", ")]
))


@dataclass(frozen=True)
class IncidentPrediction:
    label: IncidentType
    confidence: float
    category_scores: dict[str, float]


@dataclass(frozen=True)
class AffectedCount:
    count: int | None = None
    unit: str | None = None
    approximate: bool | None = None


_SMALL_NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
_NUMBER_WORD = "(?:" + "|".join(_TENS) + ")(?:[- ](?:" + "|".join(list(_SMALL_NUMBERS)[1:10]) + "))?"
_NUMBER = r"(?:\d+(?:,\d{3})*|" + _NUMBER_WORD + "|" + "|".join(_SMALL_NUMBERS) + ")"
_COUNT_PATTERN = re.compile(
    r"\b(?P<approx>around\s+|about\s+|approximately\s+)?(?P<number>" + _NUMBER + r")\s+"
    r"(?:(?:elderly|pregnant|injured|displaced|stranded|trapped|affected|chronically\s+ill)\s+)*"
    r"(?P<noun>people|persons?|residents?|passengers?|workers?|occupants?|riders?|"
    r"patients?|children|kids?|adults?|women|woman|men|man|families|family|households?)\b",
    re.IGNORECASE,
)


def _number_value(value: str) -> int:
    if value[0].isdigit():
        return int(value.replace(",", ""))
    return sum(_SMALL_NUMBERS.get(w, _TENS.get(w, 0)) for w in re.split(r"[- ]", value.lower()))


def extract_affected_count(text: str) -> AffectedCount:
    """Return the first explicit non-subgroup count, never sum mentions.

    Counts after 'including'/'of whom' are subgroup evidence, not new totals.
    First-mention selection is a documented heuristic, not coreference resolution.
    Only explicit empty occupancy PLUS absence of injury supports implicit zero.
    No-injury language on its own says nothing about total affected people.
    """
    for match in _COUNT_PATTERN.finditer(text):
        clause_prefix = re.split(r"[.!?;:]", text[:match.start()])[-1].lower()
        if re.search(r"\b(including|of whom|among them)\b", clause_prefix):
            continue
        # Reject counts negated or framed as a denied claim.
        if re.search(r"\b(?:not|no)\s*$", clause_prefix):
            continue
        noun = match["noun"].lower()
        unit = "FAMILIES" if noun in {"family", "families"} else "HOUSEHOLDS" if noun.startswith("household") else "PERSONS"
        return AffectedCount(_number_value(match["number"]), unit, bool(match["approx"]))
    if re.search(r"\b(?:nobody|no one) was inside\b", text, re.I) and re.search(
        r"\b(?:nobody|no one) was injured\b", text, re.I
    ):
        return AffectedCount(0, "PERSONS", False)
    return AffectedCount()


_GROUP_PATTERNS = {
    VulnerableGroup.CHILDREN: r"\b(?:children|child|kids?|infants?|babies)\b",
    VulnerableGroup.ELDERLY: r"\b(?:elderly|senior citizens?|older adults?)\b",
    VulnerableGroup.PREGNANT: r"\bpregnant\b",
    VulnerableGroup.INJURED: r"\b(?:injured|injuries|injury|cuts|wounds?|bleeding)\b",
    VulnerableGroup.DISABLED: r"\b(?:wheelchair|disabled|disabilities|disability)\b",
    VulnerableGroup.CHRONICALLY_ILL: r"\b(?:chronically ill|chronic illness|regular dialysis|dialysis)\b",
}


def extract_vulnerable_groups(text: str) -> list[VulnerableGroup]:
    """Lexical mapping with local negation suppression; not medical diagnosis."""
    mentions = []
    for group, pattern in _GROUP_PATTERNS.items():
        for match in re.finditer(pattern, text, re.I):
            prefix = re.split(r"[.!?;:]|\bbut\b", text[:match.start()], flags=re.I)[-1]
            # Bound negation to the current short clause to avoid suppressing a
            # later affirmative sentence. Handles 'nobody was struck or injured'.
            if re.search(r"\b(?:no|not|nobody|neither|without|no one)\b(?:\W+\w+){0,5}\W*$", prefix, re.I):
                continue
            suffix = text[match.end():]
            if re.match(r"\s+(?:were|was|are|is)\s+(?:not|absent)\b", suffix, re.I):
                continue
            mentions.append((match.start(), group))
            break
    return list(dict.fromkeys(group for _, group in sorted(mentions)))


def extract_locations(text: str) -> list[str]:
    """Case-insensitive gazetteer matches, longest overlap first, textual order."""
    matches = []
    for location in LOCATION_GAZETTEER:
        pattern = r",\s*".join(
            r"\s+".join(re.escape(word) for word in part.split())
            for part in location.split(", ")
        )
        for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text, re.I):
            matches.append((match.start(), match.end(), location))
    selected = []
    for start, end, location in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0])):
        if not any(start < e and end > s for s, e, _ in selected):
            selected.append((start, end, location))
    return list(dict.fromkeys(location for _, _, location in sorted(selected)))


class EmergencyAnalyzer:
    """Owns prototype vectors only; caller supplies the existing model wrapper."""

    def __init__(self, embedder: Embedder, threshold: float = INCIDENT_CONFIDENCE_THRESHOLD,
                 resource_intelligence: "ResourceIntelligence | None" = None,
                 assessment_engine: "EmergencyAssessmentEngine | None" = None,
                 priority_engine: "EmergencyPriorityEngine | None" = None):
        if not -1 <= threshold <= 1:
            raise ValueError("Cosine threshold must be between -1 and 1")
        self.embedder = embedder
        self.threshold = threshold
        self.resource_intelligence = resource_intelligence
        self.assessment_engine = assessment_engine
        self.priority_engine = priority_engine
        self._labels = [label for label, sentences in INCIDENT_PROTOTYPES.items() for _ in sentences]
        self._prototype_embeddings = embedder.embed_batch([
            sentence for sentences in INCIDENT_PROTOTYPES.values() for sentence in sentences
        ])

    def classify_incident(self, text: str) -> IncidentPrediction:
        if not text.strip():
            return IncidentPrediction(IncidentType.OTHER, 0.0, {})
        embedding = self.embedder.embed(text)
        similarities = self.embedder.cosine_similarity_matrix([embedding], self._prototype_embeddings)[0]
        # Maximum allows a report to match one distinct subtype (e.g. displaced
        # flood survivors) without requiring similarity to every prototype.
        scores = {
            label.value: float(max(score for key, score in zip(self._labels, similarities) if key == label))
            for label in INCIDENT_PROTOTYPES
        }
        strongest = max(scores, key=scores.get)
        confidence = scores[strongest]
        label = IncidentType(strongest) if confidence >= self.threshold else IncidentType.OTHER
        return IncidentPrediction(label, confidence, scores)

    def analyze(self, text: str, report_id: str | None = None, source_file: str | None = None) -> EmergencyReport:
        prediction = self.classify_incident(text)
        count = extract_affected_count(text)
        report = EmergencyReport(
            report_id=report_id or "", source_file=source_file or "", raw_text=text,
            incident_type=prediction.label, incident_confidence=prediction.confidence,
            incident_category_scores=prediction.category_scores,
            locations=extract_locations(text),
            affected_people=count.count if count.unit == "PERSONS" else None,
            affected_count=count.count, affected_unit=count.unit,
            affected_count_approximate=count.approximate,
            vulnerable_groups=extract_vulnerable_groups(text),
        )
        if self.resource_intelligence is not None:
            report.resource_predictions = self.resource_intelligence.analyze_resource_needs(text, report)
            report.resource_needs = [item.resource for item in report.resource_predictions]
        if self.assessment_engine is not None:
            report.assessment = self.assessment_engine.assess(text, report)
            report.urgency = report.assessment.urgency.label
            report.severity = report.assessment.severity.label
            report.actionability = report.assessment.actionability.label
        if self.priority_engine is not None:
            report.priority_explanation = self.priority_engine.score(report)
            report.priority_score = report.priority_explanation.priority_score
            report.priority_level = report.priority_explanation.priority_level
            report.priority_reasons = list(report.priority_explanation.reasons)
        return report
