"""Three independent evidence-based assessments; no combined priority score.

Injected MiniLM supplies semantic similarities. Transparent cues add support to
one label per dimension. Confidence is raw selected-label cosine, not probability.
"""

import re

from config import ASSESSMENT_CUE_BONUS, ASSESSMENT_LARGE_PERSON_COUNT
from embedder import Embedder
from emergency_schema import AssessmentResult, DimensionAssessment, EmergencyReport
from emergency_taxonomy import Urgency, Severity, Actionability


PROTOTYPES = {
    "urgency": {
        "LOW": ("Routine monitoring is sufficient and no intervention is needed now.",
                "The incident is resolved and conditions have stabilized."),
        "MEDIUM": ("A minor problem needs attention during the next work round.",
                   "Non-emergency assistance can be arranged without immediate dispatch."),
        "HIGH": ("Prompt assistance is needed today before conditions deteriorate.",
                 "People need evacuation or essential supplies before tonight."),
        "CRITICAL": ("Immediate rescue is necessary because lives are in danger right now.",
                     "A person is unresponsive and emergency help is needed without delay."),
    },
    "severity": {
        "LOW": ("There is no damage or injury and conditions are normal.",
                "The incident caused only minor superficial cuts or a small disruption."),
        "MEDIUM": ("A local access disruption affects services but no serious harm is reported.",
                   "Basic supplies are missing but people are currently safe."),
        "HIGH": ("People have serious injuries or have lost their homes.",
                 "Dangerous conditions threaten residents and have caused substantial damage."),
        "CRITICAL": ("People face immediate life-threatening conditions or fatalities are reported.",
                     "Occupants are trapped under rubble or in a burning building."),
    },
    "actionability": {
        "LOW": ("There may be bad weather somewhere, but no concrete intervention is identified.",
                "This is a routine observation with no request for a response."),
        "MEDIUM": ("A serious event needs attention but the location or operational details are unclear.",
                   "A warning gives a general response direction but local checks are still needed."),
        "HIGH": ("A specific location and clear request identify a concrete response to perform.",
                 "The report identifies affected people and the help required at a known place."),
    },
}


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def _evidence(sentences, pattern, negate=True):
    """Keep source sentences; suppress short local negation, not whole reports."""
    found = []
    for sentence in sentences:
        for match in re.finditer(pattern, sentence, re.I):
            prefix = re.split(r"[,;:]|\bbut\b", sentence[:match.start()], flags=re.I)[-1]
            if negate and re.search(r"\b(?:no|not|nobody|without)\b(?:\W+\w+){0,3}\W*$", prefix, re.I):
                continue
            found.append(sentence)
            break
    return found


class EmergencyAssessmentEngine:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._keys = [(dimension, label) for dimension, labels in PROTOTYPES.items()
                      for label, sentences in labels.items() for _ in sentences]
        self._vectors = embedder.embed_batch([sentence for labels in PROTOTYPES.values()
                                             for sentences in labels.values() for sentence in sentences])

    def assess(self, text: str, context: EmergencyReport | None = None) -> AssessmentResult:
        """Read Step-3/4 context without modifying it or reading existing triage labels."""
        if not text.strip():
            def unknown():
                return DimensionAssessment(None, 0.0, [], ["No report text was supplied; assessment remains unknown."])
            return AssessmentResult(unknown(), unknown(), unknown())
        context = context or EmergencyReport(raw_text=text)
        sentences = _sentences(text)
        similarities = self.embedder.cosine_similarity_matrix([self.embedder.embed(text)], self._vectors)[0]
        scores = {dimension: {label: float(max(value for key, value in zip(self._keys, similarities)
                                              if key == (dimension, label)))
                              for label in labels} for dimension, labels in PROTOTYPES.items()}

        trapped = _evidence(sentences, r"\btrapped\b|cannot escape|cannot get down|buried beneath")
        worsening = _evidence(sentences, r"rapidly rising|still climbing|water continues to rise|flames spreading|minutes remaining")
        distress = _evidence(sentences, r"\bunconscious\b|\bunresponsive\b") + _evidence(sentences, r"not responding|not breathing", negate=False)
        serious = _evidence(sentences, r"serious injuries|seriously injured|bleeding heavily|breathlessness|struggling to breathe")
        minor = _evidence(sentences, r"minor|superficial cuts|small disruption|non-dangerous")
        immediate = _evidence(sentences, r"\bimmediate(?:ly)?\b|without delay|right now")
        prompt = _evidence(sentences, r"\burgent\b|\btoday\b|\btonight\b|before (?:evening|night)|\bpromptly\b")
        monitoring = _evidence(sentences, r"routine|monitoring.only|no (?:further|immediate|additional) (?:dispatch|intervention|assistance)|no.*request for assistance", negate=False)
        resolved = _evidence(sentences, r"fully extinguished|power is back|stabilized|resolved|conditions are normal", negate=False)
        fatalities = _evidence(sentences, r"\b(?:fatalities|deaths|killed)\b")
        structural = _evidence(sentences, r"(?:house|building|roof|wall).*collaps|roof.*(?:torn|fell)|lost their homes")
        support_failure = _evidence(sentences, r"respiratory support|life.support") if worsening else []
        active = trapped + distress + serious + worsening
        calm = (monitoring or resolved) and not active and not context.resource_needs

        # Urgency: intervention timing, never derived from severity's label.
        if distress or support_failure or (trapped and (worsening or immediate)) or (serious and immediate):
            urgency = ("CRITICAL", distress + support_failure + trapped + worsening + immediate,
                       "Current distress, worsening entrapment or an immediate intervention cue supports time-critical response.")
        elif calm:
            urgency = ("LOW", monitoring + resolved, "The report describes monitoring or resolution without an identified active intervention need.")
        elif trapped or serious or prompt:
            urgency = ("HIGH", trapped + serious + prompt, "Entrapment, medical distress or a near-term deadline supports prompt intervention.")
        elif minor or _evidence(sentences, r"next work round|non.immediate"):
            urgency = ("MEDIUM", minor + _evidence(sentences, r"next work round|non.immediate"), "The stated assistance can be handled without immediate emergency dispatch.")
        else:
            urgency = None

        # Severity: consequences/conditions, never derived from urgency's label.
        if fatalities or distress or support_failure or (trapped and (structural or worsening)):
            severity = ("CRITICAL", fatalities + distress + support_failure + trapped + structural + worsening,
                        "Reported fatality, loss of responsiveness, life-support risk or dangerous entrapment supports critical seriousness.")
        elif serious or structural or trapped:
            severity = ("HIGH", serious + structural + trapped, "Serious injury, structural damage or entrapment supports elevated seriousness.")
        elif calm or minor:
            severity = ("LOW", monitoring + resolved + minor, "The report describes limited harm, minor injury or stabilized conditions.")
        else:
            severity = None
        # Person totals never determine seriousness on their own, and family counts
        # are not treated as persons. Context adds documented support only with harm.
        if severity is None and active and context.affected_unit == "PERSONS" and (
            context.affected_count is not None and context.affected_count >= ASSESSMENT_LARGE_PERSON_COUNT
        ):
            severity = ("HIGH", active, "Active harm accompanies a large explicitly extracted individual-person count.")

        # Actionability: operational clarity, independent of both labels above.
        requests = _evidence(sentences, r"\b(?:please|request\w*|require\w*|need\w*)\b")
        vague = _evidence(sentences, r"\bsomewhere\b|unknown location|location.*(?:unclear|unknown)|\brumou?r\b|checks.*pending")
        loc_evidence = [s for s in sentences if any(loc.lower() in s.lower() for loc in context.locations)]
        resource_evidence = [p.evidence for p in context.resource_predictions if p.evidence in text]
        operational = requests + active + resource_evidence
        if calm or (not operational and not context.resource_needs):
            actionability = ("LOW", monitoring + resolved or sentences, "No current concrete intervention is supported by the available text/context.")
        elif context.locations and (operational or context.resource_needs) and not vague:
            actionability = ("HIGH", loc_evidence + operational or sentences, "An extracted location and concrete intervention evidence support a specific response.")
        else:
            actionability = ("MEDIUM", vague + operational or sentences, "Response-relevant information is present, but location or operational scope needs clarification.")

        def finish(dimension, enum, cue):
            raw = scores[dimension]
            ranked = {label: value + (ASSESSMENT_CUE_BONUS if cue and label == cue[0] else 0)
                      for label, value in raw.items()}
            # HIGH actionability requires operational/location evidence regardless
            # of semantic descriptions of danger; this is not a severity cap.
            if dimension == "actionability" and cue[0] != "HIGH":
                ranked.pop("HIGH")
            label = max(ranked, key=ranked.get)
            evidence = list(dict.fromkeys(cue[1] if cue else sentences))
            reasons = [f"Semantic evidence was compared with independent {dimension} prototypes."]
            if cue:
                reasons.append(cue[2])
            if dimension == "severity" and context.vulnerable_groups and active:
                reasons.append("Extracted vulnerable groups are reported alongside active danger; their presence alone does not determine the level.")
            return DimensionAssessment(enum(label), raw[label], evidence, reasons, raw, cue[0] if cue else None)

        return AssessmentResult(finish("urgency", Urgency, urgency), finish("severity", Severity, severity),
                                finish("actionability", Actionability, actionability))
