"""Step-4 multi-label resource inference: request phrases + guarded MiniLM needs.

No model construction, report IDs, gold-file access, dispatch quantities or APIs.
"""

import re

from config import RESOURCE_EXPLICIT_CONFIDENCE, RESOURCE_IMPLICIT_THRESHOLD
from embedder import Embedder
from emergency_schema import EmergencyReport, ResourcePrediction
from emergency_taxonomy import IncidentType, ResourceNeed as R


RESOURCE_PROTOTYPES = {
    R.AMBULANCE: (
        "Seriously injured people need emergency transport to hospital.",
        "An unconscious patient requires immediate medical evacuation.",
        "A person struggling to breathe needs urgent help reaching medical care.",
    ),
    R.MEDICAL_TEAM: (
        "Injured victims require urgent treatment at the scene.",
        "People with wounds need first aid and medical assistance.",
        "A seriously ill person needs a clinician to attend them immediately.",
    ),
    R.RESCUE_TEAM: (
        "People are trapped and cannot escape without assistance.",
        "Stranded residents need rescuers to reach them and bring them to safety.",
        "Workers buried beneath rubble need a team to free them.",
    ),
    R.RESCUE_BOAT: (
        "People stranded by deep floodwater need water-based evacuation.",
        "Flooded access prevents residents from leaving by road or on foot.",
        "Residents surrounded by rising water need a boat to reach dry ground.",
    ),
    R.FIRE_SERVICE: (
        "An active building fire needs firefighters and suppression.",
        "Flames are spreading through a house and must be extinguished.",
        "A burning building filled with smoke requires firefighting help.",
    ),
    R.FOOD: (
        "Displaced families have no meals or food supplies.",
        "Hungry residents need something to eat at the relief camp.",
        "The food delivery has not arrived and families need meals.",
    ),
    R.DRINKING_WATER: (
        "People at a relief camp have no safe drinking water.",
        "Stored water is contaminated and residents need a safe supply.",
        "Water containers are empty and families need potable water delivered.",
    ),
    R.SHELTER: (
        "Displaced residents have nowhere safe to stay.",
        "People whose home is destroyed need somewhere safe to sleep tonight.",
        "Families waiting outside need a covered place to stay overnight.",
    ),
    R.POWER_RESTORATION: (
        "An area has lost electricity and requires utility repair.",
        "Mains power has failed and the backup electrical supply is running out.",
        "A blackout has left homes without electricity and the supply must be restored.",
    ),
    R.ROAD_CLEARANCE: (
        "Debris is blocking a road and must be removed to allow access.",
        "A fallen tree obstructs traffic and crews need to clear the route.",
        "Rocks and soil across the access road prevent vehicles from passing.",
    ),
}

# Resource names and close operational synonyms, not incident keywords.
EXPLICIT_PHRASES = {
    R.AMBULANCE: r"ambulances?",
    R.MEDICAL_TEAM: r"medical team|paramedics?|first[- ]aid team|clinical staff",
    R.RESCUE_TEAM: r"rescue team|rescue personnel|rescue crew|rescuers",
    R.RESCUE_BOAT: r"rescue boat|boat rescue|water[- ]based rescue",
    R.FIRE_SERVICE: r"fire engine|fire brigade|fire service|firefighters?",
    R.FOOD: r"food(?: packets| supplies)?|meals?",
    R.DRINKING_WATER: r"drinking[- ]water|potable water|safe water",
    R.SHELTER: r"(?:temporary )?shelter|relief accommodation",
    R.POWER_RESTORATION: r"restore (?:electricity|power)|power restoration|replacement electrical supply",
    R.ROAD_CLEARANCE: r"(?:road|debris)[- ]clearance|clear (?:the )?road",
}
_REQUEST = r"\b(?:need\w*|requir\w*|request\w*|send|dispatch|arrange|asking|asks for|please|waiting for)\b"


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def _clauses(text: str) -> list[str]:
    # Preserve source spans for explanations, splitting contrast to keep negation local.
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\s+but\s+", text, flags=re.I) if part.strip()]


def _unneeded_or_satisfied(resource: R, clause: str) -> bool:
    aliases = EXPLICIT_PHRASES[resource]
    # Include transport/treatment synonyms when checking explicit denials.
    if resource == R.AMBULANCE:
        aliases += r"|hospital transport|medical transport"
    if resource == R.MEDICAL_TEAM:
        aliases += r"|clinical team|medical assistance|treatment"
    if resource == R.DRINKING_WATER:
        aliases += r"|water"
    if resource == R.SHELTER:
        aliases += r"|accommodation"
    relevant = _has(r"\b(?:" + aliases + r")\b", clause)
    if relevant:
        if _has(r"\b(?:no|not|unnecessary|not needed|not required)\b.*\b(?:requested|required|needed)\b", clause):
            return True
        if _has(r"\b(?:do not|don't)\s+(?:send|dispatch|arrange)\b", clause):
            return True
        if _has(r"\b(?:already|has|have|is|are)\b.*\b(?:supplied|provided|delivered|adequate|ready|attending|on site)\b", clause) and not _has(r"\b(?:not|no|insufficient)\b", clause):
            return True
        if _has(r"\b(?:unnecessary|not needed|not required)\b", clause):
            return True
    if resource == R.FIRE_SERVICE and _has(r"\b(?:extinguished|fire is out|fire was put out)\b", clause):
        return True
    if resource == R.POWER_RESTORATION and _has(r"\b(?:power|electricity)\b.*\b(?:restored|is back|working normally)\b", clause):
        return True
    return False


def _context_reason(resource: R, text: str, incident: IncidentType | None) -> str | None:
    """Eligibility gates supplement (never replace) semantic similarity."""
    people = _has(r"\b(?:people|residents|workers|occupants|passengers|families|person|patients?)\b", text)
    trapped = _has(r"\b(?:trapped|stranded|buried|cannot escape|cannot get down|cannot leave)\b", text)
    serious = _has(r"serious injuries|seriously injured|unconscious|unresponsive|not responding|bleeding heavily|breathlessness|struggling to breathe", text)
    if resource == R.AMBULANCE and (serious or _has(r"(?:need|help|taken).*?(?:hospital|reaching medical care)", text)):
        return "Serious medical distress or an unmet hospital-transport need supports emergency medical transport."
    if resource == R.MEDICAL_TEAM and (serious or _has(r"\b(?:injuries|injured|cuts|wounds|first aid|medical assistance)\b", text)):
        # Avoid inferring treatment from simple negated injury statements.
        if not _has(r"\b(?:no|nobody|no one)\b.*\b(?:injuries|injured|wounds)\b", text):
            return "Injury, illness or first-aid language supports on-scene clinical assistance."
    if resource == R.RESCUE_TEAM and people and trapped:
        return "People are trapped or stranded and require assistance to reach safety."
    water_access = _has(r"flood|deep water|rising water", text) or (
        incident == IncidentType.FLOOD and _has(r"\b(?:water|submerged)\b", text)
    )
    if resource == R.RESCUE_BOAT and people and trapped and water_access and _has(r"access|road|on foot|surrounded|cut off|staircase", text):
        return "Stranded people and flood-blocked access support water-based evacuation."
    if resource == R.FIRE_SERVICE and _has(r"flames|burning|active (?:building )?fire|fire.*spreading", text):
        return "Active flames or burning indicate a fire-suppression need."
    if resource == R.FOOD and _has(r"\b(?:no|without|empty|not arrived|need|hungry)\b.*?\b(?:food|meals|eat)\b|\b(?:food|meal)\b.*?\b(?:empty|not arrived|need)\b", text):
        return "An unmet food supply or hunger statement supports meal provision."
    if resource == R.DRINKING_WATER and _has(r"water", text) and _has(r"empty|contaminated|no safe|no drinking|need|without", text):
        return "An unsafe or missing water supply supports drinking-water provision."
    if resource == R.SHELTER and _has(r"nowhere.*(?:stay|sleep)|(?:need|somewhere|dry|covered|safe).*(?:place|sleep|stay)", text):
        return "An unmet need for a safe place to stay supports temporary accommodation."
    if resource == R.POWER_RESTORATION and _has(r"power|electricity|electrical", text) and _has(r"failed|failure|outage|blackout|lost|without|backup.*(?:exhausted|running out)", text):
        return "An electricity failure or exhausted backup supply supports utility restoration."
    if resource == R.ROAD_CLEARANCE and _has(r"road|route|access", text) and _has(r"debris|tree|rocks|soil|material|obstruction", text) and _has(r"block|across|obstruct|cannot pass|prevent|remov", text):
        return "Physical material obstructs vehicle access and supports road clearance."
    return None


class ResourceIntelligence:
    """Reuse the caller's normalized Embedder; cache only semantic prototypes."""

    def __init__(self, embedder: Embedder, implicit_threshold: float = RESOURCE_IMPLICIT_THRESHOLD):
        if not -1 <= implicit_threshold <= 1:
            raise ValueError("Implicit threshold must be a cosine value between -1 and 1")
        self.embedder = embedder
        self.implicit_threshold = implicit_threshold
        self._labels = [key for key, values in RESOURCE_PROTOTYPES.items() for _ in values]
        self._vectors = embedder.embed_batch([p for values in RESOURCE_PROTOTYPES.values() for p in values])

    def analyze_resource_needs(self, text: str, incident_context: EmergencyReport | IncidentType | None = None) -> list[ResourcePrediction]:
        clauses = _clauses(text)
        if not clauses:
            return []
        incident = incident_context.incident_type if isinstance(incident_context, EmergencyReport) else incident_context
        # Two adjacent sentences preserve need context while limiting topic dilution.
        evidence_spans = clauses + [a + " " + b for a, b in zip(clauses, clauses[1:]) if a + " " + b in text]
        vectors = self.embedder.embed_batch(evidence_spans)
        similarities = self.embedder.cosine_similarity_matrix(vectors, self._vectors)
        results = []
        for resource, phrase in EXPLICIT_PHRASES.items():
            denied = any(_unneeded_or_satisfied(resource, c) for c in clauses)
            explicit = next((c for c in clauses if _has(r"\b(?:" + phrase + r")\b", c)
                             and _has(_REQUEST, c) and not _unneeded_or_satisfied(resource, c)), None)
            if explicit:
                results.append(ResourcePrediction(resource, RESOURCE_EXPLICIT_CONFIDENCE, "EXPLICIT", explicit,
                               "A named resource or operational synonym occurs in an affirmative request."))
                continue
            if denied:
                continue
            candidates = []
            for i, span in enumerate(evidence_spans):
                reason = _context_reason(resource, span, incident)
                if reason and not _unneeded_or_satisfied(resource, span):
                    score = float(max(s for label, s in zip(self._labels, similarities[i]) if label == resource))
                    if score >= self.implicit_threshold:
                        candidates.append((score, span, reason))
            if candidates:
                score, evidence, reason = max(candidates, key=lambda item: item[0])
                results.append(ResourcePrediction(resource, score, "IMPLICIT", evidence,
                               reason + " The evidence also exceeds the semantic prototype threshold."))
        return results
