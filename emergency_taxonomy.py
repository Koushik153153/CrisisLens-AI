"""Canonical domain labels and label formatting only; no NLP inference."""

import re
from enum import Enum
from typing import TypeVar


class CanonicalLabel(str, Enum):
    """String-valued enums serialize directly with the standard JSON encoder."""

    def __str__(self) -> str:
        return self.value


class IncidentType(CanonicalLabel):
    FLOOD = "FLOOD"
    FIRE = "FIRE"
    MEDICAL_EMERGENCY = "MEDICAL_EMERGENCY"
    ROAD_ACCIDENT = "ROAD_ACCIDENT"
    BUILDING_COLLAPSE = "BUILDING_COLLAPSE"
    LANDSLIDE = "LANDSLIDE"
    CYCLONE_STORM = "CYCLONE_STORM"
    POWER_OUTAGE = "POWER_OUTAGE"
    ROAD_BLOCKAGE = "ROAD_BLOCKAGE"
    OTHER = "OTHER"


class ResourceNeed(CanonicalLabel):
    AMBULANCE = "AMBULANCE"
    MEDICAL_TEAM = "MEDICAL_TEAM"
    RESCUE_TEAM = "RESCUE_TEAM"
    RESCUE_BOAT = "RESCUE_BOAT"
    FIRE_SERVICE = "FIRE_SERVICE"
    FOOD = "FOOD"
    DRINKING_WATER = "DRINKING_WATER"
    SHELTER = "SHELTER"
    POWER_RESTORATION = "POWER_RESTORATION"
    ROAD_CLEARANCE = "ROAD_CLEARANCE"
    OTHER = "OTHER"


class Urgency(CanonicalLabel):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Severity(CanonicalLabel):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Actionability(CanonicalLabel):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PriorityLevel(CanonicalLabel):
    """Reserved schema vocabulary; no priority annotations or scoring yet."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VulnerableGroup(CanonicalLabel):
    CHILDREN = "CHILDREN"
    ELDERLY = "ELDERLY"
    PREGNANT = "PREGNANT"
    INJURED = "INJURED"
    DISABLED = "DISABLED"
    CHRONICALLY_ILL = "CHRONICALLY_ILL"


class ActionType(CanonicalLabel):
    MONITOR = "MONITOR"
    WARN = "WARN"
    EVACUATE = "EVACUATE"
    RESCUE = "RESCUE"
    PROVIDE_MEDICAL_AID = "PROVIDE_MEDICAL_AID"
    DISPATCH_AMBULANCE = "DISPATCH_AMBULANCE"
    EXTINGUISH_FIRE = "EXTINGUISH_FIRE"
    PROVIDE_FOOD = "PROVIDE_FOOD"
    PROVIDE_WATER = "PROVIDE_WATER"
    PROVIDE_SHELTER = "PROVIDE_SHELTER"
    RESTORE_POWER = "RESTORE_POWER"
    CLEAR_ROAD = "CLEAR_ROAD"
    OTHER = "OTHER"


LabelType = TypeVar("LabelType", bound=CanonicalLabel)


def normalize_label(value: str | CanonicalLabel | None) -> str | None:
    """Format a label: trim, uppercase, and replace spaces/hyphens with '_'.

    None or blank means unknown. This does not validate membership or interpret
    synonyms/prose: 'medical help' becomes 'MEDICAL_HELP', not 'MEDICAL_TEAM'.
    Non-string input raises TypeError instead of silently coercing invalid data.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("A label must be a string, canonical enum, or None")
    normalized = re.sub(r"[\s-]+", "_", str(value).strip()).upper()
    return normalized or None


def canonical_label(
    value: str | CanonicalLabel | None, taxonomy: type[LabelType]
) -> str | None:
    """Normalize and validate a label in the requested enum.

    Unknown/blank input stays None. Invalid labels raise ValueError; they are
    never silently mapped to OTHER or LOW. Returns a plain JSON-safe string.
    """
    if isinstance(value, CanonicalLabel) and not isinstance(value, taxonomy):
        raise ValueError("Label belongs to a different taxonomy")
    normalized = normalize_label(value)
    return None if normalized is None else taxonomy(normalized).value
