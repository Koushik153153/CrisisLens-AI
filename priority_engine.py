"""Deterministic prototype EAPS; uses structured labels, never text or confidence."""

from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP

from config import EAPS_CONFIG
from emergency_schema import EmergencyReport, PriorityComponent, PriorityResult
from emergency_taxonomy import Urgency, Severity, Actionability, PriorityLevel, ResourceNeed, VulnerableGroup

PRIMARY = {"urgency": Urgency, "severity": Severity, "actionability": Actionability}


def _decimal(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric design parameter")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("EAPS parameters must be finite")
    return result


def _cents(value):
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class EmergencyPriorityEngine:
    def __init__(self, settings=None):
        self.settings = deepcopy(EAPS_CONFIG if settings is None else settings)
        cfg = self.settings
        if set(cfg["weights"]) != {*PRIMARY, "context"}:
            raise ValueError("Expected three primary weights and context")
        weights = [_decimal(v) for v in cfg["weights"].values()]
        if any(v < 0 for v in weights) or sum(weights) != 1 or _decimal(cfg["weights"]["context"]) >= Decimal("0.5"):
            raise ValueError("Weights must be nonnegative, sum to 1, and primarily weight the primary dimensions")
        for name, enum in PRIMARY.items():
            mapping = cfg["ordinal"][name]
            if set(mapping) != {item.value for item in enum}:
                raise ValueError("Ordinal labels must match taxonomy")
            values = [_decimal(mapping[item.value]) for item in enum]
            if values != sorted(values) or any(v < 0 or v > 1 for v in values):
                raise ValueError("Ordinal mappings must be monotone and in [0,1]")
        shares = cfg["context_shares"]
        if set(shares) != {"vulnerability", "population", "operational_need"}:
            raise ValueError("Unexpected context shares")
        for values in (shares.values(), cfg["vulnerability_factors"].values(), cfg["population_tiers"].values()):
            if any(not 0 <= _decimal(v) <= 1 for v in values):
                raise ValueError("Context factors must be in [0,1]")
        if sum(_decimal(v) for v in shares.values()) != 1:
            raise ValueError("Context shares must sum to 1")
        if set(cfg["vulnerability_factors"]) != {v.value for v in VulnerableGroup}:
            raise ValueError("Vulnerability factors must cover the taxonomy")
        tiers = sorted(cfg["population_tiers"].items())
        if not tiers or any(type(k) is not int or k < 1 for k, _ in tiers) or [v for _, v in tiers] != sorted(v for _, v in tiers):
            raise ValueError("Person-count tiers must be positive and monotone")
        bands = cfg["bands"]
        if set(bands) != {v.value for v in PriorityLevel}:
            raise ValueError("Priority bands must match taxonomy")
        boundaries = [_decimal(bands[v.value]) for v in PriorityLevel]
        if boundaries[0] != 0 or boundaries != sorted(set(boundaries)) or boundaries[-1] > 100:
            raise ValueError("Bands must strictly increase from zero within [0,100]")

    def level_for_score(self, score: float) -> PriorityLevel:
        value = _decimal(score)
        if not 0 <= value <= 100:
            raise ValueError("Score must be in [0,100]")
        return max((v for v in PriorityLevel if value >= _decimal(self.settings["bands"][v.value])),
                   key=lambda v: self.settings["bands"][v.value])

    def score(self, report: EmergencyReport) -> PriorityResult:
        """Missing/invalid primary labels make a full EAPS unavailable, never LOW."""
        missing = []
        for name, enum in PRIMARY.items():
            try:
                enum(getattr(report, name))
            except (ValueError, TypeError):
                missing.append(name)
        if missing:
            return PriorityResult(None, None, reasons=[
                "EAPS unavailable: missing or invalid primary dimensions: " + ", ".join(missing) + ".",
                "No partial or low-risk score has been assigned.",
            ])
        cfg = self.settings
        components, reasons = {}, []
        for name, enum in PRIMARY.items():
            label = enum(getattr(report, name)).value
            raw = _decimal(cfg["ordinal"][name][label])
            weight = _decimal(cfg["weights"][name])
            points = _cents(100 * weight * raw)
            components[name] = PriorityComponent(label, float(raw), float(weight), float(points))
            reasons.append(f"{label.title()} {name} contributes {points:.2f} points.")

        groups = {VulnerableGroup(g).value for g in report.vulnerable_groups}
        vulnerability = max((cfg["vulnerability_factors"][g] for g in groups), default=0)
        # Prefer the explicit count/unit fields. Legacy affected_people is already
        # an individual count, but must not override a non-person unit.
        count, unit = report.affected_count, report.affected_unit
        if count is None and unit is None and report.affected_people is not None:
            count, unit = report.affected_people, "PERSONS"
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Affected count must be a nonnegative integer or None")
        population = max((factor for minimum, factor in cfg["population_tiers"].items()
                          if unit == "PERSONS" and count is not None and count >= minimum), default=0)
        resources = {ResourceNeed(r) for r in report.resource_needs}
        operational = int(bool(resources - {ResourceNeed.OTHER}))
        factors = {"vulnerability": vulnerability, "population": population, "operational_need": operational}
        descriptions = {
            "vulnerability": "Maximum factor across identified groups " + (", ".join(sorted(groups)) or "(none)") + "; categories are not added.",
            "population": f"Reported count={count}, unit={unit}; only individual-person tiers apply; no family conversion.",
            "operational_need": "Concrete resource presence counted once, regardless of the number of resources.",
        }
        context_weight = _decimal(cfg["weights"]["context"])
        context_total = Decimal(0)
        details = []
        for name, raw in factors.items():
            share = _decimal(cfg["context_shares"][name])
            points = _cents(100 * context_weight * share * _decimal(raw))
            context_total += points
            details.append(f"{name}: factor={raw}, context share={share}, points={points:.2f}. {descriptions[name]}")
        context_total = min(context_total, _cents(100 * context_weight))
        raw_context = context_total / (100 * context_weight) if context_weight else Decimal(0)
        components["context"] = PriorityComponent(None, float(raw_context), float(context_weight), float(context_total), details)
        reasons.extend(details)
        # Sum displayed, cent-rounded components so a reviewer can reproduce it.
        total = _cents(sum(_decimal(c.points) for c in components.values()))
        if not 0 <= total <= 100:
            raise ValueError("Configured rounded contributions exceed score bounds")
        return PriorityResult(float(total), self.level_for_score(float(total)), components, reasons, True)
