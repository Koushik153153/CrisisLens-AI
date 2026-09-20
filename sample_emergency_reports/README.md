# CrisisLens-AI synthetic controlled corpus (version 1)

These 20 fictional field reports are for development, demonstration and
evaluation. They are **not a real emergency feed**, verified incident records,
or copies of current disaster reports. Chennai and Tamil Nadu place names
provide demonstration context only. All situations and person counts are
invented for this controlled corpus.

The wording intentionally varies between calls, volunteer messages, field
notes and follow-up logs. Some requests name resources explicitly; others imply
them through unmet needs. Some reports include vulnerable populations, multiple
resources, or deliberately low-actionability observations. The files do not
present the target severity/urgency/actionability labels as report text.

## Files and distribution

| Primary incident | Reports | Files |
| --- | ---: | --- |
| FLOOD | 3 | 01–03 |
| FIRE | 2 | 04–05 |
| MEDICAL_EMERGENCY | 2 | 06–07 |
| ROAD_ACCIDENT | 2 | 08–09 |
| BUILDING_COLLAPSE | 2 | 10–11 |
| LANDSLIDE | 1 | 12 |
| CYCLONE_STORM | 2 | 13–14 |
| POWER_OUTAGE | 2 | 15–16 |
| ROAD_BLOCKAGE | 2 | 17–18 |
| OTHER | 2 | 19–20 |

Examples: report 02 explicitly requests a rescue boat; report 03 explicitly
requests drinking water and meals. Report 08 implies ambulance transport through
serious injuries and an unmet need to reach hospital. Report 11 implies SHELTER
through a need for somewhere safe to sleep. Reports 05, 16, 19 and 20 require
monitoring only. Report 01 combines rescue boat, rescue team and shelter needs.

## Gold-standard annotations

`../emergency_data/sample_ground_truth.json` is a JSON array containing one
manually authored expected output per report. Each object contains:

- `report_id`: stable synthetic ID (`CL-SYN-001` through `CL-SYN-020`).
- `source_file`: basename relative to this directory, unique per report.
- `incident_type`: one primary canonical incident label. Context such as a relief
  camp remains FLOOD when flood displacement is the reason for the report.
- `locations`: place strings explicitly stated in the text, without geocoding.
- `affected_people`: stated incident/displacement/patient total, not a sum of
  overlapping subgroups. `null` means the total is not stated. `0` is used only
  when absence of directly affected persons is explicit, not merely because
  nobody is injured. Service users are not counted from household estimates.
- `vulnerable_groups`, `resource_needs`, `actions`: canonical label lists.
  `[]` means no such group/need is evidenced by this report; it is not a claim
  about unseen people or all real-world response requirements.
- `action_required`: boolean for additional intervention beyond routine
  monitoring. `false` is compatible with `actions: ["MONITOR"]`.
- `urgency`, `severity`, `actionability`: manually assigned canonical labels
  using the rubric below. Unknown values may be `null`, never default LOW.

No priority score, priority level, or EAPS target is supplied. Counts and
vulnerable groups are not invented from likely demographics. Resources already
provided are excluded. Implicit resources reflect the deliberately described
unmet need; they are annotation judgments, not validated dispatch decisions.
The dataclass preserves its Step-1 boolean and adds a separate `actions` list.
String enum values work with `asdict()` and `json.dumps()`; dataclass type hints
do not themselves validate input. Use `canonical_label()` at input boundaries.

## Annotation rubric

These are project-specific development conventions, not an official emergency
triage protocol. Severity describes stated harm or a concrete described threat;
urgency describes how soon action is needed; actionability describes how clearly
the report supports a response. Low severity does not mean missing information.

| Level | Urgency | Severity | Actionability |
| --- | --- | --- | --- |
| LOW | Routine observation or resolved event | No current harm or minor injury | No intervention beyond observation is supported |
| MEDIUM | Work round / non-immediate assistance | Local access disruption or unmet basic supplies | Response direction exists, but operational scope still needs confirmation |
| HIGH | Prompt / same-day action before deterioration | Serious injury, displacement, or significant concrete threat | Clear place, problem and feasible response, even if headcount is unknown |
| CRITICAL | Immediate response explicitly supported | Entrapment with immediate danger, unresponsiveness, or imminent life-support loss | Not an actionability label |

For report 13, HIGH severity is the forecast threat, not observed damage; MEDIUM
actionability reflects pending household checks. Report 08 is CRITICAL urgency
but HIGH severity: serious injuries and immediate transport need are stated,
without annotating an additional diagnosis. Report 15 requests power support
only because the clinical team is already attending. A resource OTHER example
is not forced into the corpus; vocabulary coverage is broader than sample use.

These labels are an initial author-defined gold standard. There has been no
independent expert adjudication or inter-annotator agreement study. Twenty
controlled examples are too small for generalization claims. If used for rules
or training later, they must not also be presented as an independent test set.

## Live ingestion separation

This directory is **not watched**, scanned at startup, or automatically indexed.
The only live ingestion directory remains `emergency_reports/`. Samples are
kept here for controlled evaluation. A user may later copy an individual sample
into that live directory for an explicitly synthetic demonstration.

Run validation from the project directory:

```powershell
.\.venv\Scripts\python.exe -m unittest test_architecture test_emergency_dataset -v
```

These tests validate structure, taxonomy, serialization and isolation, not NLP
accuracy. No classifier, extractor, predictor, scoring or RAG change is included.
