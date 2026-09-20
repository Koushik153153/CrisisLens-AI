# 🚨 CrisisLens-AI

## NLP-Based Emergency Intelligence, Resource-Need Extraction & Explainable Incident Prioritization

**CrisisLens-AI** is an NLP-based emergency intelligence and decision-support prototype designed to convert unstructured emergency reports into structured, actionable information.

Emergency reports often arrive as free-form text containing mixed information about the incident, location, affected people, vulnerable groups, injuries, required resources, and urgency. CrisisLens-AI automatically analyzes these reports and transforms them into structured emergency intelligence.

The system identifies **incident type, location, affected population, vulnerable groups and resource requirements**, independently assesses **urgency, severity and actionability**, and combines these factors using an **Explainable Action Priority Score (EAPS)**.

An evidence-grounded **Retrieval-Augmented Generation (RAG)** module additionally allows users to query emergency reports using natural language.

---

## 🎯 Problem Statement

Emergency reports are generally received as unstructured textual information.

Although NLP systems can classify incidents and RAG systems can retrieve relevant information, responders may still need to manually determine:

- What type of emergency occurred?
- Where did it occur?
- How many people are affected?
- Are vulnerable people involved?
- What resources are required?
- How urgent and severe is the situation?
- Is immediate intervention possible?
- Which incident should receive attention first?

**CrisisLens-AI addresses this problem by integrating emergency-text understanding, resource intelligence, emergency assessment, explainable prioritization and evidence-grounded retrieval into a unified prototype.**

---

## 💡 Core Idea

CrisisLens-AI converts an incoming report such as:

> "A building collapsed in Saidapet. Five workers are affected, three are trapped and two are seriously injured. Rescue teams and ambulances are required immediately."

into structured emergency intelligence:

```text
Incident Type   : Building Collapse
Location        : Saidapet
Affected People : 5
Vulnerable Group: Injured
Resources       : Rescue Team, Ambulance, Medical Team
Urgency         : Critical
Severity        : Critical
Actionability   : High
EAPS            : 96
Priority        : Critical
```

This allows multiple emergency reports to be represented using the same structure and compared systematically.

---

# ✨ Key Features

### 🧠 Emergency Incident Understanding

Extracts important information from unstructured emergency text:

- Incident type
- Location
- Affected-person count
- Vulnerable groups

Incident classification uses **MiniLM semantic embeddings and cosine similarity against predefined incident prototypes**.

---

### 🚑 Resource Intelligence

Identifies resources required by each emergency, including:

- Ambulance
- Medical Team
- Rescue Team
- Rescue Boat
- Fire Service
- Food
- Drinking Water
- Shelter
- Power Restoration
- Road Clearance

Resource detection combines:

- Explicit resource phrases
- Request/action cues
- Negation handling
- Already-supplied checks
- Semantic similarity

The system avoids inventing resource quantities that are not explicitly stated in the report.

---

### ⚠️ Emergency Assessment

Each report is independently assessed across three dimensions:

**Urgency**  
How quickly intervention is required.

**Severity**  
How serious the consequences of the incident are.

**Actionability**  
How clearly the report provides enough information for a concrete intervention.

These dimensions are assessed separately rather than treating every serious emergency as automatically actionable.

---

# 📊 Explainable Action Priority Score — EAPS

CrisisLens-AI uses an **Explainable Action Priority Score (EAPS)** to support transparent incident prioritization.

The score is calculated as:

```text
EAPS = 40U + 30S + 20A + Context
```

where:

```text
U = Urgency
S = Severity
A = Actionability
Context = Vulnerability + affected population + resource requirement
```

### Priority Levels

| EAPS Score | Priority |
|---|---|
| < 40 | Low |
| 40 – < 60 | Medium |
| 60 – < 80 | High |
| 80 – 100 | Critical |

The final score is explainable because users can inspect how urgency, severity, actionability and contextual factors contributed to the result.

> **Important:** EAPS is a prototype decision-support scoring framework and is not an official emergency triage or dispatch standard.

---

# 🔎 Emergency RAG Module

CrisisLens-AI includes an **evidence-grounded Retrieval-Augmented Generation (RAG)** module for querying emergency reports using natural language.

Emergency reports are divided into chunks and converted into semantic embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are stored in **ChromaDB**.

When a user submits a query, CrisisLens-AI retrieves the most semantically relevant report chunks and applies a relevance threshold before supplying the accepted evidence to **Google Gemini**.

### RAG Capabilities

- Semantic emergency-report search
- Top-K retrieval
- Cosine-similarity relevance filtering
- Multi-report evidence retrieval
- Evidence-grounded LLM responses
- Adjustable retrieval depth
- Adjustable relevance threshold

For example:

```text
Which emergencies require immediate medical or rescue assistance,
and what resources are needed for each?
```

The system retrieves relevant evidence from the available emergency reports and generates a response grounded in that evidence.

### Role of RAG

RAG is a **supporting evidence-retrieval component**.

The structured emergency intelligence and EAPS calculation are performed independently of RAG.

---

# 🏗️ System Architecture

```text
                  Emergency Reports
                  PDF / DOCX / TXT
                         │
                         ▼
                Document Processing
                         │
                         ▼
                Emergency Analyzer
          ┌──────────────┼──────────────┐
          │              │              │
       Incident       Location      Affected /
        Type                         Vulnerable
          │
          ▼
             Resource Intelligence
                         │
                         ▼
               Emergency Assessment
              ┌──────────┼──────────┐
              │          │          │
           Urgency    Severity   Actionability
              └──────────┼──────────┘
                         │
                         ▼
                 EAPS Priority Engine
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Structured Report        MiniLM Embeddings
           Storage                    │
                                      ▼
                                  ChromaDB
                                      │
                                      ▼
                                 Emergency RAG
             │
             └───────────┬───────────┘
                         ▼
               Streamlit Interface
```

---

# 🔄 Processing Pipeline

The complete processing flow is:

```text
Emergency Report
       ↓
Text Extraction
       ↓
Incident Understanding
       ↓
Resource-Need Detection
       ↓
Urgency / Severity / Actionability Assessment
       ↓
EAPS Calculation
       ↓
Structured Report Persistence
       ↓
Semantic Embedding + ChromaDB
       ↓
Dashboard + Emergency RAG
```

The structured NLP intelligence pipeline and RAG pipeline are intentionally separated.

This allows EAPS and emergency assessment to operate independently from LLM-generated answers.

---

# 🖥️ User Interface

The system is implemented using **Streamlit** and provides four main views.

### 1. Command Center

Provides an overview of all processed emergency reports.

Displays:

- Total reports
- Critical incidents
- High-priority incidents
- Resource requirements
- Priority queue
- Incident distribution
- Resource overview

---

### 2. Report Analysis

Provides detailed intelligence for an individual emergency report.

Displays:

- Incident type
- Location
- Affected population
- Vulnerable groups
- Required resources
- Urgency
- Severity
- Actionability
- EAPS
- Priority level
- Explanation of priority

---

### 3. Emergency RAG

Allows users to ask natural-language questions across emergency reports.

Users can configure:

- Retrieval depth (K)
- Relevance threshold

Retrieved evidence is used to generate grounded responses.

---

### 4. System Evaluation

Displays evaluation results for:

- Incident understanding
- Resource intelligence
- Emergency assessment
- EAPS structural validation
- End-to-end pipeline validation

---

# 🧪 Controlled Development Dataset

The project includes **20 synthetic emergency reports** covering scenarios such as:

- Floods
- Fires
- Medical emergencies
- Road accidents
- Building collapses
- Landslides
- Cyclones/storms
- Power outages
- Road blockages
- Low-risk monitoring situations

The reports intentionally contain different combinations of:

- Locations
- Population counts
- Vulnerable groups
- Resource requests
- Urgency levels
- Severity levels
- Actionability levels

Ground-truth annotations are stored in:

```text
emergency_data/sample_ground_truth.json
```

The dataset is intended for **controlled development and evaluation**, not as a real-world emergency benchmark.

---

# 📈 Evaluation Results

CrisisLens-AI was evaluated using the controlled 20-report synthetic development corpus.

## Incident Understanding

| Metric | Result |
|---|---:|
| Incident Classification Accuracy | 100% |
| Location F1 | 100% |
| Affected Count Exact Match | 100% |
| Vulnerable Group F1 | 100% |

---

## Resource Intelligence

| Metric | Result |
|---|---:|
| Precision | 100% |
| Recall | 92.31% |
| Micro F1 | 96% |
| Exact-Set Accuracy | 90% |

Resource intelligence produced no false-positive resource predictions in the controlled corpus, while two expected resource associations were missed.

---

## Emergency Assessment

| Component | Accuracy | Macro F1 |
|---|---:|---:|
| Urgency | 100% | 1.0000 |
| Severity | 80% | 0.7597 |
| Actionability | 90% | 0.6181 |

Severity assessment remains one of the more challenging components, particularly when distinguishing between medium, high and critical situations.

---

# 🧮 EAPS Validation

No gold-standard EAPS priority labels are available in the synthetic dataset.

Therefore, the project **does not claim EAPS accuracy**.

Instead, the scoring engine is tested for structural properties including:

- Score bounds
- Deterministic reproducibility
- Component consistency
- Urgency monotonicity
- Severity monotonicity
- Actionability monotonicity
- Missing-data handling
- Context-score limits

All implemented structural validation checks passed.

For the 20-report development corpus:

```text
Minimum EAPS : 24.10
Maximum EAPS : 98.00
Mean EAPS    : 68.895
Median EAPS  : 77.50
```

Priority distribution:

```text
Low      : 4
Medium   : 2
High     : 6
Critical : 8
```

---

# ✅ Software Testing

The project includes automated tests covering:

- Architecture
- Emergency analyzer
- Synthetic dataset
- Resource intelligence
- Emergency assessment
- EAPS priority engine
- Streamlit UI
- End-to-end emergency pipeline

Current validation:

```text
85 automated tests passed
```

These tests validate software behavior and pipeline consistency and should not be interpreted as model accuracy.

---

# 🔧 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| NLP Embeddings | all-MiniLM-L6-v2 |
| Embedding Library | Sentence Transformers |
| Vector Database | ChromaDB |
| LLM | Google Gemini |
| Data Processing | Pandas / NumPy |
| Visualization | Plotly |
| File Monitoring | Watchdog |
| Testing | Python unittest / Streamlit testing |

---

# 📁 Project Structure

```text
CrisisLens_AI/
│
├── app.py
├── config.py
├── crisislens_ui.py
├── crisislens_theme.py
│
├── document_processor.py
├── emergency_analyzer.py
├── emergency_schema.py
├── emergency_taxonomy.py
│
├── resource_intelligence.py
├── emergency_assessment.py
├── priority_engine.py
├── emergency_pipeline.py
│
├── embedder.py
├── vector_store.py
├── retriever.py
├── llm_handler.py
│
├── file_watcher.py
├── evaluator.py
├── visualizer.py
│
├── sample_emergency_reports/
│   ├── 01_flood_rooftop.txt
│   ├── ...
│   └── 20_reservoir_monitor.txt
│
├── emergency_reports/
│   └── .gitkeep
│
├── emergency_data/
│   └── sample_ground_truth.json
│
├── test_architecture.py
├── test_emergency_analyzer.py
├── test_resource_intelligence.py
├── test_emergency_assessment.py
├── test_priority_engine.py
├── test_emergency_pipeline.py
├── test_emergency_dataset.py
├── test_crisislens_ui.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Koushik153153/CrisisLens-AI.git
cd CrisisLens-AI
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

If PowerShell execution policy prevents activation, the virtual-environment Python can be used directly.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git through `.gitignore`.

**Never commit API keys or other credentials to the repository.**

---

# ▶️ Running CrisisLens-AI

Run:

```bash
streamlit run app.py
```

On Windows, the virtual-environment executable can also be used directly:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will provide a local URL for accessing the application.

---

# 📥 Adding Emergency Reports

Emergency reports can be added through the application or placed in:

```text
emergency_reports/
```

Supported formats:

```text
PDF
DOCX
TXT
```

The watched-folder mechanism detects supported report files and queues them for processing.

The current prototype processes queued changes during Streamlit application execution/reruns.

---

# 🧠 NLP Methodology

CrisisLens-AI combines semantic NLP techniques with deterministic rules.

### Incident Classification

Emergency text is embedded using:

```text
all-MiniLM-L6-v2
```

The resulting embedding is compared with predefined incident prototypes using cosine similarity.

The highest matching incident category is selected when the similarity exceeds the configured confidence threshold.

---

### Location Extraction

Locations are detected using a corpus-derived gazetteer.

---

### Affected-Person Extraction

The system identifies:

- Numeric counts
- Common number words
- Human/group nouns

Families and households are not automatically converted into person counts because doing so would require unsupported assumptions.

---

### Vulnerable-Group Detection

The system detects groups such as:

- Children
- Elderly people
- Pregnant people
- Injured people
- Disabled people
- Chronically ill people

Phrase matching and local negation handling are used to reduce incorrect detections.

---

### Resource Detection

Resource intelligence combines explicit resource mentions with semantic matching.

The system also checks contextual cues to distinguish between:

```text
"Ambulance required immediately"
```

and:

```text
"No ambulance is required"
```

or resources that have already been supplied.

---

### Emergency Assessment

Urgency, severity and actionability are independently evaluated using semantic prototypes and contextual cues.

This separation is important because:

```text
High Severity ≠ Automatically High Actionability
```

An emergency may be severe while still lacking enough information for an immediate concrete intervention.

---

# 🔍 Semantic Retrieval

Emergency-report chunks are converted into MiniLM embeddings and stored in ChromaDB.

For each query:

```text
User Query
    ↓
Query Embedding
    ↓
Cosine Similarity Search
    ↓
Top-K Candidate Chunks
    ↓
Relevance Threshold
    ↓
Accepted Evidence
    ↓
Gemini
    ↓
Grounded Answer
```

**K** controls how many top candidate chunks are retrieved.

The **relevance threshold** determines which of those candidates are sufficiently semantically similar to be accepted as evidence.

Similarity scores represent semantic closeness and should not be interpreted as probabilities or accuracy scores.

---

# ⚠️ Current Limitations

CrisisLens-AI is a research/development prototype.

Current limitations include:

- Evaluation is based on a controlled synthetic development corpus.
- Real-world emergency language can be substantially more complex and noisy.
- Severity and actionability classification require further validation.
- EAPS weights and priority bands are design choices and have not been clinically or operationally validated.
- EAPS does not currently have gold-standard priority labels.
- Location extraction uses a limited gazetteer.
- Watched-folder changes are processed during Streamlit execution/reruns rather than by an independent production service.
- Historical processed-report versions can accumulate.
- File deletion/rename synchronization is not implemented.
- The system is not connected to real emergency dispatch infrastructure.

---

# 🛡️ Safety and Intended Use

CrisisLens-AI is intended as an **academic NLP and emergency decision-support prototype**.

It is **not** intended to replace:

- Emergency dispatchers
- Medical professionals
- Disaster-response authorities
- Government emergency protocols
- Official triage systems

EAPS and other generated assessments should be interpreted as experimental NLP-derived decision-support outputs rather than authoritative emergency decisions.

---

# 🔬 Research Direction

The project explores the following question:

> **Can actionable-information extraction, urgency analysis, resource-need detection and semantic incident aggregation be combined to prioritize emergency textual reports more effectively than similarity-based retrieval alone?**

CrisisLens-AI investigates this through the integration of:

```text
Emergency NLP
      +
Resource Intelligence
      +
Urgency / Severity / Actionability
      +
Explainable Priority Scoring
      +
Evidence-Grounded RAG
```

rather than treating emergency classification and information retrieval as isolated tasks.

---

# 🔮 Future Work

Possible future improvements include:

- Evaluation using larger real-world emergency datasets
- Improved location/entity extraction
- Fine-tuned emergency-domain language models
- Learned or expert-validated priority weights
- Improved severity and actionability classification
- Multilingual emergency-report processing
- Temporal event tracking
- Duplicate/near-duplicate incident detection
- Real-time external emergency-data integration
- Human-in-the-loop validation
- Deployment with emergency-management workflows

---

# 👨‍💻 Project

**CrisisLens-AI**

NLP-Based Emergency Intelligence System for Actionability Detection, Resource-Need Extraction and Explainable Incident Prioritization.

Developed as an academic Natural Language Processing project.

---

## Repository

GitHub:

https://github.com/Koushik153153/CrisisLens-AI
