# NADI — Self-Healing Machines: Predictive Maintenance for a Connected Fleet
## Complete Project Brief for Round 1 (Idea Submission) PPT Generation

> **INSTRUCTIONS FOR CLAUDE / AI**: Use this document to create a professional, visually stunning PowerPoint presentation for the **Round 1 Idea Submission** of the **Teliport Season 4 Hackathon** by Tata Elxsi. The theme is "Self-Healing Machines: Predictive Maintenance for a Connected Fleet." Our system is called **NADI** (meaning "pulse" — sensing the heartbeat of machines). The PPT should be a **Design Deck** covering architecture, data engineering, ML approach, agent logic, and dataset strategy. Keep it concise (10-15 slides), visually clean, and technically impressive. Use dark/modern professional theme with blue/orange accents matching the Teliport branding.

---

## EVALUATION CRITERIA (What Judges Are Looking For)

| Criterion | Weight | What to Demonstrate |
|---|---|---|
| **Problem Understanding** | 15% | Clear grasp of predictive maintenance, RUL prediction, why fixed-schedule maintenance fails |
| **Architecture Clarity** | 20% | Coherent end-to-end pipeline diagram: Data → Features → Model → Agent → Work Order |
| **Data Engineering Approach** | 15% | Sound, well-reasoned data cleaning, feature extraction, leakage prevention |
| **ML Approach & Reasoning** | 20% | Suitable model selection with justification, awareness of practical challenges (class imbalance, censoring) |
| **MLOps & Agent Logic** | 15% | Practical operational plan, clear triage agent decision flow, monitoring |
| **Dataset Coverage** | 10% | Strategy for handling all 4 C-MAPSS subsets (FD001–FD004) of varying complexity |
| **Communication Quality** | 5% | Well-structured, unambiguous, internally consistent presentation |

---

## SLIDE-BY-SLIDE CONTENT

---

### SLIDE 1: Title Slide

**Title**: NADI — Self-Healing Machines
**Subtitle**: Predictive Maintenance for a Connected Fleet
**Tagline**: "Sensing the pulse of every engine, before it flatlines."
**Event**: Teliport Season 4 | Tata Elxsi
**Team Name**: [INSERT YOUR TEAM NAME]

---

### SLIDE 2: The Problem

**Headline**: Why Fixed-Schedule Maintenance Is Broken

**Key Points**:
- A fleet of connected machines (turbofan engines, industrial compressors, buses) continuously streams sensor data — vibration, temperature, pressure, motor current, RPM.
- **Today's approach**: Maintenance is triggered either AFTER a breakdown (reactive) or on fixed calendar intervals (preventive). Both are wasteful.
- **Reactive maintenance** = unplanned downtime, cascading failures, safety risks.
- **Preventive maintenance** = replacing parts that still have useful life, wasting 30-40% of component value.
- **The cost**: Unplanned downtime costs industrial companies an estimated $50B/year globally.

**What We Need**: A closed-loop system that:
1. Ingests live sensor telemetry
2. Detects early signs of degradation
3. Predicts how many cycles a machine has left (Remaining Useful Life — RUL)
4. Automatically generates maintenance work orders
5. Keeps a human in the loop only for exceptions

---

### SLIDE 3: Our Solution — NADI Architecture Overview

**Headline**: End-to-End Closed-Loop Predictive Maintenance Pipeline

**Architecture Diagram** (create a clean flow diagram):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NADI ARCHITECTURE                                    │
│                                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  SENSOR   │───▶│  DATA ENGINE  │───▶│  ML MODELS   │───▶│   TRIAGE     │ │
│  │ TELEMETRY │    │  (Cleaning &  │    │  (RUL + Anom) │    │   AGENT      │ │
│  │  STREAM   │    │   Features)   │    │              │    │  (Rule-Based)│ │
│  └──────────┘    └───────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                                   │         │
│                         ┌─────────────────────────────────────────┘         │
│                         ▼                                                   │
│                  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│                  │  WORK ORDER  │    │   MLOps &    │    │  STREAMLIT   │  │
│                  │  GENERATION  │    │   DRIFT      │    │  MONITORING  │  │
│                  │ (Idempotent) │    │  DETECTION   │    │  DASHBOARD   │  │
│                  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                             │
│  Layer 1: DATA ENGINEERING  │  Layer 2: ML  │  Layer 3: MLOps  │  Layer 4: AGENT  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles**:
- **Zero Data Leakage**: Every transform is fit on training data only, with strict engine-level grouping.
- **Deterministic Agent**: 100% rule-based triage (no LLM) — fast, auditable, reproducible.
- **Idempotent Work Orders**: Prevents duplicate alerts and alarm fatigue.
- **Observable & Explainable**: Every prediction comes with drift scores, anomaly flags, and confidence metrics.

---

### SLIDE 4: The Dataset — NASA C-MAPSS

**Headline**: NASA C-MAPSS Turbofan Engine Degradation Benchmark

**Dataset Structure**:
- **Source**: NASA Prognostics Data Repository — well-documented benchmark of medium complexity.
- **Content**: Run-to-failure sensor readings from a simulated fleet of turbofan engines.
- **Sensors**: 21 sensor channels (temperature, pressure, fan speed, physical/corrected) + 3 operational settings.
- **Ground Truth**: True Remaining Useful Life (RUL) labels for each engine at each cycle.

**Four Subsets of Increasing Complexity**:

| Subset | Engines (Train) | Engines (Test) | Operating Conditions | Fault Modes | Complexity |
|--------|-----------------|----------------|---------------------|-------------|------------|
| **FD001** | 100 | 100 | 1 (sea level) | 1 (HPC degradation) | ⭐ Simple |
| **FD002** | 260 | 259 | 6 (altitude/Mach) | 1 (HPC degradation) | ⭐⭐ Medium |
| **FD003** | 100 | 100 | 1 (sea level) | 2 (HPC + Fan) | ⭐⭐ Medium |
| **FD004** | 249 | 248 | 6 (altitude/Mach) | 2 (HPC + Fan) | ⭐⭐⭐ Hard |

**Our Coverage**: We tackle ALL 4 subsets with a unified global model pipeline, demonstrating scalability from simple single-condition to complex multi-condition, multi-fault scenarios. FD001 serves as our primary development/validation subset; the global model generalizes across all 707 test engines.

---

### SLIDE 5: Data Engineering — Cleaning & Integrity

**Headline**: Bulletproof Data Pipeline with Zero Leakage

**Data Ingestion**:
- 26-column space-delimited format: `unit_id`, `cycle`, 3 operational settings, 21 sensors.
- **Automated integrity audit**: Checks for NaN/inf values, monotonic cycle ordering, no missing cycles, valid ID ranges — fails fast on any corruption.

**RUL Target Engineering**:
- Ground truth: `RUL = max_cycle_for_engine - current_cycle`
- **Piecewise Linear Degradation with Target Capping**: Cap RUL at a maximum (e.g., 100 or 125 cycles) because early-life sensors show no degradation signal. This converts the problem from raw linear regression to a more learnable piecewise function.
  - Without capping: Models waste capacity trying to distinguish "300 cycles left" vs "280 cycles left" when sensors look identical.
  - With cap=100: The model focuses on the critical degradation window where sensor signals are actually informative.

**Leakage Prevention (Critical)**:
- **Engine-level grouped splits**: Engines are NEVER split across train/validation. All cycles of one engine stay together.
- **Strictly causal feature extraction**: Rolling windows use only past data — no future cycles ever leak into features.
- **Train-only fitting**: All scalers, K-Means clusterers, PCA transforms, and low-variance filters are fit exclusively on training engines.

---

### SLIDE 6: Feature Engineering — Causal Rolling Windows

**Headline**: Extracting Degradation Signals from Raw Telemetry

**Sensor Analysis**:
- **Flat/Non-informative sensors removed**: Sensors 1, 5, 6, 10, 16, 18, 19 have std < 0.01 on FD001 — they carry no degradation signal.
- **Informative sensors retained**: The remaining 14 sensors capture temperature rises, pressure drops, and efficiency losses that precede failure.

**Feature Extraction** (strictly causal — past-only windows):

| Feature Type | Description | Windows |
|---|---|---|
| **Rolling Mean** | Smoothed sensor average over past W cycles | W = 10, 30 |
| **Rolling Std** | Sensor volatility/noise over past W cycles | W = 10, 30 |
| **Lag Delta (Δ₁)** | Cycle-to-cycle change: `sensor[t] - sensor[t-1]` | 1 cycle |
| **Lag Delta (Δw)** | Long-range drift: `sensor[t] - sensor[t-W]` | W = 10, 30 |
| **Rolling Slope** | Linear trend rate over past W cycles | W = 10, 30 |
| **Initial Difference** | `sensor[t] - sensor[cycle=1]` — deviation from healthy baseline | N/A |

**Condition-Aware Normalization** (for multi-condition subsets FD002/FD004):
- K-Means clustering (k=6) on standardized operating settings (`setting_1`, `setting_2`, `setting_3`) — fitted on training data only.
- Each sensor is then z-score normalized within its operating regime cluster.
- This removes altitude/Mach effects so the model sees pure degradation signals regardless of flight conditions.

**Total engineered features**: ~168 features per time step from 14 informative sensors × 12 feature types.

---

### SLIDE 7: ML Approach — Model Selection & Reasoning

**Headline**: Principled Model Selection with Rigorous Validation

**Model Zoo — Why Multiple Approaches?**

We explored a comprehensive model zoo to find the best fit:

| Model Category | Models | Purpose |
|---|---|---|
| **Baselines** | Mean, Ridge Regression | Sanity checks — any real model must beat these |
| **Tabular Trees** | Random Forest, Extra Trees, **HistGradientBoosting (HGB)** | Primary workhorses — handle tabular sensor features excellently |
| **Deep Sequence** | 1D-CNN, LSTM, CNN-LSTM | Capture temporal patterns in raw sensor sequences |
| **Compact/Edge** | TinyML Dense (20→10→1), Compact TCN with SE-Attention | For on-device/edge deployment scenarios |
| **Anomaly Detection** | Autoencoder (D→64→16→64→D) | Unsupervised health monitoring — not for RUL, but for detecting abnormal degradation patterns |

**Why HistGradientBoosting (HGB) Won**:
- Handles the tabular rolling-feature representation naturally.
- Robust to missing values and outliers.
- Fast training and inference (important for real-time fleet monitoring).
- With proper feature engineering and target capping, achieves state-of-the-art results on C-MAPSS.

**Validation Protocol — 5-Fold Engine-Grouped Cross-Validation**:
- 100 FD001 engines split into 5 folds, each fold holding out 20 engines.
- Held-out engines are censored at fixed remaining-life horizons (10, 25, 50, 75, 100 cycles) to simulate real-world truncated sensor streams.
- Total: 500 pseudo-censored validation endpoints across all folds.

**PHM08 Asymmetric Scoring** (industry-standard metric):
- Late predictions (optimistic) are penalized MORE than early predictions (conservative).
- d = predicted - actual: if d < 0 → exp(-d/13) - 1; if d ≥ 0 → exp(d/10) - 1
- This makes engineering sense: predicting failure too late is dangerous; predicting it too early just means slightly premature maintenance.

---

### SLIDE 8: Model Results & Champion Selection

**Headline**: Validated Results — Honest Benchmarking with No Leakage

**FD001 Validation Champion** (5-fold engine-grouped, 500 censored endpoints):

| Model | RMSE | MAE | R² | PHM08 Mean |
|---|---|---|---|---|
| **HGB (cap=100, w=10/30)** | **7.576** | **5.390** | **0.946** | **0.985** |
| HGB (cap=125, w=10/30) | 10.063 | 6.799 | 0.905 | 2.032 |
| HGB (uncapped, w=10/30) | 16.484 | 9.596 | 0.745 | 94.983 |
| Random Forest (cap=100) | ~9.2 | ~6.5 | ~0.92 | ~1.8 |
| Ridge (cap=100) | ~12.5 | ~9.1 | ~0.85 | ~5.2 |

**Historical Official Test Benchmark** (FD001, 100 test engines — frozen, never used for tuning):

| Model | RMSE | MAE | R² | PHM08 Mean | Bias |
|---|---|---|---|---|---|
| HGB (cap=125) | 13.532 | 9.799 | 0.894 | 2.852 | +0.157 |
| HGB (cap=100) | 16.755 | 11.959 | 0.837 | 3.393 | -7.330 |

**Global Fleet Deployment Model** (All 4 subsets, 707 test engines):
- **Global HGB with Cap-200 + Condition Normalization**: Pooled RMSE **21.455**, MAE **14.622**, R² **0.824**, PHM08 mean **31.465**.
- This single model handles all operating conditions and fault modes across the entire fleet.

**Key Insight**: Target cap=100 gives the best validation scores on FD001, but cap=200 with condition normalization is optimal for multi-condition fleet deployment (FD002/FD004 have longer-lived engines).

---

### SLIDE 9: Anomaly Detection — Unsupervised Health Monitoring

**Headline**: Catching What RUL Models Miss — Early Degradation Signals

**Approach**: Windowed Autoencoder for Unsupervised Anomaly Detection

- **Architecture**: Linear autoencoder (D → 64 → 16 → 64 → D) — simple, fast, interpretable.
- **Training Data**: Only the first 20% of cycles from healthy development engines (early-life baseline).
- **Logic**: A healthy engine's sensor patterns can be reconstructed with low error. A degrading engine produces patterns the autoencoder hasn't seen → high reconstruction error.
- **Threshold**: Anomaly flagged when reconstruction MSE exceeds the 99th percentile (Q₀.₉₉) of healthy training data.

**Why This Matters**:
- RUL models predict "how long until failure" — but they need degradation to have started.
- Anomaly detection catches **abnormal behavior patterns** even before RUL models register decline.
- Together, they provide defense-in-depth: RUL for countdown + Anomaly for early warning.

---

### SLIDE 10: The Triage Agent — Autonomous Decision Making

**Headline**: Rule-Based Triage Agent — No LLM, 100% Deterministic & Auditable

**Why Rule-Based (Not LLM)?**:
- **Deterministic**: Same inputs always produce same outputs — critical for safety-critical maintenance decisions.
- **Auditable**: Every decision can be traced to explicit rules and thresholds.
- **Fast**: Sub-millisecond decisions vs. seconds for LLM calls.
- **No hallucination risk**: Rules don't make up maintenance procedures.

**Severity Tiers & Decision Logic**:

| Severity Level | Trigger Condition | Action | Assigned To |
|---|---|---|---|
| **EXCEPTION** | Stale/corrupt/non-finite telemetry | Escalate immediately | Human Systems Specialist |
| **L3 — CRITICAL** | RUL ≤ 15 cycles OR (RUL ≤ 30 AND persistent anomaly) | Ground machine, immediate maintenance | Maintenance Lead |
| **L2 — PLAN** | RUL ≤ 45 cycles OR anomalous sensor drift | Schedule hangar servicing | Planning Team |
| **L1 — INSPECT** | RUL ≤ 75 cycles OR single anomaly detection | Pre-flight line check | Line Technician |
| **L0 — MONITOR** | RUL > 75 cycles, healthy telemetry | Routine monitoring | Automated System |

**Anti-Alert-Fatigue Features**:
- **Idempotent work orders**: Unique keys `engine_{id}:{severity}` prevent duplicate tickets for the same condition.
- **Debounce/hysteresis**: Severity transitions require sustained confirmation, not single-cycle spikes.
- **Structured output**: JSON/CSV work orders with timestamp, engine ID, severity, recommended action, trigger rationale.

---

### SLIDE 11: MLOps — Monitoring, Drift Detection & Retraining

**Headline**: Production-Ready MLOps Pipeline

**Experiment Tracking**:
- **MLflow** logs all training runs, hyperparameters, metrics, and model artifacts.
- Every model artifact includes a `manifest.json` with SHA-256 checksums for tamper-proof provenance.

**Drift Monitoring**:
- **Population Stability Index (PSI)**: Compares sensor distributions between training baseline and incoming operational batches.
- **Drift Threshold**: PSI ≥ 0.25 triggers a drift alert.
- Monitors each of the 21 sensor channels independently.

**Automated Retraining Pipeline**:
1. New telemetry batch arrives → PSI computed per sensor.
2. If drift detected OR batch accumulation threshold met → trigger retraining.
3. New model trained on combined historical + new data.
4. Delta metrics compared: new model must IMPROVE on validation RMSE/MAE to be promoted.
5. Versioned model manifest generated; old model archived.

**Monitoring Dashboard** (Streamlit):
- Fleet overview: Active engines, predicted RUL, risk distribution.
- Engine deep-dive: Historical sensor trends, rolling degradation indicators.
- Model scorecards: RMSE, MAE, R², PHM08 leaderboards.
- Active work orders: Real-time view of maintenance tickets.

---

### SLIDE 12: Dataset Coverage Strategy

**Headline**: From Simple to Complex — Scaling Across the Fleet

**Progressive Complexity Strategy**:

```
FD001 (Simple)          →  FD002 (Medium)         →  FD003 (Medium)         →  FD004 (Hard)
1 condition, 1 fault       6 conditions, 1 fault      1 condition, 2 faults     6 conditions, 2 faults
────────────────────────────────────────────────────────────────────────────────────────────────
Primary dev/validation     Condition normalization     Multi-fault handling      Full fleet complexity
Target cap = 100           K-Means regime clustering   Dual failure patterns     Combined challenges
Single-regime model        Condition-aware z-scoring   Fault-aware features      Global unified model
```

**Key Design Decision**: Rather than building 4 separate specialist models, we built a **single global model** with condition-aware normalization that handles all subsets. This is more realistic for fleet deployment where you can't know which engine is in which regime a priori.

**We investigated a regime-routing approach** (routing engines to specialist models based on operating settings) but it **failed our strict cross-validation promotion gate** (pooled RMSE 20.797 vs global model). In adherence to disciplined ML principles, we rejected it. This is a strength — it shows we don't cherry-pick results.

---

### SLIDE 13: Testing & Quality Assurance

**Headline**: Comprehensive Automated Test Suite

**Test Coverage**:
- **Leakage Tests**: Verify causal features never access future cycles, initial-difference features don't leak, low-variance screening uses train-only data.
- **Condition Normalizer Tests**: Verify K-Means is fit on training data only, applied consistently at inference.
- **Triage Tests**: Verify work order idempotency (no duplicates), severity tier thresholds, debounce behavior.
- **PHM08 Metric Tests**: Mathematical exactness validated against known analytical values (d=-13 → e-1, d=10 → e-1, d=0 → 0).
- **Manifest Integrity Tests**: SHA-256 corruption detection — tampering with model artifacts is caught.
- **Sequence Window Tests**: Verify temporal windows never cross engine boundaries.

---

### SLIDE 14: Technology Stack

**Headline**: Lean, Open-Source, Production-Ready

| Layer | Technology | Why |
|---|---|---|
| **Data Engineering** | Python, Pandas, NumPy | Industry standard, fast prototyping |
| **Machine Learning** | scikit-learn (HGB, RF, Ridge), PyTorch (CNN, LSTM, Autoencoder) | Best-in-class tabular + deep learning |
| **MLOps** | MLflow (experiment tracking), PSI drift monitoring | Open-source, extensible |
| **Dashboard** | Streamlit | Rapid interactive UI, Python-native |
| **Agent** | Pure Python rule-based logic | No LLM dependency, deterministic |
| **Testing** | pytest | Comprehensive automated validation |
| **Edge/TinyML** | PyTorch int8 quantization | Future-ready for on-device deployment |

---

### SLIDE 15: Summary & Key Differentiators

**Headline**: Why NADI Stands Out

**Our Key Differentiators**:

1. **🔒 Zero Data Leakage**: Every feature, every transform, every split is rigorously isolated. We prove it with automated tests.

2. **📊 Honest Benchmarking**: We separate validation champions from historical test benchmarks. Test labels are frozen and never used for model selection.

3. **🤖 Deterministic Triage Agent**: No LLM — pure rules. Every decision is traceable, reproducible, and sub-millisecond fast.

4. **🌐 Global Fleet Model**: One model handles all 4 C-MAPSS subsets (707 engines, 6 operating conditions, 2 fault modes) through condition-aware normalization.

5. **🛡️ Defense in Depth**: RUL prediction (supervised) + Anomaly detection (unsupervised) + Rule-based triage = triple safety net.

6. **📈 Production MLOps**: Drift detection, automated retraining triggers, versioned manifests with tamper-proof SHA-256 checksums.

7. **🔬 Disciplined ML**: We explored regime routing but rejected it when it failed our promotion gate. We don't cherry-pick.

**Bottom Line**: NADI doesn't just predict failure — it closes the loop from raw sensor stream to maintenance work order, autonomously, reliably, and with full auditability.

---

## ADDITIONAL CONTEXT FOR PPT DESIGN

**Color Palette Suggestion** (matching Teliport branding):
- Primary Blue: #1A73E8 (headers, key elements)
- Accent Orange: #FF6D00 (highlights, alerts, call-to-actions)
- Dark Background: #1A1A2E or #0F1923 (for dark-themed slides)
- Text White: #F5F5F5
- Success Green: #00C853 (for healthy/good metrics)
- Warning Amber: #FFD600 (for caution states)
- Danger Red: #FF1744 (for critical states)

**Architecture Diagram Style**: Use clean boxes with rounded corners, connecting arrows, and layer labels. Icons for each component (gear for data engine, brain for ML, shield for triage, chart for dashboard).

**Slide Design Tips**:
- Use full-bleed dark backgrounds with light text for a premium feel.
- Include the NADI logo/wordmark prominently.
- Use large, bold numbers for key metrics (RMSE 7.576, R² 0.946).
- Include small "key insight" callout boxes on technical slides.
- End with a strong, confident closing statement.
