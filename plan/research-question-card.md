# Research Question Cards — FL in Healthcare

Three candidate directions ranked by fit with PI background (AI/CV/NLP) and gap strength.

---

## Card A (Recommended) — Privacy-Formal Federated Fine-Tuning of Vision-Language Models for Medical Imaging

**Hypothesis:**  
Parameter-efficient federated fine-tuning (LoRA / prefix-tuning) of a vision-language foundation model can achieve clinically acceptable performance on multi-site medical imaging tasks while providing *formal* (ε, δ)-differential privacy guarantees, even under non-IID site distributions.

**Why this is open:**  
Existing work uses LoRA in FL to reduce communication cost but provides no formal DP bound for the adapter updates. Gradient inversion attacks specifically exploit adapter sparsity (gradient inversion on LoRA updates is understudied). Formal DP + LoRA + non-IID convergence is not addressed in any 2024–2026 paper found.

**Current evidence:**  
- LoRA in FL: informal use in FedMed and related; no DP analysis (BioData Mining 2024).  
- DP-SGD for medical imaging: ε≈10 acceptable, ε≈1 problematic (npj 2025).  
- Non-IID + Byzantine: solvable separately, not jointly with DP (ScienceDirect 2025).

**Missing evidence:**  
- Does DP-LoRA converge under non-IID medical imaging distributions?  
- What is the privacy-utility frontier for adapter-only DP vs. full-model DP?  
- Which medical imaging tasks (pathology, radiology, dermatology) show the sharpest non-IID effect?

**Support criteria (what "works" means):**  
- DP-LoRA matches full fine-tuning within ≤3% on at least 2 public medical imaging benchmarks.  
- Formal ε≤8 at δ=10⁻⁵ with no accuracy collapse.  
- Outperforms FedAvg + full-model DP-SGD on communication budget.

**Falsification criteria:**  
- DP-LoRA provably cannot tighten the DP bound due to adapter sparsity structure.  
- Or: convergence under non-IID requires full-model gradient sharing, negating adapter savings.

**Minimal next action:**  
Reproduce FedAvg + DP-SGD on MedMNIST (PathMNIST, ChestMNIST) to establish the non-IID accuracy baseline. Estimated time: 1–2 weeks with existing FL framework (Flower or PySyft).

**Target venue:** ICLR / NeurIPS (ML+Privacy track) / AAAI  
**Estimated timeline:** 10–14 months to submission-ready paper

---

## Card B — LLM-Assisted Semantic Harmonisation for Privacy-Preserving FL on EHR

**Hypothesis:**  
An LLM agent that resolves ontology mismatches (ICD-9 vs ICD-10, SNOMED, local coding) across hospital EHR systems, operating *inside* the FL pipeline, can improve downstream task performance without requiring raw data sharing or violating differential privacy.

**Why this is open:**  
The Frontiers Digital Health paper (May 2026) introduces this idea but has no privacy analysis and no benchmark. This is a genuine first-mover opportunity.

**Current evidence:**  
- Ontology + LLM harmonisation for FL: Frontiers 2026 (very early, no privacy proof).  
- EHR FL in general: FED-EHR (2025) addresses privacy but not semantic heterogeneity.

**Missing evidence:**  
- Can LLM harmonisation be made differentially private?  
- Does harmonisation help more than simple one-hot encoding on standard EHR tasks (mortality, readmission)?

**Support criteria:**  
- Harmonised FL outperforms raw FL by ≥5% AUC on a 3-site EHR split.  
- Privacy cost of LLM harmonisation step is formally bounded.

**Falsification criteria:**  
- LLM harmonisation introduces inference leakage (the harmonisation queries reveal local coding distributions).

**Minimal next action:**  
Set up MIMIC-III multi-site simulation with ICD-9 → ICD-10 code mixing. Use GPT-4o or a local LLaMA to map codes. Measure raw FL vs. harmonised FL gap.

**Target venue:** KDD / ACL (clinical NLP track) / Nature Digital Medicine  
**Estimated timeline:** 12–18 months

---

## Card C — Byzantine-Robust + Differentially Private FL for Multi-Site Clinical Trials

**Hypothesis:**  
A unified aggregation rule can be simultaneously (ε, δ)-differentially private and Byzantine-robust under arbitrary non-IID distributions across hospital sites, without sacrificing convergence guarantees.

**Why this is open:**  
Existing solutions address robustness (Fed-NGA, RAGA) or privacy (DP-SGD), but not both under non-IID medical distributions. The ScienceDirect 2025 paper addresses robustness + non-IID but not DP.

**Minimal next action:**  
Survey Fed-NGA + Trimmed Mean + DP-SGD composition. Draft a theoretical proof sketch for whether DP + robustness are jointly achievable under Byzantine fraction < 1/3.

**Target venue:** NeurIPS / ICML (theory track)  
**Estimated timeline:** 14–18 months (heavier theory burden)
