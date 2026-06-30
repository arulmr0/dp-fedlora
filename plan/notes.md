# Notes: FL in Healthcare — Literature Evidence

## Surveys (2024–2026)

### Comprehensive PMC Survey (250+ articles, 2019–2024)
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/
- Covers: system architecture, federation scale, data partitioning, aggregation algorithms.
- Key finding: No standardised benchmark suite exists for healthcare FL evaluation.

### Open Challenges in Federated Foundation Models (BioData Mining, 2024)
- URL: https://biodatamining.biomedcentral.com/articles/10.1186/s13040-024-00414-9
- Key finding: FedFM for biomedical data is an open frontier. Fine-tuning large models in FL settings requires solving: (1) communication cost of large model updates, (2) privacy leakage from gradient sharing, (3) data heterogeneity across institutions.

### Healthcare FL Survey — Applications + Frameworks (IJCA, 2025)
- URL: https://www.tandfonline.com/doi/full/10.1080/1206212X.2025.2496913
- Key finding: Model interpretability and regulatory compliance remain critical unresolved gaps.

### Decentralized FL for Smart Healthcare (MDPI Mathematics, 2025)
- URL: https://www.mdpi.com/2227-7080/13/8/1296
- Key finding: Decentralised (serverless) topologies reduce single-point-of-failure but introduce new consensus and convergence problems.

### FL for Healthcare 5.0 (Soft Computing / Springer, 2025)
- URL: https://link.springer.com/article/10.1007/s00500-025-10508-z
- Key finding: IoT + FL integration creates resource-constrained edge settings needing lightweight FL.

---

## Privacy and Differential Privacy

### Sensitivity-Aware DP for Federated Medical Imaging (Sensors, 2025)
- URL: https://www.mdpi.com/1424-8220/25/9/2847
- Key finding: Gradient inversion attacks defeat naive FL. Sensitivity-aware DP improves privacy-utility trade-off over standard DP-SGD.

### DP for Medical Deep Learning — Scoping Review (npj Digital Medicine, 2025)
- URL: https://www.nature.com/articles/s41746-025-02280-z
- Key finding: DP-SGD under ε≈10 is acceptable clinically; ε≈1 causes significant accuracy loss, especially on small or heterogeneous datasets. 74 studies reviewed through March 2025.

### Privacy Preservation for FL in Healthcare (PMC, 2024)
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11284498/
- Key finding: Combining DP + secure aggregation is common but their interaction on non-IID medical data is understudied.

---

## Byzantine Robustness and Non-IID Heterogeneity

### Byzantine-Robust FL with Non-IID Data (ScienceDirect, 2025)
- URL: https://www.sciencedirect.com/science/article/abs/pii/S156625352500418X
- Key finding: Most robust aggregators (Krum, Trimmed Mean) degrade under non-IID. New 2025 method addresses both jointly.

### Fed-NGA — Byzantine + Heterogeneity (arXiv 2408.09539)
- URL: https://arxiv.org/abs/2408.09539
- Key finding: Normalised gradient aggregation converges to stationary points under Byzantine faults + non-IID.

### FL Security + Privacy Review (Int. J. Data Science Analytics, 2026)
- URL: https://link.springer.com/article/10.1007/s41060-026-01067-z
- Key finding: Critical trade-off between robustness, privacy, computation, and convergence under non-IID — no unified solution.

---

## Foundation Models + LLMs in FL

### Open Challenges in Federated Foundation Models (arXiv 2405.06784)
- URL: https://arxiv.org/pdf/2405.06784
- Key finding: Adapter-based fine-tuning (LoRA) reduces communication cost in FedFM but privacy guarantees for adapters remain informal.

### Ontology + LLM-based Data Harmonisation for FL in Healthcare (Frontiers, May 2026)
- URL: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1756555/full
- Key finding: Very new (May 2026). Using LLM agents to resolve ICD/SNOMED coding differences before FL training. Early-stage, no benchmark.

### FL Healthcare — Model Misconduct + Security Systematic Review (arXiv 2405.13832)
- URL: https://arxiv.org/html/2405.13832v1
- Key finding: Foundation model backdoor attacks in FL are largely unstudied in healthcare context.

---

## Regulatory Compliance

### Federated Learning and Regulatory Cooperation (Frontiers Drug Safety, 2025)
- URL: https://www.frontiersin.org/journals/drug-safety-and-regulation/articles/10.3389/fdsfr.2025.1579922/full
- Key finding: FL "aligns with" GDPR/HIPAA but formal provable compliance is not established. The gap between alignment and proof is open.

### FED-EHR (MDPI Electronics, 2025)
- URL: https://www.mdpi.com/2079-9292/14/16/3261
- Key finding: Privacy-by-Design FL for IoMT EHR, but evaluation is narrow (single dataset).

---

## Synthesis: Where the Gaps Are

| Gap | Evidence | Tractability |
|-----|----------|-------------|
| Formal DP for federated fine-tuning of VLMs/LLMs | BioData Mining 2024, npj 2025 | Medium — needs theory + experiment |
| Byzantine + non-IID joint solution for healthcare | ScienceDirect 2025, arXiv 2408 | Medium — existing baselines available |
| Ontology/LLM harmonisation + FL privacy | Frontiers 2026 (very new) | High — first-mover advantage |
| Standardised FL healthcare benchmark | PMC survey 2024 | High — engineering + community work |
| Backdoor attacks on federated foundation models | arXiv 2405.13832 | Medium — security angle |
| Personalized FL for heterogeneous patient populations | Multiple surveys | Lower — crowded but fundable |
