# Research Proposal: DP-FedLoRA
## Differentially Private Federated Fine-Tuning of Vision-Language Models for Multi-Site Medical Imaging

**PI:** Dr. Arulmurugan Ramu, Heriot-Watt University Kazakhstan  
**Target venue:** ICLR 2027 / NeurIPS 2026 (ML+Privacy track)  
**Estimated submission window:** 12–14 months  
**Status:** Pre-experiment — all claims below are hypotheses until experimental evidence is collected

---

## One-Sentence Contribution

We propose DP-FedLoRA, a federated fine-tuning framework that provides the first *formal* (ε, δ)-differential privacy guarantee for low-rank adapter updates across non-IID medical imaging sites, matching full-model fine-tuning accuracy within 3% while reducing per-round communication cost by more than 90%.

---

## 1. Problem and Motivation

Foundation models — vision transformers, vision-language models (VLMs) — are transforming medical image analysis. Adapting them to clinical tasks requires data from multiple hospital sites to achieve the diversity needed for generalisation. Hospitals cannot pool raw patient data: HIPAA and GDPR prohibit centralisation, and cross-border transfer is operationally infeasible.

Federated learning (FL) is the natural answer. In FL, each site trains locally and shares only model updates. The problem is that gradient sharing is not private: gradient inversion attacks reconstruct patient images directly from shared updates [Sensors 2025, sensitivity-aware DP paper]. The standard defence — differentially private stochastic gradient descent (DP-SGD) — adds calibrated noise to gradients. However, applying DP-SGD to a full VLM at each round is prohibitively expensive: (i) communication cost scales with model size (hundreds of millions of parameters), and (ii) the noise required for formal DP at the full-gradient scale causes clinically unacceptable accuracy loss, particularly on small heterogeneous hospital cohorts [npj Digital Medicine 2025].

The field has responded to the communication problem with parameter-efficient fine-tuning (PEFT): LoRA freezes the backbone and trains small low-rank adapter matrices at each site, reducing the transmitted gradient from ~300M to ~1M parameters. This is already used informally in federated biomedical LLM work [BioData Mining 2024]. However, no existing work analyses whether DP noise applied to low-rank adapters yields a *tighter formal privacy budget than full-model DP*, whether the interaction between DP noise and low-rank structure disrupts convergence under realistic non-IID site distributions, or where the privacy-utility frontier sits for adapter-only DP relative to full-model DP. These are the three open questions this project answers.

---

## 2. Prior Work and the Gap

**Federated fine-tuning of foundation models** is emerging. Informal adapter-based FL has been applied in FedMed and related systems for clinical NLP [BioData Mining 2024], but without privacy proofs. Open Challenges in Federated Foundation Models [arXiv 2405.06784] explicitly identifies formal privacy analysis for adapters as an unresolved problem.

**DP for medical imaging** is well-studied at the empirical level: a scoping review of 74 studies (npj Digital Medicine 2025) shows that ε≈10 is clinically tolerable but ε≈1 produces substantial accuracy degradation, especially on small or heterogeneous cohorts. No study tests DP specifically on adapter parameters rather than full model gradients.

**Non-IID robustness** has been addressed independently of DP (Fed-NGA, RAGA, ScienceDirect 2025). The joint problem — DP + non-IID convergence for PEFT — is unaddressed in any surveyed 2024–2026 paper. [CLAIM: systematic arXiv sweep needed to confirm this gap before submission.]

**This project's gap:** There is no unified treatment of DP guarantees, PEFT communication efficiency, and non-IID convergence for VLM fine-tuning in federated healthcare settings.

---

## 3. Proposed Method: DP-FedLoRA

DP-FedLoRA has three technical components.

### 3.1 Calibrated DP Noise for LoRA Adapter Updates

In standard DP-SGD, noise is added to full gradients. DP-FedLoRA applies the Gaussian mechanism only to the adapter matrices {A, B} per LoRA layer. Because the adapter is low-rank (rank r ≪ d), the L2 sensitivity of the update is bounded by the adapter norm rather than the full gradient norm. We conjecture this yields a tighter ε for the same noise scale σ, providing a better privacy-utility tradeoff than full-model DP-SGD. The formal analysis will use the Rényi DP accountant (RDP) to compose privacy across communication rounds and FL clients.

**Key theoretical question:** Does the reduced dimensionality of LoRA updates produce a provably tighter RDP bound? The answer depends on whether gradient clipping applied to the adapter interacts differently with the low-rank structure. We will derive the bound analytically and verify numerically.

### 3.2 Convergence Under Non-IID Site Distributions

Medical imaging data is highly heterogeneous across sites (scanner differences, patient demographics, disease prevalence). We will model this as a Dirichlet(α) partition of labels and derive convergence bounds for DP-FedLoRA under varying α. The analysis extends the FedProx convergence proof to the noisy, low-rank update setting. We identify the noise-heterogeneity tradeoff: more non-IID data requires more rounds to converge, but more rounds accumulate more DP noise. DP-FedLoRA will include an adaptive round-budget mechanism that terminates early when the marginal utility per round falls below a threshold.

### 3.3 Communication Efficiency

Adapter size for rank r=16 on a ViT-L/16 backbone is approximately 0.8M parameters vs. 307M for the full model — a 383× reduction in transmitted gradient size. We quantify the actual wall-clock and bandwidth savings across a simulated 10-site FL federation and compare to FedAvg, FedProx, and FedDF baselines.

---

## 4. Experimental Plan

**Datasets (all public, no IRB required for this phase):**

| Dataset | Task | Non-IID simulation |
|---|---|---|
| MedMNIST PathMNIST | Colorectal tissue classification (9 classes) | Dirichlet α ∈ {0.1, 0.5, 1.0} |
| MedMNIST ChestMNIST | Chest X-ray multi-label classification | Dirichlet α ∈ {0.1, 0.5, 1.0} |
| MedMNIST DermaMNIST | Dermatoscopy 7-class | Dirichlet α ∈ {0.1, 0.5, 1.0} |
| RSNA Pneumonia (Kaggle) | Binary detection | Hospital-split by institution ID |

**Foundation model backbone:** BioViL-T or BiomedCLIP (open weights, healthcare-pretrained VLM). Fallback: ViT-B/16 pretrained on ImageNet.

**Baselines:**

| Method | Privacy | Communication |
|---|---|---|
| Centralised fine-tuning + DP-SGD | Formal ε | N/A (upper bound) |
| FedAvg (no DP) | None | Full model |
| FedAvg + full-model DP-SGD | Formal ε | Full model |
| FedAvg + LoRA (no DP) | None | Adapter only |
| **DP-FedLoRA (proposed)** | **Formal ε** | **Adapter only** |

**Primary metrics:** Top-1 accuracy (or AUC for multilabel), formal ε at δ=10⁻⁵, per-round communication cost (MB), total communication to convergence.

**Success criteria (from Card A):**
- DP-FedLoRA accuracy within ≤3% of centralised DP fine-tuning on ≥2 datasets.
- Formal ε ≤ 8 at δ = 10⁻⁵ without accuracy collapse.
- Communication cost ≤ 10% of full-model FedAvg + DP-SGD at equivalent rounds.

**Framework:** Flower (flwr) for FL orchestration, Opacus for DP-SGD, HuggingFace PEFT for LoRA. All code will be released under MIT licence.

---

## 5. Anticipated Contributions

1. **Theoretical:** First formal RDP analysis of adapter-only DP noise in a federated fine-tuning setting, with convergence bounds under non-IID Dirichlet distributions.
2. **Algorithmic:** DP-FedLoRA algorithm with adaptive round budget and privacy accountant.
3. **Empirical:** Privacy-utility-communication frontier on three MedMNIST tasks + RSNA, establishing whether adapter-only DP dominates full-model DP-SGD in the healthcare FL setting.
4. **Open source:** Reproducible codebase and pre-computed baselines for the community.

The paper will make a falsifiable claim: *either* DP-LoRA yields a tighter formal ε than full-model DP-SGD under equivalent accuracy, *or* it does not, and we will report the negative result with analysis.

---

## 6. Timeline

| Month | Milestone |
|---|---|
| 1–2 | Environment setup; reproduce FedAvg + DP-SGD on MedMNIST baselines |
| 3–4 | Implement DP-FedLoRA; derive RDP bound for adapter updates (first version) |
| 5–6 | Non-IID convergence experiments (Dirichlet α sweep); refine theoretical bounds |
| 7–8 | RSNA experiments; ablation studies (rank r, noise σ, non-IID level α) |
| 9–10 | Theory write-up; paper draft |
| 11–12 | Internal review + revision; submission to NeurIPS 2026 or ICLR 2027 |

**Minimum viable result (6-month checkpoint):** Empirical evidence that DP-FedLoRA achieves better accuracy than full-model DP-FedAvg at the same formal ε on PathMNIST under Dirichlet α=0.1. This alone justifies continuing.

---

## 7. Risks and Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| DP noise + LoRA rank interact adversarially (theoretical gap cannot be closed) | Medium | Fall back to empirical privacy claim only; shift venue to applied track |
| BiomedCLIP licence restrictions | Low | Switch to ViT-B/16 + ImageNet pretraining |
| Non-IID simulation too synthetic for reviewer acceptance | Medium | Add RSNA real multi-site split as a second evaluation |
| Related work found mid-project that closes the gap | Low | Monitor arXiv weekly via keyword alert; pivot to extension |

---

## 8. Venue Justification

**First choice: ICLR 2027 (submission ~Oct 2026)**  
ICLR rewards theory + empirics combinations. ML+Privacy is an established interest area. Page limit: 9 pages.

**Second choice: NeurIPS 2026 (submission ~May 2026)**  
If baseline results are strong by month 5, NeurIPS is feasible. Requires tighter theory write-up earlier.

**Third choice: AAAI 2027**  
Lower theory bar; accepts applied FL + healthcare papers. Fallback if theory does not close cleanly.

---

*Citations in this document reference papers found via web search (May 2026). Full BibTeX must be fetched programmatically and verified before manuscript submission. Marked [CLAIM] items require a systematic arXiv sweep at the start of Phase 3.*
