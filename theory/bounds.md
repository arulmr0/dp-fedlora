# Formal Privacy and Utility Analysis of DP-FedLoRA

**Status:** First draft (Month 3–4). Claims are proven at the level of direct application of known results. The convergence theorem (Section 3) requires full proof before submission.

---

## 1. Setup and Notation

Let there be $N$ hospital clients, each holding $n_i$ training samples. In each FL round, each client:
1. Receives global adapter parameters $\{\mathbf{A}_l, \mathbf{B}_l\}_{l=1}^L$ from the server.
2. Runs $E$ local epochs with batch size $B$, giving $S = \lfloor n_i / B \rfloor \cdot E$ local steps per round.
3. Applies DP-SGD (Opacus, Gaussian mechanism) to the adapter gradients only.
4. Sends clipped, noised adapter updates back to the server.

**Parameters:**

| Symbol | Meaning | Our value |
|--------|---------|-----------|
| $\sigma$ | Noise multiplier | 1.1 |
| $C$ | Per-sample gradient clip norm | 1.0 |
| $\delta$ | DP failure probability | $10^{-5}$ |
| $q = B/n_i$ | Per-step sampling rate | $\approx 3.56 \times 10^{-3}$ |
| $S$ | Steps per round per client | 843 (full scale) |
| $T$ | Total FL rounds | 20 |
| $d$ | Full ViT-B/16 parameter count | 85,805,577 |
| $k$ | Adapter parameter count (r=16, q/v proj, 12 layers) | 589,824 |
| $k/d$ | Adapter fraction | $\approx 6.87 \times 10^{-3}$ |

---

## 2. Theorem 1: Privacy Equivalence

**Theorem 1 (Privacy Equivalence).** *Under the Gaussian mechanism applied via Opacus DP-SGD with noise multiplier $\sigma$, gradient clip norm $C$, sampling rate $q$, and $S \cdot T$ total steps per client, DP-FedLoRA and full-model DP-FedAvg satisfy the same $(\varepsilon, \delta)$-differential privacy guarantee.*

**Proof sketch.** The RDP guarantee of the Sampled Gaussian Mechanism at Rényi order $\alpha$ is (Mironov 2017, Wang et al. 2019):

$$\varepsilon_{\text{RDP}}(\alpha) = \frac{\alpha}{2\sigma^2} + O(q^2)$$

This depends only on $\sigma$, $q$, and the number of composition steps. The L2 sensitivity $C$ determines the absolute noise standard deviation ($\sigma C$) but cancels in the ratio $\sigma = \text{noise\_std} / C$ used by RDP. Since DP-FedLoRA and full-model DP-FedAvg both use the same $\sigma$, $q$, and total step count, their RDP curves are identical. Converting to $(\varepsilon, \delta)$-DP via the standard RDP-to-DP conversion gives the same $\varepsilon$ for both.

**Numerical verification** (see `theory/rdp_analysis.py`):

| Configuration | $\varepsilon$ (RDP tight) | $\alpha^*$ |
|---|---|---|
| Smoke test (3 rounds, 2 clients, fast\_dev\_run=2) | 0.639 | 18.0 |
| CPU baseline (8 rounds, 3 clients, fast\_dev\_run=2) | 0.683 | 16.5 |
| Full scale (20 rounds, 10 clients, full data) | **2.614** | 9.5 |

At full scale, $\sigma = 1.1$ achieves $\varepsilon \approx 2.61 \ll 8$, with substantial privacy budget remaining.

**Sigma budget** (full scale, $T=20$, $n_i=9{,}000$, $B=32$):

| Target $\varepsilon$ | Required $\sigma$ |
|---|---|
| 1.0 | 2.2944 |
| 2.0 | 1.3141 |
| 4.0 | 0.8738 |
| 8.0 | 0.6697 |
| 10.0 | 0.6245 |

Our $\sigma = 1.1$ gives $\varepsilon \approx 2.61$. To achieve $\varepsilon = 8$, we could relax to $\sigma \approx 0.67$, potentially recovering accuracy at the cost of looser privacy.

---

## 3. Proposition 1: Noise Energy Advantage

**Proposition 1 (Noise Energy Advantage).** *At the same $(\varepsilon, \delta)$-DP guarantee, DP-FedLoRA injects a total noise energy of $\sigma^2 C^2 k$ per round, compared to $\sigma^2 C^2 d$ for full-model DP-FedAvg. The reduction factor is $d/k \approx 145$.*

**Derivation.** In DP-SGD, the Gaussian mechanism adds noise $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 C^2 \mathbf{I}_m)$ to the gradient in $\mathbb{R}^m$, where $m$ is the number of trainable parameters. The expected squared noise norm is:

$$\mathbb{E}[\|\boldsymbol{\eta}\|_2^2] = \sigma^2 C^2 m$$

For DP-FedLoRA, $m = k = 589{,}824$. For full-model DP-FedAvg, $m = d = 85{,}805{,}577$. Thus:

$$\frac{\mathbb{E}[\|\boldsymbol{\eta}_{\text{full}}\|_2^2]}{\mathbb{E}[\|\boldsymbol{\eta}_{\text{LoRA}}\|_2^2]} = \frac{d}{k} \approx 145$$

**Implication for convergence.** Convergence bounds for DP-SGD in federated settings (e.g., Levy et al. 2021) typically include a noise floor term proportional to $\sigma^2 C^2 m / (T B n_i)$. Substituting $m = k$ for DP-FedLoRA vs $m = d$ for full-model DP gives a $d/k \approx 145\times$ improvement in the noise floor at the same $\varepsilon$.

**Caveat.** This comparison holds the number of optimized parameters fixed within each method. Full-model DP optimizes a richer function class than adapter-only DP, so the comparison is with respect to the noise floor (variance) term in the convergence bound, not the bias term. The total suboptimality gap also includes a bias from the restricted adapter hypothesis class; the empirical results will quantify the net effect.

---

## 4. Corollary: Communication–Privacy–Utility Frontier

**Corollary 1.** *DP-FedLoRA simultaneously achieves: (i) identical $(\varepsilon, \delta)$-DP to full-model DP-FedAvg, (ii) $d/k \approx 145\times$ less injected noise energy per round, and (iii) $d/k \approx 145\times$ less per-round communication cost.*

The coincidence that the noise energy and communication reductions are both $d/k$ is not accidental: both scale with the number of trainable parameters.

**Communication costs:**

| Method | Params transmitted/round | Bandwidth |
|---|---|---|
| Full-model FedAvg + DP-SGD | $d = 85{,}805{,}577$ | 343.2 MB |
| DP-FedLoRA (r=16) | $k = 589{,}824$ | 2.36 MB |
| Reduction | — | **145× less** |

---

## 5. Convergence Theorem (Sketch — to be formalised)

**Theorem 2 (DP-FedLoRA Convergence — informal).** *Under $L$-smooth, $G_A$-Lipschitz adapter losses, assuming the backbone is frozen after LoRA injection and each client runs $S$ steps of DP-SGD per round with noise multiplier $\sigma$ and clip norm $C$, the average gradient norm after $T$ rounds satisfies:*

$$\frac{1}{T} \sum_{t=1}^T \mathbb{E}\left[\|\nabla F(\bar{\theta}^t)\|^2\right] \leq \frac{2(F(\bar{\theta}^0) - F^*)}{\eta T} + \underbrace{\frac{L \eta \sigma^2 C^2 k}{S B n}}_{\text{noise floor}}$$

*where $\bar{\theta}^t$ is the global adapter at round $t$, $\eta$ is the learning rate, and $n = \min_i n_i$.*

The noise floor scales as $k/d$ of the full-model DP bound, giving the utility advantage claimed in Proposition 1.

**Proof strategy:**
1. Apply the standard DP-FedAvg convergence framework (McMahan et al. 2017, Levy et al. 2021).
2. Replace the full gradient noise variance $\sigma^2 C^2 d$ with $\sigma^2 C^2 k$ for adapter-only DP.
3. Account for the non-IID distribution via the gradient divergence term $\Gamma = \|\nabla F - \nabla F_i\|$ under Dirichlet($\alpha$) partitioning.

Full proof deferred to Month 5–6 alongside the non-IID convergence analysis.

---

## 6. Open Questions for Month 5–6

1. **Non-IID interaction:** How does the noise floor $\sigma^2 C^2 k / (TBn)$ interact with the non-IID gradient divergence $\Gamma(\alpha)$ under Dirichlet($\alpha$) partitioning? Does DP noise effectively "regularise" the non-IID divergence, or compound it?

2. **Optimal rank $r$:** Is there an optimal adapter rank $r^* = \arg\min_r [\text{bias}(r) + \text{noise\_floor}(r)]$ that balances expressivity against noise injection? The noise floor scales as $k(r) = 2r(d_\text{in}+d_\text{out})L$, which is linear in $r$. The bias from rank restriction presumably decreases with $r$.

3. **Clipping norm calibration:** In full-model DP-SGD, many gradients get clipped (‖∇θ ℓ‖ > C), wasting noise budget. For adapter-only DP, do per-sample adapter gradients have smaller L2 norms on average? If ‖∇_{A,B} ℓ‖_2 ≤ C_A ≪ C on average, we can tighten the sensitivity assumption.

4. **Composition across heterogeneous clients:** With Dirichlet partitioning, different clients have different $n_i$, hence different $q_i$ and $\varepsilon_i$. The privacy guarantee is determined by the worst-case client. The analysis should track per-client privacy and aggregate to a group privacy guarantee.

---

## 7. References

- Mironov (2017). Rényi differential privacy. *IEEE CSF*.
- Wang et al. (2019). Subsampled Rényi differential privacy and analytical moments accountant. *AISTATS*.
- Levy et al. (2021). Learning with user-level privacy. *NeurIPS*.
- McMahan et al. (2017). Communication-efficient learning of deep networks. *AISTATS*.
- Hu et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
- Abadi et al. (2016). Deep learning with differential privacy. *CCS*.
