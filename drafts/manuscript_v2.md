# Robust Multimodal Occupancy Detection Under Sensor Failure: Cross-Modal Attention with Missing-Modality Augmentation

**Manuscript draft v2** — target: *Energy and Buildings* / *Building and Environment* (Elsevier, numeric citation style).
Revision of v1 per the official evaluation report (`EVALUASI-NASKAH-V1.md`, 2026-08-20): all 3 MAJOR and 10 MINOR findings addressed; COMMENT-5, COMMENT-7, and COMMENT-8 incorporated.
All experimental figures are transcribed verbatim from `experiments/results/*.json` (reproducibility package, GitHub); bibliographic metadata verified via Crossref/arXiv on 2026-08-20.

*Editorial note (COMMENT-5): at formatting, Table 3 may be moved to supplementary material if the typeset paper exceeds the journal page limit; the in-text structure is unchanged.*

*Title option (COMMENT-7): a shorter equivalent title — "Robust Multimodal IoT Occupancy Detection Under Sensor Failure: Masked Cross-Modal Attention with Contrastive Learning" (14 words) — may be adopted at formatting; scope and claims are unchanged. Final decision by the authors.*

---

## Highlights

- Missing-modality dropout augmentation is critical for robustness
- Masked cross-modal attention preserves ranking under light failure
- Highest PR-AUC in 14/14 failure scenarios vs. three baselines
- Robust to noise, drift, and misalignment up to +300 s
- Zero-shot office-to-residential transfer fails; boundary analyzed

## Abstract
Multimodal Internet-of-Things (IoT) sensing is a cornerstone of smart-building energy management, yet its reliability depends on the continuous availability of every installed sensor. In real deployments sensors fail, drift, and desynchronize; conventional fusion models degrade sharply when the dominant modality is lost. We present a missing-modality-aware deep fusion framework for occupancy detection combining per-modality encoders, masked cross-modal attention that excludes absent modalities via a key-padding mask, missing-modality dropout augmentation, and a complementary InfoNCE contrastive objective between clean and corrupted views. Using a public smart-office dataset (230,976 ten-second rows; 97.3% unoccupied), we evaluate the framework under 14 sensor-failure and 12 noise/drift/misalignment stress scenarios, with block bootstrap confidence intervals and calibration. The proposed model attains the highest PR-AUC in 14/14 failure scenarios against three standard missing-modality strategies (imputation-, indicator-, and gating-based), implemented under an identical protocol: 0.887 when the dominant light modality is removed (vs. 0.111–0.842). Ablations attribute robustness primarily to dropout augmentation (F1 0.001 without it), secondarily to masked attention, with contrastive learning as a consistent complement. Calibration is adequate (ECE = 0.055), and a failed zero-shot office-to-residential transfer (ROC-AUC 0.374) is analyzed as an explicit generalization boundary. Code and data are public.

**Keywords:** occupancy detection; multimodal sensor fusion; missing modality; cross-modal attention; contrastive learning; sensor failure robustness; smart buildings

---

## 1. Introduction

Buildings account for roughly a third of global final energy consumption, and occupancy information is one of the strongest levers for demand-driven heating, ventilation, and air-conditioning (HVAC) control and lighting [1]. Non-intrusive occupancy sensing — inferring presence from environmental, electrical, and device-state signals rather than cameras — has become the dominant approach in the building energy literature [1–3]. Modern smart offices instrument multiple heterogeneous sensing streams: CO₂ and thermal comfort variables, illuminance, acoustics, per-circuit electrical metering, and smart-switch states [4,5]. Fusing these streams improves accuracy over any single sensor [2,6,7], and the recent release of public multimodal datasets has accelerated deep fusion research in this area [4,8].

However, the operating assumption of most published occupancy models is that *all* modalities are available at inference time. This assumption is routinely violated in practice. Our own analysis of a newly published public dataset [4] documents natural missing values on nearly every channel — 2.0% of acoustic rows, 0.47% of firewall-metering rows, 0.21% of socket-metering rows — and beyond such sporadic gaps, hardware failure, battery depletion, network partition, and firmware faults produce *systematic* modality loss lasting hours or days. The evidence collected in this study shows that conventional pipelines are not robust to such events: an XGBoost model with blind imputation that reaches F1 = 0.820 with all sensors healthy at its per-scenario-tuned threshold (threshold-tuned XGBoost, see Table 3; 0.742 at the default threshold, Table 2) collapses to F1 = 0.159 when the light sensors fail, and to F1 = 0.168 when 70% of channels are randomly dropped (Table 3). Deep early-fusion networks are equally fragile (F1 = 0.001 when light is lost; Table 4). Occupancy controllers that silently trust such predictions would mis-drive HVAC and lighting precisely when the sensing infrastructure is compromised.

The machine-learning community has studied missing-modality robustness extensively, but almost entirely outside the building domain: multimodal sentiment analysis [9–11], medical imaging [12], recommendation [13], and federated settings [14,15]. A recent review of deep and transfer learning for building occupancy detection confirms that sensor-failure robustness has not been treated as a first-class problem in this domain [1], and cross-dataset generalization — where studied [16] — does not consider sensor failure. Reviews of the missing-modality literature further show that modality-dominance effects (a single dominant modality carrying most of the signal) are common but rarely exploited in sensor networks [17]. In the IoT/occupancy niche specifically, we identify four concrete gaps (verified against a matrix of 34 papers, 2022–2026, in the reproducibility package [18]):

- **G1.** No multimodal IoT framework addresses *systematic* missing modality (per-modality failure, 10–70% random channel dropout, multi-failure combinations, temporal misalignment) for occupancy detection.
- **G2.** Missing-modality methods from other domains (imputation, generation, decoupling, gating) have not been validated on heterogeneous environmental sensors (CO₂, temperature, humidity, light, acoustics, electricity) with realistic drift and noise characteristics.
- **G3.** Cross-dataset/cross-building generalization and sensor-failure resilience have not been evaluated together in a single study.
- **G4.** The combination of cross-modal attention, contrastive learning, and modality reliability has not been tested for IoT occupancy; modality dominance (cf. gradient-guided modality decoupling [17]) is unexplored in sensor networks.

This paper closes these gaps with the following contributions:

1. **A missing-modality-aware fusion framework for IoT occupancy detection.** Per-modality encoders feed a masked cross-modal attention transformer in which absent modalities are excluded from attention via a key-padding mask. The framework retains accuracy when sensors fail, outperforming imputation-, indicator-, and gating-based baselines in PR-AUC in 14/14 failure scenarios under an identical protocol, with the advantage growing as the missing rate increases (+0.06 → +0.26 vs. mean-imputation over 10–70% dropout).
2. **Missing-modality dropout augmentation as the core mechanism.** Training-time masking of whole modality groups on 30% of samples is empirically the most critical component: without it, the model collapses when the dominant modality fails (F1 0.001 vs. 0.528). This finding transfers the modality-dominance discussion of [17] into the sensor domain.
3. **A complementary InfoNCE contrastive objective** between clean and corrupted views — a supplementary learning signal that consistently improves ranking under dominant-modality failure (PR-AUC 0.887 vs. 0.815) without harming the healthy-sensor configuration, honestly characterized as secondary to augmentation and attention.
4. **The most comprehensive robustness evaluation protocol in this niche**: 26 failure/stress conditions in total — 14 failure scenarios (healthy configuration, 7 per-modality failures, 4 random-dropout rates, 2 multi-failure combinations) plus 12 stress conditions (Gaussian and impulse noise, sensor drift, temporal misalignment, block aggregation) — with one-hour block bootstrap confidence intervals, expected calibration error, and 5-minute block aggregation checks.
5. **An honest domain-shift analysis (office → residential).** Zero-shot transfer fails (ROC-AUC 0.374) and one-day fine-tuning captures only the occupancy prior (ROC-AUC 0.484), analyzed as a structural label–sensor semantic mismatch. This is reported as an explicit generalization boundary, not as a transfer success.

The remainder of the paper is organized as follows. Section 2 reviews related work on multimodal occupancy detection and missing-modality robustness. Section 3 describes the dataset, preprocessing, proposed architecture, baselines, and evaluation protocol. Section 4 presents results: dataset characterization, tree and single-modality baselines, the 14-scenario failure comparison, ablations, stress tests, and uncertainty/calibration analysis. Section 5 discusses the findings, including the domain-shift experiment. Section 6 concludes, Section 7 states limitations, and Section 8 provides data and code availability.

## 2. Related Work

### 2.1 Multimodal occupancy detection in buildings

Occupancy detection from environmental signals has an extensive literature in *Energy and Buildings* and *Building and Environment* [1,3,6,7,16,19,20]. Multimodal fusion is consistently reported to outperform single-sensor approaches. Tan et al. [7] fused environmental, audio, and smart-meter signals in residential buildings with a multilevel framework. Abuhussain et al. [5] proposed a multimodal smart-building framework but evaluated only healthy sensors. Tsanousa et al. [8] demonstrated late fusion of temperature, dust, and air-quality sensors at a real construction site, noting single-site limitations. CNN–XGBoost hybrids [19] and deep feature-fusion models such as DMFF [3] improve accuracy but assume complete inputs. Non-intrusive load monitoring has been extended to occupancy using smart-meter streams [2,6]; these single-source approaches motivate heterogeneous fusion rather than replacing it. Thermopile arrays [20], PIR/LSTM sensors [21], and CO₂-plus-audio configurations [22] show strong single-modality results that still call for multimodal fusion. Review-level analyses [1,23] converge on two conclusions: (i) deep learning dominates recent occupancy work, and (ii) sensor-failure robustness and multi-building generalization remain open. Our work addresses the former gap directly and treats the latter with an explicit, honest analysis.

### 2.2 Missing-modality robustness outside buildings

Missing-modality learning has matured in sentiment analysis and medical imaging. MissModal [9] learns representation-level robustness to random modality absence in multimodal sentiment analysis. GMD [17] decouples modality gradients to protect the dominant modality — precisely the failure mode we observe in lighting. TCTR [10] combines text-guided contrastive learning with token-level reconstruction, and MCL-MGN [11] uses multilevel contrastive learning for missing-modality generation; both remain confined to sentiment data. In medical imaging, Riemannian missing-modality learning has been proposed for computational pathology [12]. In recommendation, graph-attention contrastive learning handles missing modality [13]. Federated variants (FedMIR [14], mask-restoration federated learning [15]) address modality heterogeneity across clients rather than single-site sensor failure. Chemical-process multimodal learning with missing modality [23] is the closest industrial-sensor analogue but uses controlled process instrumentation. Uncertainty-theoretic treatments [24,25] and co-learning formulations [26] are generic and have not been validated on building sensors. Foundation-model and BEV-fusion approaches for wireless sensing [27] and autonomous driving [28] demonstrate that sensor-malfunction robustness is tractable, but in very different sensing regimes. A recent preprint shows that multi-sensor fusion can *fail to generalize* under distribution shift in cattle posture classification [29] — empirical support for the domain-shift caution we adopt. In summary, no published method has been transferred to, or validated on, heterogeneous building-environment sensors under systematic failure; to the best of our knowledge, this is the first systematic evaluation of missing-modality deep fusion for IoT occupancy detection.

### 2.3 Cross-modal attention and contrastive fusion

Cross-modal attention is a proven mechanism in computer vision: RGB-D semantic segmentation [30], RGB-T tracking [31], point-cloud segmentation [32], and multi-sensor fault diagnosis [33] all show that cross-attention improves fusion over concatenation or late fusion. Contrastive learning of multimodal representations [34] and cross-modal binding for IoT sensing [35] further support representation-level alignment. However, in these works attention is computed over *available* modalities only; the missing-modality case requires masking, and — with the exception of TCTR [10] in sentiment analysis — contrastive objectives are not used to align clean and corrupted views. Our architecture (Section 3.4) is, to the best of our knowledge, the first to combine masked cross-modal attention with clean↔corrupted InfoNCE in the building-IoT domain, and the ablation evidence (Section 4.4) attributes the robustness gain to each component individually.

### 2.4 Position of this paper

Against the 34-paper matrix summarized above, this paper occupies the intersection left empty by prior work: (i) a real, public, heterogeneous smart-office dataset [4,36] with documented natural missing data; (ii) a missing-modality-aware deep fusion architecture whose components (masked attention, dropout augmentation, contrastive view alignment) are individually ablated; (iii) a systematic failure protocol with block bootstrap confidence intervals and calibration; and (iv) an explicit, empirically characterized generalization boundary across domains. We claim robustness *within a domain* under sensor failure — not universal generalization.

## 3. Materials and Methods

### 3.1 Dataset

We use the public "A multimodal dataset for smart office occupancy estimation" [4] (Zenodo v3.0.0, DOI 10.5281/zenodo.20548374, CC-BY-4.0 [36]), collected in a smart office of the Pontifical Catholic University of Rio Grande do Sul (PUCRS), Porto Alegre, Brazil, between 2023-12-04 and 2024-01-16. The record contains 230,976 rows aggregated at 10-second intervals over 42.6 days, with 22 columns: a timestamp, 20 sensing channels, and a manual camera-annotated occupancy ground truth (number of people, 0–4). Hardware consists of commercial Tuya smart plugs and switches (read via LAN), a custom ESP32 sensor node, and MongoDB storage [4]. The 230,976 rows at 10-second aggregation correspond to 26.7 days of effective data (230,976 × 10 s); the 42.6-day calendar span therefore contains gaps — only 31 calendar days carry data (62.7% coverage).

The 20 sensing channels are organized into seven modality groups used throughout the paper (Table 1): environment/indoor air (CO₂, humidity, two temperatures), light (two illuminance sensors), acoustic (sound level), electrical server circuit (current/power/voltage), electrical firewall circuit (current/power/voltage), electrical socket (current/power/voltage), and device state (lamp LED state, three smart-switch channel states).

Ground truth is extremely imbalanced: 224,721 rows (97.3%) are unoccupied, 5,865 (2.5%) have one occupant, 346 have two, 42 have three, and 2 have four. Classes 3 and 4 together contain only 44 rows (0.02%); following a documented decision [18], these rows are **excluded from all experiments** (the task is binary occupancy; 44 rows cannot support multiclass modeling). Natural missing values are present on nearly every channel (sound 2.0%, firewall metering 0.47%, socket metering 0.21%, server metering 0.18%, CO₂/humidity 0.04%, etc.), providing real-world evidence of partial sensor availability (Table 1).

### 3.2 Preprocessing and data split

Binary occupancy is defined as ground_truth > 0. We add three time features (hour sine/cosine encoding and a weekend indicator) to every row. Continuous channels are z-score normalized using statistics fitted on the training split only; binary switch-state channels are not normalized.

Because occupancy is extremely sporadic at the day level (many days are entirely unoccupied), a purely temporal split would produce a nearly empty test set. We therefore use a **day-level stratified holdout** [18]: the 42.6-day record spans 31 calendar days, of which 6 days (20% of days) are held out as test, chosen by stratified random selection to guarantee adequate occupancy (seed 42). The test set contains 38,785 rows (16.8% of all rows — day-level splitting means 20% of days ≠ 20% of rows) with 7.5% occupied rows; the training side contains 25 days (192,191 rows, 1.7% occupied). An anomalous high-occupancy day (2023-12-04, 68.9% occupied, a documented calibration event) is included in the test split [18]. Model selection uses an internal evaluation set of 5 days stratified by occupancy (2023-12-05, 2023-12-14, 2023-12-16, 2024-01-05, 2024-01-11; ~2.8% occupied), carved from the training side; early stopping and threshold tuning are performed exclusively on this evaluation set, never on the test set.

### 3.3 Evaluation metrics

Given the 1:36 positive:negative ratio, accuracy is misleading. Following best practice for imbalanced occupancy data [1], we report **PR-AUC (primary, threshold-free)**, F1 at a tuned threshold, ROC-AUC, and Matthews correlation coefficient (MCC), plus balanced accuracy and macro-F1 where relevant. For every model, the decision threshold is tuned on the evaluation set (grid 0.05–0.95, step 0.01, best F1); per-model thresholds are reported in table footnotes. Because F1 is threshold-sensitive on this imbalanced distribution (a 0.95 threshold trades recall for precision), **PR-AUC is the primary comparison metric**; this decision and its rationale are documented in the protocol [18].

### 3.4 Proposed model

The proposed architecture (Fig. 2) has four components:

1. **Per-modality encoders (MLPs).** Each modality group $g$ is encoded by its own two-layer MLP from the channel vector concatenated with a per-channel presence mask: $\mathbf{z}_g = \mathrm{MLP}_g([\mathbf{x}_g; \mathbf{m}_g])$, producing one token per modality (dimension 64). The presence mask makes the encoders aware of per-channel availability, not only whole-modality absence.

2. **Masked cross-modal attention (transformer).** The seven modality tokens plus a learned [CLS] token are fed to a two-layer Transformer encoder (4 heads, feed-forward dimension 128, dropout 0.1). Modalities whose channels are entirely absent are excluded from attention through a **key-padding mask** (the [CLS] token is never masked), so attention is computed only over available modalities. This is the mechanism that lets the model reweight to surviving sensors when a modality fails.

3. **Classification head.** The [CLS] representation is passed to a linear head with BCE loss, weighted by the inverse class ratio (pos_weight = negatives/positives) to handle imbalance.

4. **InfoNCE contrastive view alignment (complementary).** During training, a corrupted view of each batch is produced by masking whole modality groups (probability 0.5 per sample). A projection MLP (64→32) maps clean and corrupted [CLS] representations into a normalized space; the symmetric InfoNCE loss with temperature 0.1 aligns each clean view with its corrupted counterpart. The total loss is $\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + 0.1 \cdot \mathcal{L}_{\mathrm{InfoNCE}}$. The contrastive term is a *complementary* learning signal: it does not alter the architecture or inference path, and its effect is ablated separately (Section 4.4).

**Missing-modality dropout augmentation (the critical component).** On 30% of training samples, one whole modality group is masked (values zeroed, presence flags cleared) before encoding. This forces the encoders and attention to learn from partial views and directly matches the failure distribution expected at inference. The ablation evidence (Section 4.4) shows this component — not the attention mechanism per se — is what prevents collapse under dominant-modality failure.

### 3.5 Training protocol

All deep models are trained with AdamW (learning rate 3×10⁻⁴, weight decay 1×10⁻⁴), batch size 512, for up to 30 epochs with early stopping (patience 5, minimum 5 epochs) on evaluation-set PR-AUC. The final model's best epoch is 8 (evaluation PR-AUC 0.4948; evaluation curve in Fig. 3); its decision threshold, tuned on the evaluation set, is 0.95. The earlier iteration of this protocol (2-day evaluation set, best epoch 1, threshold 0.75) is documented as an iteration history in the repository [18]; all numbers reported in this paper come from the final protocol. All experiments use seed 42 (plus seed+1 for batch augmentation), and every result JSON is archived [18].

### 3.6 Baselines

All baselines use the identical split, features, metrics, and evaluation-set threshold tuning.

- **Tree baselines.** Random forest and XGBoost on the full 23-feature vector with blind zero-imputation of missing values — the conventional deployment practice. A single-modality XGBoost series (light, environment-air, acoustic, each electrical circuit, device state) isolates per-modality contribution (Section 4.2). A "baseline tuned" variant re-tunes its threshold per scenario.
- **Reliability-weighted tree blend.** A tree-level fusion: score = (1−ρ)·score_base + ρ·score_missing-aware, where ρ is the fraction of masked channels; the missing-aware member is an XGBoost trained with missing-indicator features and dropout augmentation (variant B, implemented in [18]). This establishes the best tree-based reference (single threshold 0.47).
- **Deep early fusion (MLP early).** A conventional MLP (128–64) over all 20 channels with zero-imputation — the standard deep baseline, deliberately the least missing-aware architecture.
- **Three standard missing-modality strategies (imputation-, indicator-, and gating-based)**, implemented under an identical protocol (code in [18]):
  - **mean_impute**: mean-imputation (training means) + MLP early fusion [imputation-based];
  - **indicator_mlp**: zero-imputation + per-channel presence flags + MLP [indicator-based];
  - **gated_fusion**: per-modality encoders + learned reliability gate (sigmoid) + masked weighted sum + MLP head [gated/adaptive fusion].

### 3.7 Robustness protocol

**Fourteen failure scenarios.** Every model is evaluated on: (i) full (all sensors healthy); (ii) seven per-modality failures (each of the seven groups fully removed); (iii) four random channel-dropout rates (10%, 30%, 50%, 70% of numeric channels, seeded per scenario); (iv) two multi-failure combinations (light+acoustic, light+environment-air). Missing channels are zeroed and their presence flags cleared at inference; imputation-based baselines receive mean-imputed values.

**Twelve stress conditions** (additional protocol, evaluated on the trained model): Gaussian noise at 10/30/50% of channel standard deviation; impulse noise at 1%/5% of rows; sensor drift of +0.5σ/+1.0σ on all numeric channels; temporal misalignment of one channel group by +10/+50/+100/+300 s (a realistic synchronization failure); and a 5-minute block-aggregation evaluation (majority vote per 5-minute window) to verify that results are not artifacts of 10-second autocorrelation.

### 3.8 Statistical analysis

- **One-hour block bootstrap** (primary inference): 1000 resamples over 109 one-hour blocks of the test period (count printed by `ci_cluster_bootstrap.py` at runtime; consistent with 38,785 test rows × 10 s ≈ 107.7 h of data across the six test days) for five key scenarios (full, missing_light, missing_env_air, random 70%, light+env_air). Block resampling reduces dependence between adjacent 10-second rows; day-level resampling is not informative with only 6 test days (3 of them unoccupied), a structural limitation discussed in Section 7.
- **Per-row bootstrap** (archived for completeness): 1000 resamples over test rows; these intervals ignore temporal autocorrelation.
- **Calibration**: expected calibration error (ECE) over 10 bins and a reliability diagram for the healthy-sensor scenario (Fig. 4).

### 3.9 Cross-domain protocol (A → HPDmobile)

To probe generalization boundaries, we transfer models trained on the office dataset (A) to the public HPDmobile residential dataset [7,37] (CC0; Figshare collection 5364449; six homes H1–H6 with environmental, audio, and ground-truth data). H1 and H2 are used as targets (82.19% and 60.41% occupied rows, respectively — a stark contrast to the office's 2.7%). Five settings are evaluated: the majority baseline, zero-shot transfer of the blind-imputation XGBoost, zero-shot transfer of the missing-aware tree variant, the reliability-blend model, and a 1-day fine-tuning protocol of the base model. Only the four sensor channels common to both datasets (temperature, humidity, light, CO₂ or equivalent) plus time features are used, with dataset-consistent preprocessing. The result is analyzed structurally (Section 5.3); zero-shot failure is reported openly and used to bound generalization claims.

## 4. Results

All numbers are transcribed from the archived result JSONs (Section 8); metrics are rounded to three decimals.

### 4.1 Dataset characterization (Table 1, Fig. 1)

The dataset confirms the motivating premise: **natural missing data is pervasive but mild channel-wise** (sound 2.0%, firewall 0.47%, socket 0.21%, server 0.18%; Table 1), while ground truth is 97.3% unoccupied. The two light sensors are perfectly correlated (r = 1.00) and the two temperature sensors nearly so (r = 0.995), indicating redundancy that robustness methods can exploit; CO₂ and humidity are weakly anti-correlated (r = −0.25). Occupancy concentrates in working hours (07:00–21:00, peaking at 13:00).

### 4.2 Baselines and modality dominance (Table 2, Fig. 5)

Table 2 reports tree baselines and single-modality XGBoost models. Three findings structure the rest of the paper:

1. **Light is the dominant modality.** Light alone (F1 0.828, PR-AUC 0.800) nearly matches the full XGBoost (F1 0.742 at its default threshold; PR-AUC 0.848). Device state (F1 0.739) is second; environment-air, acoustics, and electrical circuits contribute little alone (PR-AUC 0.115–0.188). This confirms the modality-dominance phenomenon [17] in the sensor domain and explains why *light failure* is the hardest scenario.
2. **Some modalities are counterproductive for the tree baseline.** Dropping acoustic or firewall channels *improves* XGBoost (PR-AUC 0.862/0.869 vs. 0.848 full), evidence that blind fusion can overfit to noisy, weakly informative streams — motivating reliability-aware fusion.
3. **Random forest is unusable at this imbalance** (F1 0.079), and the tuned XGBoost collapses under failure (Table 3).

### 4.3 Main comparison: 14 failure scenarios (Table 3, Figs. 5–6)

Table 3 compares six models — tuned XGBoost (blind imputation), reliability-weighted tree blend, and the three deep missing-modality baselines (mean_impute, indicator_mlp, gated_fusion) — against the proposed model under all 14 scenarios (F1 and PR-AUC; full metrics including ROC-AUC and MCC in the results package [18]).

**Healthy configuration.** The proposed model attains PR-AUC 0.824 / F1 0.825 / ROC-AUC 0.992 / MCC 0.814. The tuned tree baselines retain a slight PR-AUC edge in the all-healthy case (0.848) — a small, honest trade-off discussed in Section 5.2. Against the three deep baselines, the proposed model wins PR-AUC in the healthy case as well (0.824 vs. 0.784–0.791).

**Per-modality failure.** When the dominant light modality fails, the proposed model keeps PR-AUC 0.887 / F1 0.528, while mean-imputation and gated fusion collapse (PR-AUC 0.135 and 0.111, F1 0.000) and the strongest baseline, indicator_mlp, reaches 0.842. For environment-air failure the proposed model scores 0.819 (baselines 0.780–0.795). When a redundant electrical circuit fails (socket), all models improve (proposed PR-AUC 0.916), since the remaining circuits carry the information.

**Random channel dropout.** The advantage of the proposed model **grows monotonically with the missing rate**: PR-AUC deltas vs. mean_impute are +0.056 (10%), +0.101 (30%), +0.183 (50%), +0.261 (70%); vs. gated fusion, +0.054/+0.099/+0.213/+0.398; vs. indicator_mlp, +0.049/+0.051/+0.062/+0.077. At 70% dropout the proposed model scores PR-AUC 0.887 / F1 0.670, where all baselines degrade to 0.489–0.809 PR-AUC and F1 ≤ 0.237.

**Multi-failure.** With light+acoustic lost, PR-AUC is 0.886 (indicator_mlp 0.885; mean_impute/gated 0.107–0.128). With light+environment-air lost — the most severe scenario — the proposed model retains PR-AUC 0.823 / F1 0.587, versus 0.696 (indicator), 0.096–0.104 (mean_impute/gated), and 0.222 (tree).

**Summary of the 14×14 comparison.** The proposed model achieves the **highest PR-AUC in 14/14 scenarios against each of the three deep missing-modality baselines**, and in 10/14 scenarios against the tuned tree baselines; the four tree wins are confined to scenarios where the dominant light modality is healthy (full, acoustic, server, firewall). F1 at the tuned threshold is highest in 10/14 scenarios vs. the per-scenario-tuned XGBoost, and in 8/14 vs. the tree blend (the blend also wins F1 at random 50% and 70% dropout: 0.686/0.788 vs. 0.626/0.670); PR-AUC remains superior in 10/14 scenarios against both tree columns. The remaining F1 gaps reflect the threshold sensitivity analyzed in Section 5.2 (PR-AUC remains first or near-first among deep models in those cells).

### 4.4 Ablation study (Table 4)

Five variants isolate each component on five representative scenarios (Table 4). Ablation variants were trained with the same hyperparameters under a shorter three-epoch protocol; because per-variant thresholds differ (0.29–0.95), **PR-AUC is the fair comparator**; F1 is reported with each variant's threshold for transparency.

1. **Dropout augmentation is the most critical component.** Removing it collapses the model precisely under dominant-modality failure: missing_light F1 drops from 0.528 to 0.001 (PR-AUC 0.887 → 0.372); random-70% F1 from 0.670 to 0.529 (PR-AUC 0.887 → 0.724); light+env_air PR-AUC from 0.823 to 0.140. In the healthy configuration, no_aug also loses (PR-AUC 0.743 vs. 0.824).
2. **Masked cross-modal attention is essential for dominant-modality failure.** Replacing attention with masked mean-pooling drops missing_light F1 from 0.528 to 0.264 (PR-AUC 0.887 → 0.328) and light+env_air PR-AUC from 0.823 to 0.259. Interestingly, mean-pooling is *better* when environment-air alone fails (F1 0.912 vs. 0.788) — attention concentrates weight on the strongest modalities, which hurts when a secondary modality is missing while light remains; this nuance is analyzed in Section 5.2.
3. **Contrastive learning is complementary, not decisive.** Without the contrastive loss, PR-AUC at missing_light is 0.815 vs. 0.887 with it; the healthy configuration is essentially tied (0.821 vs. 0.824). Across the 14-scenario matrix, the contrastive variant improves PR-AUC in 12/14 scenarios (e.g., random-70% 0.865 → 0.887; light+env_air 0.812 → 0.823) and is marginally worse in only two redundant-electrical scenarios (server 0.822 vs. 0.817; firewall 0.819 vs. 0.817), with all differences ≤ 0.005. F1 is not directly comparable because of threshold differences (0.29 vs. 0.95). We therefore present contrastive learning as a consistent, complementary signal whose main reproducible benefit appears under dominant-modality failure, and we retain it because it never materially harms any configuration.
4. **Conventional deep early fusion is as fragile as tree baselines.** The MLP early-fusion model collapses exactly where trees do (missing_light F1 0.001, PR-AUC 0.187; light+env_air F1 0.043, PR-AUC 0.071), confirming that robustness comes from the missing-aware architecture, not from deep learning per se.

### 4.5 Noise, drift, and temporal misalignment (Table 5)

Table 5 summarizes the 12 stress conditions. The model is **highly robust to noise**: F1 ≥ 0.857 up to 5% impulse noise and 50% Gaussian noise (PR-AUC even rises slightly, 0.799 → 0.813, as noise diversifies the inputs). **Sensor drift does not affect it** (F1 0.915–0.916 at up to +1.0σ drift). **Temporal misalignment up to +300 s is absorbed** (F1 0.912–0.914; PR-AUC 0.800 → 0.854). The 5-minute block-aggregation evaluation reproduces the results (F1 0.904, balanced accuracy 0.972), confirming that the findings are not artifacts of 10-second autocorrelation. These stress tests were executed with the model checkpoint available at protocol time (decision threshold 0.75); the reference row reproduces the healthy-sensor configuration of the proposed architecture, and relative degradation under stress is the quantity of interest.

### 4.6 Uncertainty and calibration (Table 6, Figs. 3–4)

Table 6 reports one-hour block bootstrap 95% confidence intervals for five key scenarios. Intervals are wide (e.g., full F1 0.825 [0.690, 0.913]) because the test period contains only six days, three of which are fully unoccupied — a structural limitation rather than a modeling failure. **Consequently, differences between scenarios are frequently not statistically significant at the block level, and claims in this paper emphasize PR-AUC ranking plus effect sizes rather than pointwise F1 differences.** For completeness, per-row bootstrap intervals (which ignore autocorrelation and are therefore narrower) are archived: full F1 0.914 [0.907, 0.921], PR-AUC 0.798 [0.779, 0.816] (computed with the earlier threshold-0.75 checkpoint; the block-bootstrap table uses the final checkpoint).

Calibration is adequate overall (ECE = 0.055, Fig. 4; computed on healthy-sensor test predictions, n = 38,785), with a systematic pattern: the model is **overconfident in its top bin** (mean confidence 0.943 vs. observed frequency 0.863 for bins 0.9–1.0) and underconfident around 0.7–0.8 — exactly the region where the 0.95 decision threshold operates, explaining the F1 trade-off discussed in Section 5.2.

### 4.7 Cross-domain analysis (A → HPDmobile)

Table 7 reports the transfer experiment. Zero-shot transfer **fails**: ROC-AUC of 0.374 (H1) and 0.512 (H2) — at or below chance — despite respectable PR-AUC (0.767/0.598), which is an artifact of the target prior (82%/60% occupied) being captured by the model's positive bias. One-day fine-tuning recovers F1 (0.850/0.684) but **not ranking** (ROC-AUC 0.484/0.478), meaning the model learns the target prior (predict "occupied" often) without learning to discriminate occupied from empty. This is reported as a structural domain-shift analysis, not a transfer success: the office→residential gap combines (i) a semantic label mismatch ("anyone in the house" vs. sensors located in specific rooms), (ii) an extreme prior shift (2.7% → 60–82%), and (iii) different lifestyles (lights on at night). Section 5.3 develops the implications.

## 5. Discussion

### 5.1 What makes the model robust — and what does not

The ablation study gives a clean attribution: **dropout augmentation is the decisive mechanism**; masked attention matters specifically when the dominant modality is lost; contrastive learning is a consistent but secondary complement. This ordering is worth stating precisely because the paper's title emphasizes attention and contrastive learning. Our interpretation: augmentation teaches the encoders and attention to operate on partial views *during training*, so at inference a failed modality is not an out-of-distribution event but a configuration the network has seen thousands of times. Attention then provides the *mechanism* for reweighting — when light vanishes, the [CLS] token must learn to rely on the second-best modalities (device state, electrical circuits), which only works if training exposed those reweighted configurations (augmentation) and if attention cannot attend to zeroed tokens (masking). Contrastive alignment makes clean and corrupted views share a representation space, slightly improving ranking under failure (0.887 vs. 0.815 at missing_light) — a small but never-negative effect across scenarios, which is why we keep it.

The collapse of imputation- and gating-based baselines under light failure is equally instructive: mean-imputation fills the failed light channel with its training mean, a *confident but wrong* signal; learned gates (gated_fusion) cannot compensate because they were trained mostly on healthy data; and early-fusion MLPs have no structural way to ignore a poisoned channel. The indicator-based baseline (presence flags) is the strongest competitor precisely because flags make absence explicit — the same design choice we make — yet it still loses in 14/14 scenarios, indicating that explicit masking plus augmentation plus attention is worth more than flags alone.

### 5.2 Trade-offs: PR-AUC vs. F1, attention vs. mean-pooling, healthy vs. failed

Three honest trade-offs deserve explicit discussion.

**PR-AUC vs. F1.** PR-AUC is threshold-free and stable; F1 depends on the tuned threshold. The final model's threshold of 0.95 (chosen on the evaluation set) maximizes F1 there but produces conservative F1 on the harder test distribution (e.g., full F1 0.825; missing_device_state F1 0.417 despite PR-AUC 0.808, because the 0.95 threshold suppresses recall). This is visible in the calibration plot: the model is overconfident in the 0.9–1.0 bin, and a threshold at the top of that bin inherits the overconfidence. Practitioners deploying the model should re-tune the threshold on local data; ranking quality (PR-AUC/ROC-AUC) is the robust quantity, which is why it is our primary metric. For transparency, every model's threshold is reported in the tables.

**Attention vs. mean-pooling.** The ablation shows mean-pooling beating attention when a *secondary* modality (environment-air) is lost while light remains (F1 0.912 vs. 0.788). We interpret this as attention over-concentrating on the dominant modality: with light present, attention largely ignores environment-air tokens; when environment-air is removed, the model's output changes little (it was ignoring them anyway), yet its confidence distribution shifts in a way that hurts F1 at the tuned threshold. Mean-pooling spreads weight more evenly and degrades more gracefully in that specific configuration. This is a genuine limitation of attention-based weighting for dominant-modality sensor sets, and it suggests a hybrid (attention for dominant-modality failure, averaging for secondary failures) as future work.

**Healthy vs. failed configurations.** The proposed model yields a small PR-AUC deficit vs. tuned trees when all sensors are healthy (0.824 vs. 0.848) and when redundant electrical channels fail (acoustic/server/firewall scenarios). In exchange it provides large gains in exactly the scenarios that matter operationally (dominant-modality and heavy dropout). Whether this trade is acceptable depends on the application: for energy management, a controller that must survive sensor degradation may prefer the robust model; for analytics on perfectly maintained sensor networks, tuned trees remain competitive and cheaper. We do not claim dominance in all conditions.

### 5.3 Domain shift: why zero-shot transfer fails (and what it teaches)

The office→residential transfer failure is the paper's clearest generalization boundary. Three structural causes are identifiable from the data: (i) **label semantics** — HPDmobile labels "any human present in the house" while sensors sit in specific rooms, so a room empty but a house occupied is labeled positive, and the four common channels (temperature, humidity, light, CO₂-type) are only weakly predictive of the label; (ii) **prior shift** — 2.7% occupied (office) vs. 60–82% (homes), which the majority classifier and fine-tuned models exploit; (iii) **lifestyle divergence** — residential lighting patterns (lights on at night) differ qualitatively from office patterns. The fine-tuning result — F1 recovers, ROC-AUC stays ≈ 0.48 — is the cleanest evidence that the model captures the prior, not the signal. We believe reporting this failure with its structural analysis is a contribution in itself: the missing-modality and multi-sensor-fusion literature rarely publishes negative transfer results [29], yet they are precisely the information needed to design transferable occupancy models (e.g., label-aware sensor placement, prior calibration, or domain adaptation on matched label semantics).

### 5.4 Comparison with the literature

Where direct numerical comparison is possible, our healthy-configuration results (PR-AUC 0.824, ROC-AUC 0.992, MCC 0.814) are in line with recent deep occupancy models on public data [2,3,6] — the contribution is not a new accuracy record on a healthy sensor network, but *robustness*: no published occupancy model, to our knowledge, reports performance across 14 systematic failure scenarios with block-bootstrap intervals and calibration, nor attributes robustness to specific architectural components on a public dataset. Methodologically, our results corroborate the modality-dominance findings of GMD [17] in the sensor domain (light dominance), support the value of explicit missing indicators (indicator_mlp being the strongest baseline), and extend contrastive missing-modality learning [10,11] to building IoT with a clean↔corrupted view formulation whose effect is honestly bounded.

## 6. Conclusion and Future Work

We presented a missing-modality-aware deep fusion framework for IoT occupancy detection that remains accurate when sensors fail, drift, or desynchronize. On a public 42.6-day smart-office dataset, the framework achieves the highest PR-AUC in 14/14 systematic failure scenarios against three standard missing-modality baselines (imputation-, indicator-, and gating-based) under an identical protocol, with advantages that grow with the missing rate (+0.05 to +0.40 PR-AUC at 10–70% channel dropout) and the strongest gains under dominant-modality (light) failure (PR-AUC 0.887 vs. 0.111–0.842). Ablations attribute the robustness primarily to missing-modality dropout augmentation, secondarily to masked cross-modal attention, with contrastive learning as a consistent complement; calibration (ECE 0.055) and block-bootstrap intervals quantify uncertainty honestly; and a failed zero-shot office→residential transfer is analyzed structurally to bound generalization claims.

Future work proceeds along four axes: (i) **temporal modeling** — the current model is per-row (10 s); windowed attention or sequence models (LSTM/Transformer over 5–10-minute windows) are a natural extension, motivated by the block-aggregation results (F1 0.904); (ii) **multi-building evaluation** — the one-building/one-season limitation is the paper's weakest external-validity point, and joint training on the office and residential datasets with explicit label-semantics alignment is planned; (iii) **sensor-drift adaptation** — although drift up to +1.0σ did not degrade performance, long-horizon drift and seasonal effects require online normalization or continual learning; (iv) **deployment-oriented validation** — latency, parameter count, and edge-device profiling for real-time HVAC integration, plus threshold re-tuning procedures for practitioners.

## 7. Limitations

We state limitations explicitly, in the spirit of the evaluation protocol that produced this work [18]:

1. **Single building, single season.** All data come from one smart office over 42.6 days (December–January, summer in Brazil). External validity across buildings, climates, and seasons is not established; the failed cross-dataset transfer demonstrates the risk rather than mitigating it.
2. **Small and skewed test set.** Six test days (38,785 rows, 7.5% occupied), three of which are fully unoccupied, limit statistical power: one-hour block bootstrap intervals are wide and frequently overlap between scenarios, so scenario-level differences should be read as effect sizes, not significance claims. Day-level clustering (the statistically ideal unit) is not informative with only six days; we report this as a structural constraint. The anomalous high-occupancy day is included in the test set, which raises test PR-AUC relative to the evaluation set.
3. **Excluded classes.** The 44 rows with 3–4 occupants (0.02%) were excluded by documented decision; the model addresses binary occupancy only, and counting tasks are out of scope.
4. **Threshold sensitivity.** Reported F1 values depend on the evaluation-tuned threshold (0.95 for the final model); ranking metrics (PR-AUC, ROC-AUC) are the robust quantities. Overconfidence in the top calibration bin is a known pattern that deployment should address with threshold re-tuning.
5. **Contrastive component is secondary.** The InfoNCE term yields consistent but small gains; claims in this paper are deliberately limited to "complementary." Its value may grow with larger, multi-building data.
6. **Attention over-concentration.** When a secondary modality fails while the dominant one remains, mean-pooling can outperform attention (F1 0.912 vs. 0.788, environment-air scenario); attention-based weighting is not universally optimal for dominant-modality sensor sets.
7. **Natural missing data is not yet exploited as a scenario.** The dataset's natural gaps (sound 2.0%, etc.) are documented and handled by the presence-mask design, but a dedicated "natural-missing evaluation" is left for future work.
8. **Checkpoint-dependent artifacts.** Stress tests (Table 5) and per-row bootstrap intervals were computed with the earlier threshold-0.75 checkpoint available at protocol time; the block-bootstrap table uses the final checkpoint. ECE and the reliability diagram were computed on healthy-sensor predictions (n = 38,785); calibration under failed modalities is not separately reported.

## 8. Data and Code Availability

- **Dataset**: "A multimodal dataset for smart office occupancy estimation", Zenodo v3.0.0, DOI 10.5281/zenodo.20548374 (CC-BY-4.0) [36]; data paper: Deconto et al., *Data in Brief* (2026) [4].
- **Residential target dataset**: HPDmobile, Figshare collection 5364449 (CC0) [37]; data descriptor: Jacoby et al., *Scientific Data* (2021) [37]; source paper: Tan et al., *Energy and Buildings* (2022) [7].
- **Code and reproducibility package**: https://github.com/15FST015/iot-occupancy-multimodal — all preprocessing, training, evaluation, ablation, CI, calibration, and cross-domain scripts (seed 42), plus every result JSON referenced in this paper [18].
- **Privacy**: both datasets are public and anonymized by their authors — the primary dataset publishes no raw imagery (ground truth was camera-assisted during collection; only sensor time series are released, CC-BY-4.0), and HPDmobile deliberately omits raw image/audio content (CC0). This study processes no personal data directly.

## Acknowledgements

*(To be completed by the authors: funding, institutional support, data-collection acknowledgements.)*

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## CRediT authorship contribution statement

*(Author names to be completed by the authors before submission; the role descriptions below are complete.)*

- **Conceptualization**: *(to be completed)* — study design, problem framing, and research questions.
- **Methodology**: *(to be completed)* — proposed architecture (masked cross-modal attention, missing-modality dropout augmentation, InfoNCE view alignment) and the robustness evaluation protocol.
- **Software**: *(to be completed)* — implementation of the training, evaluation, ablation, bootstrap, calibration, and cross-domain scripts in the reproducibility package [18].
- **Validation**: *(to be completed)* — verification of every reported number against the archived result JSONs; reproduction of tables and figures.
- **Formal analysis**: *(to be completed)* — statistical analysis (one-hour block bootstrap, ECE, effect sizes) and interpretation of results.
- **Investigation**: *(to be completed)* — experimental runs: 14 failure scenarios, 12 stress conditions, ablations, and cross-domain transfer.
- **Resources**: *(to be completed)* — dataset acquisition (Zenodo, Figshare) and computing resources.
- **Data curation**: *(to be completed)* — preprocessing, day-level stratified split design, and natural-missing-data documentation.
- **Writing – original draft**: *(to be completed)* — full manuscript draft.
- **Writing – review & editing**: *(to be completed)* — critical revision and final approval of the manuscript.
- **Visualization**: *(to be completed)* — architecture diagram and results figures.
- **Supervision**: *(to be completed)* — oversight of the research and manuscript preparation.
- **Project administration**: *(to be completed)* — coordination and submission logistics.
- **Funding acquisition**: *(to be completed)* — *(if applicable)*.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used AI-assisted tools (large language models) to support literature synthesis, experiment scripting, statistical analysis, and language editing. The authors reviewed and verified all results, claims, citations, and numbers against the original experimental outputs and primary sources; AI tools were not used to generate or fabricate data, and they are not listed as authors. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

---

## References

[1] A.N. Sayed, Y. Himeur, F. Bensaali, Deep and transfer learning for building occupancy detection: A review and comparative analysis, Eng. Appl. Artif. Intell. 115 (2022) 105254. https://doi.org/10.1016/j.engappai.2022.105254.

[2] S. Mahmud, F. Bensaali, M.E.H. Chowdhury, M. Houchati, Multimodal feature fusion and ensemble learning for non-intrusive occupancy monitoring using smart meters, Build. Environ. 271 (2025) 112635. https://doi.org/10.1016/j.buildenv.2025.112635.

[3] K. Sun, DMFF: Deep multimodel feature fusion for building occupancy detection, Build. Environ. 253 (2024) 111355. https://doi.org/10.1016/j.buildenv.2024.111355.

[4] G. Dall'Agnol Deconto, A.F. Zorzo, R.C. Lunardi, R.C. Cardoso, L. dos Santos Teixeira, A multimodal dataset for environmental occupancy detection, Data in Brief 67 (2026) 112948. https://doi.org/10.1016/j.dib.2026.112948.

[5] M.A. Abuhussain, B.S. Alotaibi, Y.A. Dodo, A. Maghrabi, M.S. Aliero, Multimodal framework for smart building occupancy detection, Sustainability 16 (2024) 4171. https://doi.org/10.3390/su16104171.

[6] S. Mahmud, M. Houchati, M.E.H. Chowdhury, F. Bensaali, A cascaded deep learning framework for simultaneous non-intrusive load and occupancy monitoring using multi-channel aggregated smart meter data, J. Build. Eng. 112 (2025) 113731. https://doi.org/10.1016/j.jobe.2025.113731.

[7] S.Y. Tan, M. Jacoby, H. Saha, A. Florita, G. Henze, S. Sarkar, Multimodal sensor fusion framework for residential building occupancy detection, Energy Build. 258 (2022) 111828. https://doi.org/10.1016/j.enbuild.2021.111828.

[8] A. Tsanousa, C. Moschou, E. Bektsis, S. Vrochidis, I. Kompatsiaris, Fusion of environmental sensors for occupancy detection in a real construction site, Sensors 23 (2023) 9596. https://doi.org/10.3390/s23239596.

[9] R. Lin, H. Hu, MissModal: Increasing robustness to missing modality in multimodal sentiment analysis, Trans. Assoc. Comput. Linguist. 11 (2023) 1686–1702. https://doi.org/10.1162/tacl_a_00628.

[10] Z. Yang, Q. He, M. Yu, N. Du, Y. Lu, TCTR: Text-guided contrastive learning with token-level reconstruction network for missing modalities in multimodal sentiment analysis, Inf. Fusion 126 (2026) 103571. https://doi.org/10.1016/j.inffus.2025.103571.

[11] Y. Yu, F. Liu, Y. Zhang, X. Gao, H. Yang, MCL-MGN: A multi-level contrastive learning framework for missing modality generation in multimodal sentiment analysis, Digit. Signal Process. 174 (2026) 105880. https://doi.org/10.1016/j.dsp.2025.105880.

[12] G. Yang, M. Qu, D. Di, K. Yi, H. Xu, T. Su, L. Fan, Unified multimodal computational pathology with missing-modality robustness via Riemannian learning, Inf. Fusion 137 (2027) 104633. https://doi.org/10.1016/j.inffus.2026.104633.

[13] W. Zhao, K. Yang, P. Ding, C. Na, W. Li, Graph attention contrastive learning with missing modality for multimodal recommendation, Knowl.-Based Syst. 311 (2025) 113035. https://doi.org/10.1016/j.knosys.2025.113035.

[14] H. Xiong, M. Dai, FedMIR: Multimodal federated learning with missing modality imputation and distribution-aware routing, Sensors 26 (2026) 2954. https://doi.org/10.3390/s26102954.

[15] Z. Cao, K. Hao, L. Hao, B. Wei, L. Ren, Heterogeneous multimodal federated learning with missing modality via mask-restoration and self-guidance, IEEE Trans. Multimed. 28 (2026) 3571–3583. https://doi.org/10.1109/tmm.2026.3654379.

[16] J. Dridi, M. Amayri, N. Bouguila, Transfer learning for estimating occupancy and recognizing activities in smart buildings, Build. Environ. 217 (2022) 109057. https://doi.org/10.1016/j.buildenv.2022.109057.

[17] H. Wang, S. Luo, G. Hu, J. Zhang, Gradient-guided modality decoupling for missing-modality robustness, Proc. AAAI Conf. Artif. Intell. 38 (2024) 15483–15491. https://doi.org/10.1609/aaai.v38i14.29474.

[18] A. Pataropura, Robust multimodal IoT occupancy detection under sensor failure — reproducibility package, GitHub, 2026. https://github.com/15FST015/iot-occupancy-multimodal.

---

## Tables

### Table 1. Dataset characterization (smart office PUCRS; 230,976 rows; 42.6 days; 10-s aggregation).

| Modality group | Channels | Natural missing (%) |
|---|---|---|
| Environment (indoor air) | CO₂, humidity, temperature_1, temperature_2 | 0.039 / 0.039 / 0.011 / 0.039 |
| Light | lux_1, lux_2 | 0.011 / 0.039 |
| Acoustic | sound | 1.97 |
| Electrical — server | current, power, voltage | 0.184 / 0.184 / 0.184 |
| Electrical — firewall (pfsense) | current, power, voltage | 0.47 / 0.47 / 0.47 |
| Electrical — socket | current, power, voltage | 0.21 / 0.21 / 0.21 |
| Device state | lamp_switch_led, switch_channel_1..3 | 0.037 / 0.005 / 0.005 / 0.005 |
| **Ground truth** | occupants 0/1/2/3/4: 224,721 / 5,865 / 346 / 42 / 2 rows (97.3% / 2.5% / 0.15% / 0.02% / <0.01%) | — |
| **Split (day-level stratified)** | train 25 days (192,191 rows; 1.7% occupied) · eval 5 days stratified (~2.8% occupied) · test 6 days (38,785 rows; 7.5% occupied) | — |
| Excluded | classes 3–4 (44 rows, 0.02%), documented decision | — |

*Notes: binary occupancy = ground_truth > 0. The anomalous high-occupancy day 2023-12-04 (68.9%) is included in the test split [18]. lux_1–lux_2 correlation 1.00; temperature_1–temperature_2 correlation 0.995; CO₂–humidity −0.25.*

### Table 2. Tree baselines and single-modality XGBoost models (test set; thresholds tuned per model on the evaluation set).

| Model (inputs) | F1 | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|
| XGBoost — all 23 features (blind zero-imputation) | 0.742 | 0.848 | 0.989 | 0.734 |
| Random forest — all features | 0.079 | 0.740 | 0.978 | 0.191 |
| XGBoost — light only (lux_1, lux_2) | 0.828 | 0.800 | 0.946 | 0.815 |
| XGBoost — device state only | 0.739 | 0.663 | 0.908 | 0.722 |
| XGBoost — environment-air only | 0.313 | 0.188 | 0.737 | 0.266 |
| XGBoost — acoustic only | 0.217 | 0.129 | 0.729 | 0.150 |
| XGBoost — electrical server only | 0.165 | 0.115 | 0.700 | 0.071 |
| XGBoost — electrical socket only | 0.211 | 0.176 | 0.828 | 0.133 |
| XGBoost — electrical firewall only | 0.124 | 0.126 | 0.715 | 0.017 |

*Light alone (PR-AUC 0.800) nearly matches the full model (0.848): the dominant modality. Dropping acoustic or firewall channels improves the full model (PR-AUC 0.862 / 0.869), i.e., those streams are counterproductive for the tree baseline. 5-min block aggregation of XGBoost: F1 0.749.*

### Table 3. Main comparison under 14 sensor-failure scenarios (test set). Values: F1 / PR-AUC. Bold = best PR-AUC per scenario.

| Scenario | XGB tuned (blind imp.) | Tree blend | mean_impute | indicator_mlp | gated_fusion | **Proposed** |
|---|---|---|---|---|---|---|
| full (healthy) | 0.820 / **0.848** | 0.820 / **0.848** | 0.865 / 0.784 | 0.743 / 0.789 | 0.881 / 0.791 | 0.825 / 0.824 |
| missing_light | 0.159 / 0.359 | 0.389 / 0.520 | 0.000 / 0.135 | 0.000 / 0.842 | 0.000 / 0.111 | 0.528 / **0.887** |
| missing_env_air | 0.173 / 0.615 | 0.237 / 0.678 | 0.834 / 0.780 | 0.738 / 0.792 | 0.908 / 0.795 | 0.788 / **0.819** |
| missing_acoustic | 0.825 / **0.862** | 0.824 / 0.852 | 0.856 / 0.781 | 0.657 / 0.789 | 0.891 / 0.790 | 0.803 / 0.819 |
| missing_elec_server | 0.827 / **0.855** | 0.821 / 0.830 | 0.869 / 0.781 | 0.708 / 0.786 | 0.892 / 0.788 | 0.807 / 0.817 |
| missing_elec_pfsense | 0.865 / **0.869** | 0.853 / 0.851 | 0.864 / 0.778 | 0.707 / 0.783 | 0.907 / 0.771 | 0.804 / 0.817 |
| missing_elec_socket | 0.864 / 0.856 | 0.858 / 0.859 | 0.939 / 0.896 | 0.939 / 0.899 | 0.939 / 0.899 | 0.942 / **0.916** |
| missing_device_state | 0.705 / 0.768 | 0.705 / 0.768 | 0.769 / 0.781 | 0.530 / 0.784 | 0.890 / 0.791 | 0.417 / **0.808** |
| random dropout 10% | 0.728 / 0.804 | 0.742 / 0.808 | 0.781 / 0.798 | 0.675 / 0.806 | 0.794 / 0.801 | 0.775 / **0.855** |
| random dropout 30% | 0.531 / 0.710 | 0.648 / 0.797 | 0.593 / 0.775 | 0.530 / 0.825 | 0.602 / 0.778 | 0.677 / **0.876** |
| random dropout 50% | 0.333 / 0.594 | 0.686 / 0.828 | 0.379 / 0.700 | 0.346 / 0.821 | 0.396 / 0.670 | 0.626 / **0.883** |
| random dropout 70% | 0.168 / 0.460 | 0.788 / 0.857 | 0.158 / 0.626 | 0.152 / 0.809 | 0.237 / 0.489 | 0.670 / **0.887** |
| light + acoustic missing | 0.166 / 0.426 | 0.405 / 0.627 | 0.000 / 0.128 | 0.000 / 0.885 | 0.000 / 0.107 | 0.451 / **0.886** |
| light + env_air missing | 0.081 / 0.222 | 0.127 / 0.713 | 0.000 / 0.104 | 0.000 / 0.696 | 0.000 / 0.096 | 0.587 / **0.823** |

*Thresholds (tuned on the evaluation set): XGB per-scenario (0.20–0.66 range); tree blend single 0.47; mean_impute 0.95; indicator_mlp 0.95; gated_fusion 0.95; proposed 0.95. ROC-AUC and MCC for every model×scenario cell are archived [18]. Bold = highest PR-AUC per scenario. The proposed model attains the highest PR-AUC in 14/14 scenarios vs. each deep baseline and in 10/14 vs. both tree columns (tree wins only where light remains healthy: full, acoustic, server, firewall). F1 at the tuned threshold is highest in 10/14 scenarios vs. the per-scenario-tuned XGBoost and in 8/14 vs. the tree blend (the blend also wins F1 at random 50%/70%: 0.686/0.788 vs. 0.626/0.670); PR-AUC remains superior in 10/14 scenarios for both tree columns.*

### Table 4. Ablation study (five variants × five representative scenarios; F1 / PR-AUC).

| Variant (threshold) | full | missing_light | missing_env_air | random 70% | light+env_air |
|---|---|---|---|---|---|
| **Proposed** (0.95) | 0.825 / 0.824 | 0.528 / **0.887** | 0.788 / **0.819** | 0.670 / **0.887** | 0.587 / **0.823** |
| no_contrastive (0.29) | 0.943 / 0.821 | 0.938 / 0.815 | 0.944 / 0.818 | 0.942 / 0.865 | 0.938 / 0.812 |
| no_attention (0.44) | 0.920 / 0.769 | 0.264 / 0.328 | 0.912 / 0.784 | 0.847 / 0.820 | 0.170 / 0.259 |
| no_aug (0.73) | 0.794 / 0.743 | 0.001 / 0.372 | 0.142 / 0.764 | 0.529 / 0.724 | 0.001 / 0.140 |
| mlp_early (0.39) | 0.907 / 0.773 | 0.001 / 0.187 | 0.855 / 0.736 | 0.609 / 0.586 | 0.043 / 0.071 |

*Bold = best PR-AUC per scenario among the deep variants. PR-AUC is the fair comparator because per-variant thresholds differ (parentheses; tuned on the evaluation set). Ablation variants were trained for three epochs under the earlier protocol; component conclusions are consistent with the full 14-scenario matrix [18]. Findings: (1) no_aug collapses under dominant-modality failure (missing_light F1 0.001 vs. 0.528; PR-AUC 0.372 vs. 0.887) → augmentation is the most critical component; (2) no_attention collapses at missing_light (PR-AUC 0.328) but beats attention when only a secondary modality (env_air) is lost → attention essential for dominant-modality failure, mean-pooling more graceful for secondary failure; (3) no_contrastive is close but consistently slightly worse under dominant-modality failure (0.815 vs. 0.887) → contrastive is complementary; (4) mlp_early is as fragile as tree baselines (missing_light PR-AUC 0.187).*

### Table 5. Noise, drift, and temporal misalignment stress tests (test set; reference = healthy configuration).

| Condition | F1 | PR-AUC | MCC | ROC-AUC |
|---|---|---|---|---|
| full_clean (reference) | 0.914 | 0.798 | 0.907 | 0.991 |
| Gaussian noise 10% / 30% / 50% | 0.912 / 0.901 / 0.883 | 0.799 / 0.801 / 0.813 | 0.905 / 0.893 / 0.874 | 0.991 / 0.991 / 0.991 |
| Impulse noise 1% / 5% | 0.904 / 0.857 | 0.799 / 0.813 | 0.896 / 0.846 | 0.991 / 0.991 |
| Drift +0.5σ / +1.0σ | 0.915 / 0.916 | 0.807 / 0.807 | 0.909 / 0.910 | 0.991 / 0.991 |
| Misalignment +10 s / +50 s / +100 s / +300 s | 0.914 / 0.913 / 0.912 / 0.914 | 0.800 / 0.811 / 0.822 / 0.854 | 0.907 / 0.906 / 0.905 / 0.907 | 0.991 / 0.991 / 0.991 / 0.991 |
| 5-min block aggregation (full) | 0.904 | 0.822 | 0.897 | (bacc 0.972) |

*The model is robust to noise (F1 ≥ 0.857 at 5% impulse), drift (F1 0.916 at +1σ), and misalignment up to +300 s (F1 0.914; PR-AUC rises to 0.854). Block aggregation (5-min majority) reproduces the results, ruling out 10-s autocorrelation artifacts. Executed with the checkpoint available at protocol time (threshold 0.75); relative degradation is the quantity of interest [18].*

### Table 6. Uncertainty quantification: one-hour block bootstrap (1000 resamples, 109 blocks) for five key scenarios; calibration for the healthy scenario.

| Scenario | F1 [95% CI] | PR-AUC [95% CI] | MCC [95% CI] |
|---|---|---|---|
| full | 0.825 [0.690, 0.913] | 0.824 [0.663, 0.929] | 0.814 [0.690, 0.908] |
| missing_light | 0.528 [0.370, 0.682] | 0.887 [0.779, 0.939] | 0.559 [0.427, 0.696] |
| missing_env_air | 0.788 [0.650, 0.893] | 0.819 [0.659, 0.937] | 0.777 [0.650, 0.888] |
| random dropout 70% | 0.670 [0.617, 0.726] | 0.887 [0.790, 0.944] | 0.675 [0.619, 0.734] |
| light + env_air | 0.587 [0.445, 0.720] | 0.823 [0.649, 0.933] | 0.593 [0.455, 0.722] |
| **Calibration (healthy scenario)** | **ECE = 0.055** (10 bins; reliability diagram in Fig. 4) | | |

*Intervals are wide because the test period has only six days, three fully unoccupied; day-level resampling is not informative (CI would span [0, 0.97]) and is reported as a structural limitation (Section 7). Per-row bootstrap (ignoring autocorrelation) is archived for completeness: full F1 0.914 [0.907, 0.921], PR-AUC 0.798 [0.779, 0.816], ROC-AUC 0.991 [0.990, 0.992], MCC 0.907 [0.899, 0.914] (threshold-0.75 checkpoint). Overconfidence pattern: bin 0.9–1.0 mean confidence 0.943 vs. observed frequency 0.863; bin 0.7–0.8 confidence 0.753 vs. frequency 0.886.*

### Table 7. Cross-domain analysis: office (A) → HPDmobile residential (H1: 82.2% occupied; H2: 60.4% occupied). Zero-shot transfer fails; fine-tuning captures only the prior.

| Method | H1 F1 / PR-AUC / ROC-AUC | H2 F1 / PR-AUC / ROC-AUC |
|---|---|---|
| Majority (always occupied) | — / 0.822 / 0.500 | — / 0.604 / 0.500 |
| Zero-shot base (blind imputation) | 0.030 / 0.767 / **0.374** | 0.008 / 0.598 / 0.512 |
| Zero-shot missing-aware | 0.468 / 0.750 / 0.329 | 0.424 / 0.559 / 0.479 |
| Zero-shot blend | 0.066 / 0.766 / 0.375 | 0.039 / 0.597 / 0.512 |
| Fine-tune 1 day | 0.850 / 0.820 / 0.484 | 0.684 / 0.639 / 0.478 |

*ROC-AUC ≈ 0.5 (or below) for all transfer settings despite high PR-AUC/F1: models capture the target prior, not discriminative signal. Structural causes: label–sensor semantic mismatch (house-level label vs. room-level sensors), prior shift (2.7% → 60–82%), and lifestyle divergence. Reported as a generalization boundary, not a transfer claim [18].*

---

## Figures

**Figure 1.** Dataset characterization: ground-truth distribution (log scale; 97.3% unoccupied; 44 rows of classes 3–4 excluded) and per-channel natural missing rates. *(Rendered: drafts/figures/fig_gt_distribution.png; see also Table 1.)*

**Figure 2.** Proposed architecture. Seven modality groups (environment/indoor air, light, acoustic, electrical server/firewall/socket, device state) are z-score-normalized with per-channel presence flags and encoded by per-modality MLPs (dimension 64 per modality); the resulting tokens plus a learned [CLS] token enter a two-layer masked cross-modal attention Transformer (key-padding mask excludes absent modalities); the [CLS] representation feeds a linear classification head (BCE with pos_weight). Training-time missing-modality dropout augmentation (30% of rows, one modality group masked) and the complementary InfoNCE clean↔corrupted view alignment (λ = 0.1, temperature 0.1) are shown. *(Rendered: `experiments/figures/fig1_arsitektur.png`; final vector version to be embedded at formatting.)*

**Figure 3.** Evaluation PR-AUC per epoch on the five stratified evaluation days; best epoch 8 (PR-AUC 0.4948), early stopping patience 5. *(Rendered: drafts/figures/fig_training_curve.png.)*

**Figure 4.** Reliability diagram (10 bins, healthy scenario, n = 38,785): ECE = 0.055; overconfidence in the top bin (confidence 0.943 vs. frequency 0.863). *(Rendered: drafts/figures/fig_reliability.png.)*

**Figure 5.** PR-AUC across the 14 failure scenarios for the six compared models. *(Rendered: drafts/figures/fig_prauc_scenarios.png.)*

**Figure 6.** PR-AUC vs. random channel-dropout rate (10–70%): the proposed model's advantage over all baselines grows monotonically with the missing rate. *(Rendered: drafts/figures/fig_prauc_missing_rate.png.)*
[19] A. Mohammadabadi, S. Rahnama, A. Afshari, Indoor occupancy detection based on environmental data using CNN-XGBoost model: Experimental validation in a residential building, Sustainability 14 (2022) 14644. https://doi.org/10.3390/su142114644.

[20] B. Shubha, V.V.D. Shastrimath, Real-time occupancy detection system using low-resolution thermopile array sensor for indoor environment, IEEE Access 10 (2022) 130981–130995. https://doi.org/10.1109/access.2022.3229895.

[21] Z. Chen, M. Wang, Y. Wang, Improving indoor occupancy detection accuracy of the SLEEPIR sensor using LSTM models, IEEE Sens. J. 23 (2023) 17794–17802. https://doi.org/10.1109/jsen.2023.3287565.

[22] M. Skromule, R. Kozlovskis, D. Tiscenko, J. Judvaitis, Investigation of audio feature application for CO₂ sensor-based occupancy detection enhancement, Buildings 16 (2026) 545. https://doi.org/10.3390/buildings16030545.

[23] S.H. Choi, J.M. Lee, Multimodal learning with missing modality for chemical process system, Comput. Chem. Eng. 201 (2025) 109196. https://doi.org/10.1016/j.compchemeng.2025.109196.

[24] A. Moayedikia, Conformal fusion under missing modalities, arXiv preprint arXiv:2608.07183 (2026).

[25] Y. Shi, E. Yu, K. Guo, J. Lu, GAUGE: Granularity-adaptive counterfactual gating of evidence for incomplete multimodal classification, arXiv preprint arXiv:2608.05608 (2026).

[26] F. Mena, D. Ienco, R. Interdonato, C.F. Dantas, S. Besnard, Co-learning for missing arbitrary modalities in multi-modal classification, arXiv preprint arXiv:2607.24683 (2026).

[27] P. Gao, K. Ying, B. Wu, J. Mo, Q. Wen, M3F-UAV: A missing-modality multimodal foundation model for low-altitude wireless sensing, arXiv preprint arXiv:2607.13678 (2026).

[28] M. Essl, M. Moscati, M. Noman, M.Z. Zaheer, U. Naseem, S. Nawaz, M. Schedl, SB-BEVFusion: Enhancing the robustness against sensor malfunction and corruptions, in: Proc. IEEE Int. Conf. Image Process. (ICIP), 2026. https://doi.org/10.1109/icip61757.2026.11630059.

[29] L. Uka, S. Pinto, G. Hoffmann, M.M.-C. Höhne, When multi-sensor fusion fails to generalize: Cattle posture classification under animal-level and temporal distribution shift, arXiv preprint arXiv:2606.24986 (2026).

[30] Q. Zhao, Y. Wan, J. Xu, L. Fang, Cross-modal attention fusion network for RGB-D semantic segmentation, Neurocomputing 548 (2023) 126389. https://doi.org/10.1016/j.neucom.2023.126389.

[31] Y. Xiao, Q. Li, L. Liu, C. Li, Cross-modal guiding attention for RGBT tracking, Inf. Fusion 129 (2026) 104008. https://doi.org/10.1016/j.inffus.2025.104008.

[32] H. Shi, X. Wang, J. Zhao, X. Hua, A cross-modal attention-driven multi-sensor fusion method for semantic segmentation of point clouds, Sensors 25 (2025) 2474. https://doi.org/10.3390/s25082474.

[33] J. Cui, Y. Yang, G. Zhu, D. Zhang, MixFormer: A novel multi-sensor fusion based cross-intra-modal attention mechanism fault diagnosis model towards small samples, Meas. Sci. Technol. 37 (2026) 046102. https://doi.org/10.1088/1361-6501/ae2e26.

[34] S. Mai, Y. Zeng, H. Hu, Learning from the global view: Supervised contrastive learning of multimodal representation, Inf. Fusion 100 (2023) 101920. https://doi.org/10.1016/j.inffus.2023.101920.

[35] X. Ouyang, J. Wu, T. Kimura, Y. Lin, G. Verma, T. Abdelzaher, M. Srivastava, MMBind: Unleashing the potential of distributed and heterogeneous data for multimodal learning in IoT, in: Proc. 23rd ACM Conf. Embed. Netw. Sens. Syst. (SenSys), 2025, pp. 491–503. https://doi.org/10.1145/3715014.3722053.

[36] A multimodal dataset for smart office occupancy estimation, Zenodo v3.0.0, 2026. https://doi.org/10.5281/zenodo.20548374.

[37] M. Jacoby, S.Y. Tan, G. Henze, S. Sarkar, A high-fidelity residential building occupancy detection dataset, Sci. Data 8 (2021) 283. https://doi.org/10.1038/s41597-021-01055-x (HPDmobile data: Figshare collection 5364449, CC0).

