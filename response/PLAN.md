# Response plan: ROTATE vs. "Returning to the Neuron Basis" (Shone & Dunlap, Anthropic Fellows, 23 Nov 2026)

Sources read in full: Part 1 (alignment.anthropic.com/2026/neuron-basis-theory/), Part 2
(…/neuron-basis-applications/), arXiv:2604.06005v2 (main text and Appendices A–E), and this repo.

---

## 0. TL;DR

* The two works disagree less than it first looks. Both treat **neurons as the unit** and read
  **weights through a vocabulary lens**. Both split by activation sign (ROTATE §6 polarity split ≈
  the blog's ±w). Both find that activation-based descriptions miss things (blog Fig. 1 "John"
  neuron; ROTATE beats MaxAct++). ROTATE's "rotations" act on one neuron's weight vector in
  residual space. They never mix neurons, so ROTATE makes no claim against the neuron basis.
* **The real disagreement is narrow.** Does one sign of a neuron's output vector carry one concept
  (blog: type 3 ≤ ~1%), or a bundle of several vocabulary channels (ROTATE's implication)? There is
  a secondary disagreement on whether input-side multiplicity ("type 1") is harmless.
* **On that point, ROTATE's current evidence is the weakest and the blog's is the strongest.**
  ROTATE has no null model, its output test is Patchscopes at block 0 rather than in-context
  ablation, and its input ablation is partly circular. The paper alone cannot rebut the ≤1% claim.
* **The blog has real gaps on weights.** See G1 to G6 in §2.2. The main ones: the type-3 test
  requires the neuron to *fire on both topics*, so one-way output interference is never counted;
  recall is measured on easy synthetic cases; by the blog's own census, 56–81% of output vectors
  have J-share within the random-direction range. So "one concept per sign" rests on the top
  tokens of vectors that those tokens mostly don't explain.
* **Plan:** (1) reproduce their setup (Llama-3.1-8B-Instruct, J-lens, their four checks);
  (2) null-calibrate ROTATE, which gates everything; (3) test the output side in context;
  (4) show the practical value of sub-neuron channels on the input side; (5) choose the write-up
  from the outcomes in §7.

---

## 1. Claims map

| Topic | Blog | ROTATE paper | Real disagreement? |
|---|---|---|---|
| Unit of analysis | Neurons ("context-gated steering vectors") | Neurons, each decomposed into vocabulary channels | **No.** Both are neuron-basis works. The difference is granularity. |
| Weights vs activations | Output vector *w* read with J-lens; input side read from activations and descriptions | gate/in/out weights all decomposed, data-free; activations used only for evaluation | Methodological. This is where "missing the weights" applies. |
| Activation sign | Treat +w / −w separately (gate-open rule) | §6: split w_in channels by skew polarity; Fig. 12 neuron 9005 is a clean type-2 example | **No.** ROTATE did this earlier; worth citing. |
| Input-side multiplicity | Type 1 (fires on unrelated inputs, adds one concept) is common and harmless | Gate channels partition top activations (faithfulness 0.46–0.71, completeness 49–60%) | Agree on the fact. Disagree on whether it matters (§4, Phase 3). |
| **Output per sign** | Type 3 ≤ 0.4–1.1% of signed directions with an effect | w_out ≈ up to 50 channels; e.g. 9005(+) promotes polarity words + code-closing syntax + dashes | **Yes, the crux.** Untested by ROTATE in context. |
| Activation-based descriptions | Miss function (John neuron) | ROTATE descriptions win 2–3× (win-rate ratio) vs MaxAct++ | **No.** Supportive. |
| Lens | J-lens (Gurnee et al. 2026) | Logit lens W_U; W_Eᵀ for Llama layers ≤ 12 | Confound that must be controlled (E1.2). |

---

## 2. Honest assessment

### 2.1 ROTATE's vulnerabilities (fix these first; the response will invite scrutiny of the paper)

1. **No null model.** Kurtosis is maximised over projections onto the columns of U, and U is full of
   token clusters (case and morphological variants). So high-kurtosis directions exist near
   *any* vector. "50 channels" is a budget, not an estimate. A single Householder reflection can
   map w to any vector of equal norm, so the method is really "find a high-kurtosis vector near
   w". The "rotation" wording should not suggest a basis change.
2. **C.9 cuts both ways.** A median rank of 122,302 of 256,000 for channel tokens in w's raw
   projection reads, to a sceptic, as "channel tokens are unrelated to w".
3. **Input ablation is partly circular.** v* = argmax_v x·v, so ablating v* lowers w·x by
   construction. It needs controls matched on x·v and on cos(w, v).
4. **The output test is not in context.** w_out is injected at the input to block 0 and decoded
   with Patchscopes, and ablating v removes v's top tokens almost by construction. The blog's
   interventions (zeroing the neuron on real text and measuring next-token log-probs) are the
   stronger standard.
5. **Scale and judges.** Two models, 100 neurons per layer, LLM-generated examples and LLM judges.
   "2–3×" is a ratio of win rates and is near-tied against MaxAct+VocabProj on top Pile activations.
6. **Polysemanticity is assumed from citations, not measured.**

### 2.2 Gaps in the blog regarding weights (where a response has traction)

* **G1. The taxonomy operationalisation has a hole.** Check 1 requires strong activation on
  documents about *both* topics. A neuron that fires on A and adds both A and B is in none of
  types 1–3 and is never tested. That one-way interference is the classic superposition signature.
  (Part 1, App. B.1.)
* **G2. Recall is optimistic.** Synthetic neurons are *equal sums* of two full output vectors.
  Check 1 is skipped. The topic-proposal step is not included in recall, and the procedure that
  gives topic pairs for the random sample (B.2) is not specified. The ≤1% bound is corrected only
  for checks 2–4 on easy cases.
* **G3. Readouts see the dominant component only.** In A.1, 56–81% of neurons have "middle"
  J-share, which is inside the 1st–99th percentile of random directions. The median neuron's
  J-share is 1.02–1.26× the median random direction's. For middle J-share, only 31% of readouts are
  rated related to the top passages. ROTATE C.9 finds that channel tokens fall outside the raw
  top-200 97% of the time.
* **G4. The checks exclude whole channel types.** Check 3 excludes same-category pairs. Check 4
  excludes word-piece effects. Syntax, subword and formatting channels are out of scope by
  construction.
* **G5. Input weights are not examined.** The input side is read from top passages and Transluce
  descriptions. g and u are used only as filters (C.2) and for trigger voting (Part 2). Type 1 is
  called harmless, but it costs them in their own applications. WMDP-cyber vs security news gives
  AUROC 0.49 for neuron probes against 0.95 for 40-label probes (Part 2 Limitations). They also
  acknowledge that removing shared multi-hop neurons hits other facts.
* **G6. The detection threshold hides weak components.** The thresholds are ≥3σ over 20
  norm-matched random directions and ≥0.03 nats. Weak components spread across many neurons,
  which is the distributed form of superposition, are invisible by design.
* **G7. ROTATE is not cited** (COLM 2026), despite overlap on weight-based neuron readouts beating
  MaxAct descriptions and on sign-split analysis. Raise this once, politely. Don't claim priority
  for "in-context two-concept tests": ROTATE didn't run those.

---

## 3. Principles for the response

1. **Reproduce before rebutting.** Use their model (Llama-3.1-8B-Instruct is common to both
   works), their lens (J-lens) and their checks, reimplemented to App. B.1 and validated on their
   Table B1 neurons.
2. **Null-calibrate every count.** No channel counts without matched nulls.
3. **Pre-register predictions and decision thresholds (§6) before running E1 and E2.** Adopt
   their own standard: "state what is measured and what result would show the claim to be wrong".
4. **Make the core claim rest on causal, in-context evidence,** not only on LLM judges.
5. **Share the draft with the authors** about a week before posting. Ask for their type-3 candidate
   lists, the random-sample topic pairs, fact-trace neuron sets and synthetic WMDP hard negatives.

---

## 4. Experiments

Priority: **P0** = needed for any write-up, **P1** = strongly recommended, **P2** = stretch.

### Phase 0: common ground (P0, week 1)

**E0.1 Port ROTATE to their setting.**
* Add a `jlens` projection to `rotate/models.py`: rows of W_U·J_L (final-norm gain folded in),
  row-normalised to match their cosine readout. Use the J-lens files of Gurnee et al. (2026) or
  the Neuronpedia J-lens for Llama-3.1-8B.
* Add configs for Qwen3-8B, and OLMo-3-32B if compute allows.
* Validate by reproducing J-share for about 200 Llama neurons against 2,048 random directions, and
  the high/middle/low split in A.1.

**E0.2 Reimplement the four type-3 checks exactly (B.1).**
* Data: Pile-10k minus Books3, BookCorpus2, StackExchange and OpenWebText2 (7,070 docs).
* Thresholds: 99.9th percentile, ≥5 docs, +0.07 share, 0.03 nats, z ≥ 3 against 20 random
  directions, the quarter rule and the word-continuation rule.
* Validation: Llama L20 n2287, L16 n60, L21 n2095 and L20 n7696 must pass. The John neuron L28
  n2760 must not. Reproduce synthetic recall of 62–81%.

**E0.3 Sanity check.** Reproduce the John neuron (L28 n2760): −w readout ≈ John variants, and the
gate opens only for negative activations.

### Phase 1: is the weight-space channel structure real? (P0, gates everything)

**E1.1 Null-calibrated channel count.**
* *Vectors:* w_gate, w_in, w_out for 300 neurons per layer. Llama layers 4, 8, 12, 16, 20, 24 and
  28; Gemma layers 4 and 18 for continuity with the paper.
* *Nulls:*
  1. Isotropic Gaussian, norm-matched.
  2. Gaussian with the **mean and covariance of that layer's weight matrix**. This keeps the
     layer's geometry relative to U and removes neuron-specific structure. It is the key null.
  3. The same neuron at **initialisation and early checkpoints** (OLMo-2-1124-7B, already used
     in App. B).
* *Per-channel statistics:*
  1. Kurtosis of the projection.
  2. **Token-cluster alignment:** cos(w, c_T), where c_T is the normalised signed sum of the lens
     directions of the channel's top-k tokens, scored against the null at the *same channel rank*.
     Same-rank comparison controls for the selection bias that ROTATE applies to null vectors too.
  3. Blind LLM coherence rating on shuffled real and null channels.
  4. For gate/in only, **faithfulness of fake neurons:** σ(n·x)(u·x) from cached residuals,
     then the paper's §5.2 protocol on n.
* *Output:*
  * K_sig per neuron: channels whose statistic 2 exceeds the null's 99th percentile, with
    BH correction.
  * Fraction of ‖w‖² in significant channels, relative to null. Compare it with J-share relative
    to its own null, i.e. head-to-head with G3.
* *Decision:* Gate A (§7).

**E1.2 Lens robustness.** Repeat E1.1 for w_out under the logit lens, the J-lens and W_Eᵀ. Report
K_sig per lens and the token Jaccard of matched channels. If the structure appears under only one
lens, say so.

**E1.3 Fix the circular ablation (paper hygiene).** Re-run §5.2 causal validity with two controls:
(a) a random channel matched on cos(w, v) and x·v; (b) a random direction in span(w, v*).

### Phase 2: the output side, i.e. is one sign one concept? (P0, the core)

**E2.1 In-context channel ablation** (replaces the Patchscopes test).
* *Setup:* sampled neurons and their *used* sign(s) under the blog's gate-open rule, on Pile-10k
  contexts where the neuron fires strongly.
* *Interventions:*
  * Necessity: replace a·w with a·(w − (w·v̂)v̂) for each significant channel v.
  * Sufficiency: replace a·w with a·(w·v̂)v̂.
* *Measure:* Δlog p of T_v, Δlog p of other channels' tokens, and KL of the next-token
  distribution.
* *Controls:* norm-matched random directions (the blog's control) and full-neuron zeroing.
* *Metrics:*
  * Dominant-channel share D = max_j e_j / Σ_j e_j.
  * S = number of channels with in-context effect z ≥ 3 over the random-direction ablations.
* *Efficiency:* screen with attribution patching, then confirm exactly on the top 10% plus a
  random 10%.

**E2.2 Context × channel matrix (the key experiment).**
* *Grouping:* group firing contexts by their top input channel (ROTATE gate channels, restricted
  to significant ones). Cross-check the grouping with an independent LLM topic labeller.
* *Matrix:* M[k, j] = effect of ablating output channel j, in context group k, on T_j and on loss.
* *Reading:*
  * **Diagonal** (input channel k ↔ output channel k, e.g. 9005: gate "until" ↔ out "until/Till"):
    several key→value memories share one neuron.
  * **Off-diagonal** (the neuron raises T_j in contexts of type k): one activation adds
    concept j where it doesn't belong. This is the blog's type-3 notion, measured *without*
    check 1 (G1).
* *Calibration on their labelled neurons:* Table B1 type-3 neurons should show off-diagonal mass.
  High-J-share neurons (e.g. L22 n5208 "call") should show D ≈ 1. If they don't, the measure is
  broken.

**E2.3 Their four checks, with ROTATE-proposed topics** (a direct test of the ≤1% bound).
* *Sample:* random sample as in B.2. Llama-3.1-8B, middle layers (their search used 8–28),
  ≥1,000 signed directions with a measurable effect. Add Qwen3-8B if compute allows.
* *Topic proposers:*
  * P1: readout clustering (theirs, reimplemented).
  * P2: pairs of significant ROTATE output channels, named by an LLM.
  * P3: the union of P1 and P2.
* *Variants:*
  * V1: all four checks as written.
  * V2: one-way, i.e. drop check 1 and require check 2 only in the direction "zeroing lowers B's
    words in A's documents".
* *Recall, end to end with the proposer in the loop:* synthetic w = cosθ·ŵ_A + sinθ·ŵ_B with
  θ ∈ {15°, 30°, 45°}. If feasible, also synthetic gates, so that check 1 applies.
* *Report:* recall-corrected rates per proposer and variant, next to their 0.4–1.1%.

### Phase 3: the input side, i.e. type 1 is not harmless (P1, constructive)

**E3.1 John neuron, a surgical edit.**
* ROTATE on the gate and up vectors of L28 n2760. Find the input channels for the cyber context
  (John the Ripper) and the Gospel context.
* Project the cyber channel out of g (and/or u).
* Measure the neuron's activation and p(John) on John-the-Ripper prompts, Gospel prompts and
  unrelated "John" contexts. Compare with zeroing the neuron, which kills both.
* *Success:* ≥50% loss of the neuron's contribution on cyber prompts with ≤10% on Gospel prompts.
* Extend to 20–50 type-1 neurons, e.g. from their fact traces: the Colombia and Victorian neurons
  shared across novels.

**E3.2 Removing facts with sub-neuron edits.**
* For 4–6 of their 18 facts (Jagr → hockey, Jane Eyre → Brontë, opencv-python → cv2, …), compare
  removal by neuron sets (theirs; ask for the lists, or approximate them with D's attribution) with
  removal by input channels in the *same* neurons.
* *Metrics:* answer no longer top-1; collateral damage on other facts that share those neurons
  (same-nationality athletes, other Victorian novels); general LM loss.
* This addresses their own caveat that "removing the multi-hop paths could also affect other
  facts".

**E3.3 Detection specificity.**
* *Benchmark:* AxBench Concept500 with their exact neuron pipeline (top-20 neurons by
  Qwen3-Embedding-4B similarity of Transluce description + J-lens readout). Their AUROC is 0.903;
  on hard negatives the neurons get 0.889 and the best SAE baseline 0.834.
* *Variant:* refine to channels by keeping, for each of the 20 neurons, the gate channel closest
  to the concept. Score σ(g_ch·x)(u·x) or x·v_ch.
* *Also:* their failure case, WMDP-cyber vs security news (AUROC 0.49). Ask for their synthetic
  set or regenerate an equivalent one.
* *Hypothesis:* the largest gains are on hard negatives.

### Phase 4: stretch (P2)

**E4.1 Weight-difference channels for backdoor triggers.**
* Run ROTATE on Δg and Δu (finetune minus base) through W_E, on the 74 Thoughtworks models. Rank
  tokens by channel membership.
* Compare with neuron voting (top-50 share 54%; AUC of the recovery curve). Focus on the 24
  Qwen2.5 models where all their methods fail.

**E4.2 Scale.** Run E1.1 and E2.1 on a subsample of Qwen3-14B or OLMo-3-32B: does K_sig shrink
with scale?

**E4.3 Population interference.** For a channel's token cluster, sum the components of all neurons
along it in contexts where only another channel is relevant. Is interference cancelled across the
population? This tests the distributed form of superposition (G6).

---

## 5. Compute and resources (rough)

| Block | Estimate |
|---|---|
| E1.1 + E1.2 (≈ 3–4k vectors × 2 lenses × 50 channels, plus nulls) | 20–40 H100-h |
| E2.1 + E2.2 (≈ 300 neurons, ≤ 20 channels, ≤ 1k contexts; screen with attribution patching first) | 10–30 H100-h |
| E2.3 (≥ 1k signed directions, ~5 pairs each, random-direction controls) | 30–60 H100-h |
| E3.x | 10–30 H100-h |
| E4.x | 20–60 H100-h |
| LLM judges and topic namers | moderate API budget |

Access needed: a gated Llama-3.1 HF token, J-lens files, Transluce descriptions, AxBench,
Thoughtworks models, and OLMo-2 checkpoints.

---

## 6. Pre-registration (fill in before running E1 and E2)

For each item, record a prediction, a threshold and what would change our minds:

* Median K_sig(w_out), middle layers, J-lens / logit lens.
* Median S and D in E2.1 (per context group).
* Share of signed directions with one-way type-3 effects (E2.2 off-diagonal, LLM-rated
  unrelated).
* E2.3 recall-corrected rates, P2-V1 and P2-V2.
* E3.1 selectivity; E3.3 ΔAUROC on hard negatives.

*Claude's priors, for calibration only:*
* K_sig(w_out) in middle layers: a small handful, with high variance.
* In context, one channel usually dominates, with interference that is detectable in aggregate.
* P2-V1 somewhat above their bound but low single digits; V2 noticeably higher.
* E3.1 likely works for John-type neurons.
* E3.3 shows a modest gain on hard negatives.
* Most likely overall outcome: Scenario C.

---

## 7. Write-up decision tree

**Gate A (E1.1, w_out, middle layers, both lenses).** Are channels beyond the first above null?
* *No* (median K_sig ≤ 1) → **Scenario A.**
* *Yes* → Gate B.

**Gate B (E2.1/E2.2/E2.3).** Does more than one channel matter in context?
* Median S ≤ 1 and D ≥ 0.7, one-way type-3 < 5%, E2.3 P2-V1 within about 2× their bound →
  **Scenario C.**
* ≥ 10–20% of signed directions with S ≥ 2 within one context group on unrelated topics, or
  P2-V1 ≥ 3× their bound → **Scenario B.**

(Fix the exact thresholds in §6 before seeing data.)

### Scenario A: "Neurons are units; their inputs aren't atoms"
* **Concede the output side.** Don't publish a rebuttal of the ≤1% claim.
* **Format:** a short Alignment Forum / LessWrong post. Content: the replication (E0), the
  methodological notes G1 and G2 as open questions, and the input-side results (E3) as the
  contribution.
* **Paper:** revise the ROTATE narrative in v3. Add the null calibration, drop the implied claim
  of output polysemanticity, and keep the input-side and description results.

### Scenario C (most likely): "Polysemantic weights, near-monosemantic function"
* **Thesis:** the two views are reconciled by a two-level taxonomy. Weight vectors carry
  several significant vocabulary channels (E1). In context, one channel usually dominates, with
  measurable but small interference (E2). The input side is where sub-neuron structure pays off
  in practice (E3).
* **Format:** an AF/LW post plus a short arXiv note.
* **Headline figures:** K_sig against nulls, the context × channel matrix, and the John-neuron
  surgical edit.

### Scenario B: "The type-3 rate depends on how you look"
* **Format:** a full response, as an arXiv paper plus an AF/LW post.
* **Headlines:**
  * Under their own four checks, recall-corrected type-3 rates rise to X% when weight-space
    channels propose the topics.
  * One-way interference (G1) affects Y%.
* Include E3 as the reason this matters for safety tools.

### Applies to all scenarios
* **E3 positive:** add a section "sub-neuron weight edits for surgical removal and specific
  detection". **E3 negative:** report it as a ROTATE limitation.
* **E1.2:** if the result depends on the lens, make the lens the open variable for *both* works.
* **Tone:**
  * Open with the agreements; the blog is largely an ally of weight-based neuron interpretability.
  * Make one polite note about the missing citation.
  * Every core claim must be falsifiable and causal.
  * Avoid "rotation" wording that implies rotating the neuron basis.
* **Venue and timing:** share the draft with Shone & Dunlap at about week 5, post at about weeks
  6–7, and possibly a workshop paper later.

### Skeleton for Scenario C
1. What we agree on: neurons as units, weight readouts, sign split, the limits of MaxAct.
2. The precise question: one concept per sign, or several channels?
3. Weight vectors carry several significant channels (null-calibrated; both lenses).
4. In context: dominance and interference (E2.1, E2.2), and the type-3 rate under different
   proposers (E2.3).
5. Why the input side matters: surgical edits and specificity (E3).
6. Limits of both approaches: lens dependence, vocabulary-orthogonal features, judges.
7. Falsifiable predictions for scale (E4.2).

---

## 8. Week-1 actions
1. Get HF access to Llama-3.1-8B-Instruct, the J-lens files and the Transluce descriptions.
   Provision GPUs.
2. E0.1: `jlens` projection and Qwen3 config in `rotate/models.py`, plus J-share reproduction.
3. E0.2: reimplement the four checks and validate on the Table B1 Llama neurons.
4. Write §6 predictions (all authors) and freeze the thresholds.
5. CPU-feasible pilot: ROTATE (logit lens) on the John neuron, L20 n2287 and L22 n1766, with a
   covariance-matched null, as a first look at Gate A.
6. Email the authors with a short note about the related work and a request for their candidate
   lists, topic pairs and fact-trace neuron sets.
