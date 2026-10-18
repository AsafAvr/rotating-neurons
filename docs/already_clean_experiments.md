# Testing "Neural Networks Are Already Clean" with ROTATE

A plan for experiments that use ROTATE (this repo) to test the claims in Nick Cammarata's essay
*Neural Networks Are Already Clean* (Goodfire, 15 Oct 2026, already-clean.goodfire.com).

> **Source caveat.** This plan was written from secondary coverage of the essay (news write-ups,
> the LessWrong linkpost and responses), not the essay text itself, and before the companion
> technical paper was out. Check every claim summarised in section 1 against the essay before
> relying on it, and swap in Goodfire's released labels and method once they are public.

## 1. The claims under test

| # | Claim (as reported) | What it is measured on |
|---|---|---|
| C1 | Read each neuron as two **poles** (what drives it strongly up, what drives it strongly down). In Qwen3-32B about 90% of poles have **one main job**, about 7% have two related jobs, and 2-4% are not understood. | An agent's labels of a pole's strong activations. The essay's footnote says the share moves with the threshold for weak activations, with depth, and with what counts as "one thing". |
| C2 | Apparent polysemanticity is mostly **recursive generalisation**: a neuron's function spreads to nearby cases and then branches (`iy` → `diy` → `do it yourself`), so the branches look unrelated but are linked by a chain. | Case studies: the InceptionV1 cat/car neuron read as a V-shape detector, curve detectors. |
| C3 | Packing **unrelated** concepts into one neuron (superposition in the strong sense) is rare. The essay does not deny that a model represents more features than it has dimensions. | Follows from C1 and C2. |
| C4 | So neurons are a good enough basis that labelling neurons with agents and making small, targeted edits is a workable plan. | Stated as a programme, not measured. |

## 2. Where the essay and ROTATE differ

1. **Head of the distribution vs. the full spectrum.** A pole's "main job" is read from its strong
   activations. ROTATE splits the neuron's *weights* into several vocabulary channels, and ablating
   a single channel removes a specific set of activations. Both can hold at once: the top of a pole
   comes from the channel with the most gain, while the other channels produce the mid- and
   low-range activations that a "weak activation" threshold throws away. The essay's own footnote
   (the share depends on that threshold) points at exactly this part of the distribution. This
   would also reconcile its ~90% with the ~1.7% from OpenAI's 2023 explanation-simulation scoring,
   which scores the whole activation distribution.
2. **A pole is a one-dimensional cut through a two-dimensional gate.** Qwen3, Gemma-2 and Llama-3
   all use SwiGLU: `a = SiLU(w_gate·x) * (w_in·x)`. SiLU is close to zero for negative inputs (its
   minimum is about −0.28), so the sign of `a` is set mostly by `w_in·x` when the gate is open. The
   gate decides **when** the neuron fires and `w_in` decides **which sign and how much**. If so, the
   two poles share the gate's contexts and differ mainly in content, and any polysemanticity in
   `w_gate` (several contexts that open it) is split across both poles and never shows up in a
   per-pole audit. ROTATE decomposes `w_gate` and `w_in` separately, so it can see this.
3. **What activates a neuron is not what it does.** Both poles write along the same line, `±w_out`.
   A pole label describes inputs; ROTATE's `w_out` channels describe what the neuron promotes or
   suppresses, and the paper shows they are faithful to that effect. A neuron is clean in a
   computational sense only if its output is appropriate in **every** context that triggers it.
4. **"Related" needs a null.** Given enough freedom, an agent can link almost any two concepts by a
   chain (`iy` → `diy` → `do it yourself`). C2 makes a claim about relatedness, so it needs a
   baseline: how related are two channels taken from *different* random neurons?
5. **Related origin is not the same as no interference.** Even if branch B grew out of branch A
   during training, downstream layers read one scalar per neuron. If B's contexts need a different
   output than A's, other features have to fix the interference, and that is superposition in the
   sense that matters for editing, however the branch arose. Section 3 (E3) tests this directly.

The essay does have real points against our side, and the plan has to handle them:

* ROTATE only sees structure that is **aligned with the vocabulary**. Neurons whose features are not
  lexical (syntax, position, abstract state), most common in middle layers, have low kurtosis in
  every direction, and ROTATE has nothing to say about them.
* ROTATE always returns `C` channels. Without a significance floor, "a neuron has many channels"
  can be an artefact of optimising kurtosis in a 2-5k dimensional space against a vocabulary of
  more than 150k tokens. **E0 has to come first.**
* A channel can exist in the weights and never fire on real data. Every claim below about channels
  is tied to activations on a corpus.

## 3. Experiments

Models: Gemma-2-2B-it and Llama-3.1-8B-Instruct (already supported, used in the paper), plus the
**Qwen3 family**. The Qwen3 sizes share one tokenizer and one SwiGLU design up to Qwen3-32B, the
model the essay audited, which gives a scale sweep with a fixed vocabulary. Before running, check
each Qwen3 checkpoint's `tie_word_embeddings` and its padded, untrained vocabulary rows; they need
the same outlier-token treatment as Gemma.

Corpus: a general web sample (e.g. FineWeb) of about 10-50M tokens per model, plus a code slice
and a multilingual slice. Record the pre-activations `g = w_gate·x`, `u = w_in·x`, and `a`.

### E0. A significance floor for channels (do this first)

Run `find_channels` with the paper's hyperparameters on **null vectors**: (i) random Gaussian
vectors with the same norm as the neurons, and (ii) real neuron vectors after a fixed random
orthogonal rotation of the residual space, which keeps norms and the weight statistics but breaks
alignment with the vocabulary. A real channel `k` counts as **significant** if its kurtosis is
above the 99th percentile of the null kurtosis at the same iteration `k`. Report, for each layer,
the number of significant channels per neuron and per weight type (gate, in, out). Use this count,
and not `C = 50`, everywhere below.

### E1. Channel attribution across the activation spectrum (tests C1)

For every activation event, attribute it to channels: compute how much of the event's activation
is removed when each significant channel of `w_gate` (and separately of `w_in`) is projected out of
the weight vector, as in the paper's channel ablation. Assign the event to the channel with the
largest drop.

Bin events by quantile within each pole (top 0.01%, 0.1%, 1%, 10%, 50%). For each bin, report:

* the entropy of the channel assignment, and the share held by the most common channel;
* the share of the pole's total |activation| and of its total downstream effect (|a|·‖w_out‖,
  or the change in loss when the neuron is ablated) that falls in each bin.

| Outcome | Reading |
|---|---|
| One channel dominates every bin (low entropy all the way down) | Strong support for C1 |
| One channel dominates the top bins, entropy rises sharply in lower bins, and lower bins hold a large share of the effect | The "main job" is a fact about the head of the pole. This is the spectrum view, and it explains both the ~90% and the ~1.7% |
| Several channels share even the top bins | C1 fails even on its own terms |

Also replicate the pole labelling itself (an LLM labels each pole from its top examples and judges
one job, two related jobs, or unclear) so our 90% can be compared directly with theirs. Use
Goodfire's labels once they are released.

### E2. Are a neuron's channels related? (tests C2 against a null)

Take neurons with at least 2 significant channels. For each pair of channels, measure:

* **Geometric overlap** after removing the shared neuron direction. All channels sit close to `w`
  by construction (the faithfulness term), so raw cosines are inflated. Compare the residuals
  `c_k − (c_k·ŵ)ŵ`.
* **Semantic relatedness** of the two top-token sets: mean embedding similarity from an external
  sentence encoder, plus a blind LLM judge asked "related, linked by a short chain, or unrelated?".
  The judge sees within-neuron pairs and control pairs mixed together, in a random order.

The control pairs are channels taken from two different random neurons in the same layer, matched
on kurtosis and token count. Also build each neuron's relatedness graph (an edge where the judge
says "related" or "chain") and test whether it is connected more often than null graphs with the
same density. Recursive generalisation predicts **connected chains or trees**; packing unrelated
features predicts that within-neuron relatedness is about the same as the null.

### E3. Is every branch useful? The decisive test (C2 vs. C3)

This test separates "related branches that are all part of one function" from "branches the
network puts up with as interference".

For each neuron and each significant input channel `k` (from E1's attribution), take the contexts
where channel `k` drove the activation. In those contexts, ablate the **whole neuron** (set `a` to
its mean, or to zero) and measure the change in next-token loss.

* Recursive generalisation, with all branches functional, predicts that ablation **hurts in every
  branch's contexts**, because the shared `w_out` is useful wherever the neuron fires.
* Superposition with interference predicts that ablation hurts in the main branch's contexts and has
  **no effect, or lowers loss**, in some secondary branches' contexts, because there the neuron's
  write is noise that other components have to cancel.

Report, for each neuron, the share of the activation mass that sits in branches where the neuron
is useful. This is a computational version of "one main job" and does not depend on how anyone
labels the branches.

A cheap second check, using the weights only: does the `w_out` channel promote tokens that fit each
input channel's contexts? Score the agreement between the top tokens of the input channels and of
the output channels with the same blind judge.

### E4. Poles in the weights (tests the pole framing)

Prediction from section 2.2: the two poles share gate channels and differ in `w_in` channels.

* For each neuron, compare the distribution of **gate**-channel assignments for positive-pole events
  with the one for negative-pole events (Jensen-Shannon divergence), against a null that shuffles
  events between neurons.
* Check that the positive and negative pole labels line up with the positive and negative **tails**
  of the `w_in` channels. ROTATE already chooses each channel's tail by its skewness (`tail_mask`).

If the gate is shared, "two poles, one job each" is really "one context, a signed content". That is
a cleaner description than the essay's, and it means a per-pole audit misses polysemanticity in the
gate. If instead the poles use separate gate channels, the per-pole framing holds up.

### E5. The editing claim (tests C4)

Pick neurons whose main job is clear (from E1). Make the small edit the essay has in mind: ablate or
steer the neuron to change its main-job behaviour. Then measure **collateral change** in the contexts
of the secondary channels from E1 and E3 (loss, and changes to specific behaviours). Compare a
neuron-level edit with a **channel-level** edit: remove only the main channel from `w_gate` or
`w_in`, which ROTATE makes possible. If channel-level edits get the same target effect with clearly
less collateral damage, the neuron is not the right unit for editing, even when its top
activations are clean.

### E6. Scale sweep in the Qwen3 family (tests the "larger is cleaner" trend)

E0 and the weights-only part of E2 need no forward passes, so they can be run on a sample of
neurons from every Qwen3 size up to 32B. Plot against model size, by relative depth:

* the number of significant channels per neuron;
* the share of neurons whose channels are all related (E2);
* the share of neurons with no significant channel. ROTATE cannot see these neurons; check whether
  they line up with the essay's 2-4% "not understood" share or with its "one job" neurons.

The essay predicts unrelated channels become rarer with scale. If the count of significant,
unrelated channels per neuron stays flat with width, that is evidence against C3.

## 4. Which results support or count against the essay

| Result | Supports the essay | Supports the spectrum / superposition view |
|---|---|---|
| E1: entropy across quantiles | Low in every bin | Rises in the lower bins, which also hold a large share of the effect |
| E2: relatedness vs. null | Clearly above the null; connected chains | About the same as the null |
| E3: usefulness of secondary branches | Ablation hurts in all branches | Ablation has no effect, or helps, in some branches |
| E4: gate overlap between poles | The poles have separate gates | The poles share a gate and the gate has several channels |
| E5: unit of editing | Neuron edits are as precise as channel edits | Channel edits are clearly more precise |
| E6: scale | Unrelated channels shrink with size | Flat |

The likely outcome is mixed: the top of each pole is clean, as the essay says, while the full
spectrum has several channels, as ROTATE finds. E3 then decides whether those extra channels are
part of the function or interference. Fix the thresholds in advance: the E0 percentile, the
quantile bins, the judge prompt, and what counts as "hurts" in E3.

## 5. Order and cost

1. **E0** (weights only, hours on one GPU per layer sample).
2. **E2, weights-only part, and E6** (weights only; the Qwen3-32B sample is the expensive part).
3. **E1 and E4** (one forward pass over the corpus per model, caching `g`, `u` and channel projections).
4. **E3 and E5** (ablation runs; restrict to about 100 neurons per layer, as in the paper).

Code still needed: Qwen3 entries in `rotate/models.py` (with outlier-token weights), a null-vector
driver for E0, an activation-caching script that also stores channel projections, and the
attribution and ablation code for E1, E3 and E5.
