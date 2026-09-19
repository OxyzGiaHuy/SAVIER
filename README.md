# SAVIER

**Source-Anchored Visual Inspection and Evidence-Guided Repair for Vietnamese Cultural Text-to-Image Generation**

SAVIER is a one-pass, multi-agent *prompt-repair* module that sits behind any text-to-image generator. A first
image is generated from the prompt; three role-specialised agents sharing one MLLM backbone then (i) anchor
what must be preserved and what counts as culturally correct to the **original prompt and hand-chosen
Wikipedia pages**, (ii) **inspect the image without seeing the prompt**, and (iii) turn the discrepancies into a
short, positive repair prompt. The generator is run once more with the same seed and curated reference
photographs. Nothing is edited in pixel space; the output is a repaired prompt and the image it produces.

> Contracts (cultural evidence) are drafted offline from Wikipedia-grounded descriptions using an LLM and
> subsequently inspected and corrected by a Vietnamese annotator. The three agents are role-specialised
> agents sharing the same MLLM backbone. Curated reference images are used during the repair generation.

## Contributions

1. **VN-Culture-50 prompts + source-anchored visual contracts.** 50 Vietnamese cultural prompts
   (`data/prompts_simple.json`, bilingual, 6 categories) and, for each, a human-verified *visual contract*
   (`data/contracts_v2.json`): 2–5 camera-observable identity attributes with importance weights and a
   Wikipedia source each, plus 2–3 *confusables* (the look-alike objects generators actually drift to:
   qipao for áo dài, cart for shoulder-pole vendor, bánh tét rolls for bánh chưng, …). Contracts are frozen
   before generation and are used **only for evaluation**; the agents never read them at inference time.
2. **The SAVIER repair protocol.** Curator → Observer → Refiner, one repair, no acceptance loop, every
   inter-agent message a structured JSON that is machine-validated (verbatim quotes, no negation, no
   framing changes, actions already satisfied in I0 are discarded). We report the method on SDXL 1.0 and
   FLUX.1-dev with a fixed-seed, one-image-per-arm protocol and independent metrics.

## Method

```
P0 (original prompt) ──► G ──► A                              (arm A: prompt-only baseline)
P_ct = P0 + Culture-TRIP expansion ──► G ──► I0               (arm B: prompt-refinement baseline)

I0 ──► O  Prompt-Blind Image Observer ──► visual report
P0 + curated Wikipedia ──► C  Source-Aware Prompt Curator ──► Preservation Card + Evidence Card
report + cards ──► R  Discrepancy-Guided Prompt Refiner ──► ≤3 positive repair actions, phrases to drop
P1 = P0 (verbatim) + trimmed expansion + "Keep clearly visible: …" + actions
P1 + reference photos ──► G (same seed) ──► I1                (arm C: SAVIER)
```

| agent | modality | sees | produces |
|---|---|---|---|
| **C** Source-Aware Prompt Curator | text | `prompt_vi`, `prompt_en`, hand-chosen Wikipedia pages (≤4k chars, cut around entity nouns) | *Preservation Card* (everything the prompt asks for, no citation needed) and *Evidence Card* (≤3 identity cues, ≤2 conditional cues, ≤2 positive disambiguators; every cue carries a **verbatim** Vietnamese quote and URL) |
| **O** Prompt-Blind Image Observer | vision | the image **only** | exhaustive report in 10 fixed sections; never names a culture, never guesses outside the frame |
| **R** Discrepancy-Guided Prompt Refiner | text | P0, expansion text, both cards, the report | gap analysis (`missing_prompt_explicit`, `missing_cultural_identity`, `contradictions`, `already_satisfied`, `uncertain_no_repair`), ≤3 repair actions, `drop_phrases` |
| G | generator | P1 (+ 1–2 reference photos) | I1 |

All three agents run on **Mistral-Small-3.1-24B-Instruct** with different system prompts (`savier/llm.py`,
`savier/agents.py`). Generators (`savier/generators.py`): SDXL 1.0 base (DPM++ 2M Karras, 30 steps,
CFG 5, compel for prompts > 77 CLIP tokens, IP-Adapter Plus ViT-H with two photos, scale 0.5) and
FLUX.1-dev (28 steps, guidance 3.5, T5 `max_sequence_length=512`, XLabs IP-Adapter with one photo, scale
0.6). **No negative prompt is used in any arm.** Same seed for A, I0 and I1.

### Machine checks on agent outputs

Each check encodes a failure that was observed in development, not a hypothetical one.

| where | check | observed failure it blocks |
|---|---|---|
| C | `quote_vi` must occur verbatim in a Vietnamese source **and** share a content word with `cue_vi` | a verbatim quote about boiling broth attached to a cue "stone bowl" |
| C | ≤4k characters of source text in total, two separate calls | 12k-character pages truncated the JSON reply |
| R | actions containing negation or framing words are dropped | diffusion models render the negated noun; framing changes destroyed correct images |
| R | each action is scored on I0 first; actions already ≥ 8/10 are removed and replaced by identity cues | "a high collar" occupied a repair slot on an image that already had one |
| R | `drop_phrases` must occur verbatim in the expansion and never in P0 | the Culture-TRIP expansion pushed áo dài prompts toward qipao on both generators |
| P1 | the Keep clause lists the Preservation Card's supporting objects | reference conditioning dropped the "school gate" and the "tray" the prompt asked for |

If R produces no valid action the module is a **no-op** and I1 = I0; the no-op rate is reported.

## Evaluation protocol

Metrics are computed by `scripts/eval_metrics.py` with an evaluator that is **not** the agents' model
(Qwen2.5-VL-7B-Instruct), on images whose filenames are hashed and whose (image, statement) order is
shuffled with a fixed seed, so the evaluator cannot tell I0 from I1. Contracts are frozen before generation
(`data/contracts_v2.json`, commit `e1c9da0`). Confidence intervals are bootstrapped **by prompt** (2000
resamples).

Let `C` be the contract's `required` items (`scoring = required`), `w_i` their `importance`, and
`y_i(I) ∈ {0, 0.5, 1}` the evaluator's judgement (YES / PARTIAL / NO) for item `i` on image `I`.

**VCFS ↑ — Visual Contract Fulfillment Score**
```
VCFS(I) = 100 · Σ_{i∈C} w_i y_i(I) / Σ_{i∈C} w_i
```
`conditional` items are reported separately; prompt-specific details (e.g. "white") are left to VQAScore.

**CCR ↓ — Cultural Confusion Rate**, over confusable cues `F` with severities `v_j` (all 1 in `contracts_v2`)
```
CCR(I) = 100 · Σ_{j∈F} v_j z_j(I) / Σ_{j∈F} v_j        (N/A when a prompt has no confusables)
```

**NRG ↑ — Net Repair Gain.** With `D0 = {i : y_i(I0) < 1}` (missing on I0) and `S0 = {i : y_i(I0) = 1}`:
```
FR  = Σ_{i∈D0} w_i · max(0, y_i(I1) − y_i(I0)) / Σ_{i∈D0} w_i
RR  = Σ_{i∈S0} w_i · max(0, y_i(I0) − y_i(I1)) / Σ_{i∈S0} w_i
NRG = 100 · (FR − RR)                                    (N/A when D0 = ∅; the correct no-op rate is reported instead)
```

**VQAScore ↑** (Lin et al. 2024) against the **original** English prompt P0, never against P1:
`P(Yes | "Does this figure show "P0"? Please answer yes or no.")`. The reference implementation uses
`clip-flant5-xxl`; `--vqa-backend t2v` calls it through `t2v_metrics`, `--vqa-backend qwen` computes the same
quantity with Qwen2.5-VL-7B. The backbone is printed in every table — numbers from different backbones are
not comparable.

**CAIRE ↑** (Yayavaram et al., EACL 2026): cultural relevance to the label *Vietnam* on a 1–5 scale, run with
the authors' code and knowledge base (`--target_list "Vietnam, China, Japan, South Korea, Thailand, …"`);
merged with `--caire-csv`.

We report I0, I1 and Δ = I1 − I0 per generator, plus paired counts (NRG > 0 / = 0 / < 0 / N/A). Human
pairwise preference is the primary evidence and is collected separately.

### Results — 50 prompts, one seed (S001–S050)

Independent evaluator Qwen2.5-VL-7B (blind to arm and order); contract `contracts_v2.json` frozen before
generation; 95% bootstrap CIs by prompt (2000 resamples). Full tables and every raw evaluator answer are in
`results/metrics/` (`metrics_final_*` for the main arms, `metrics_abl_*` for the ablation controls).

| generator | arm | CAIRE-VN² ↑ | CAIRE-CN² ↓ | VQAScore¹ ↑ | VCFS ↑ | CCR ↓ | NRG ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| SDXL 1.0 | A · prompt only | 2.66 | 2.46 | 0.720 | 52.9 | 20.8 | – |
| SDXL 1.0 | I0 · refined prompt | 3.00 | 2.54 | 0.610 | 55.6 | 15.3 | – |
| SDXL 1.0 | **I1 · SAVIER (ours)** | **3.44**³ | **2.38**³ | **0.730** | **67.9** | 16.8 | **39.1** [24.9, 53.3] |
| SDXL 1.0 | Δ (I1 − I0) | +0.44 [−0.02, +0.90] | −0.16 [−0.54, +0.22] | +0.11 [−0.00, +0.24] | **+12.3** [+4.6, +20.3] | +1.5 [−4.0, +7.0] | |
| FLUX.1-dev | A · prompt only | 2.50 | 2.54 | 0.670 | 44.8 | 13.3 | – |
| FLUX.1-dev | I0 · refined prompt | 2.50 | 3.08 | 0.700 | 59.1 | 15.0 | – |
| FLUX.1-dev | **I1 · SAVIER (ours)** | **2.92**³ | **2.88**³ | **0.770** | **66.0** | **13.7** | **20.7** [5.5, 35.6] |
| FLUX.1-dev | Δ (I1 − I0) | **+0.42** [+0.06, +0.78] | −0.20 [−0.56, +0.16] | +0.07 [−0.02, +0.16] | **+6.9** [+0.2, +13.8] | −1.3 [−7.0, +4.3] | |

Paired outcome of I1 vs I0 (NRG): SDXL 24 > 0 · 12 = 0 · **2 < 0** · 12 N/A (I0 already complete);
FLUX 19 > 0 · 16 = 0 · **5 < 0** · 10 N/A. All 100 units were produced by the same final code version
(15 units first generated with an earlier revision were re-run with the same seed; A and I0 are unchanged).

¹ VQAScore here uses a Qwen2.5-VL-7B backbone (same question and P(Yes) formula as the paper, different
model); it is comparable across arms in this table, not with numbers published for `clip-flant5-xxl`.
² CAIRE (Yayavaram et al., EACL 2026) run with the authors' code, knowledge base and Qwen2.5-VL-7B scorer on the
same 300 blinded images; culture labels `Vietnam, China, Japan, South Korea, Thailand, India, France, United States`;
VN = relevance to Vietnam, CN = relevance to China (1–5). Paired ΔVN: SDXL 17 > 0 / 25 = 0 / 8 < 0; FLUX 16 / 27 / 7.
CAIRE measures cultural *relevance*, not entity correctness (a Vietnamese street scene with the wrong object still
scores high) — read it together with VCFS. Human pairwise preference is reported separately when available.
³ CAIRE was scored before 15 units (S001–S010 SDXL, S001–S005 FLUX) were re-run with the final code; the CAIRE
columns therefore refer to the earlier I1 images of those units and will be refreshed.

### Ablation — what the agents add beyond reference conditioning

Same seed and the same I0 for every arm; the only change between consecutive rows is the component named.
`refs-only` = Culture-TRIP prompt + IP-Adapter on two curated photos, no agent. `keep-only` = refs-only + the
Preservation Card's "Keep clearly visible" clause (one Curator call, no Observer, no Refiner). Full table with
CIs and paired differences: `results/metrics/ablation_50.md`; images per prompt in `results/<model>/<pid>/`.

| generator | arm | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG >0 / =0 / <0 |
|---|---|---:|---:|---:|---|
| SDXL 1.0 | I0 · refined prompt | 0.612 | 55.6 | 15.3 | – |
| SDXL 1.0 | + reference (refs-only) | 0.713 | **72.3** | **14.2** | 29 / 6 / 3 |
| SDXL 1.0 | + keep-only | 0.722 | 70.7 | **14.2** | 27 / 8 / 3 |
| SDXL 1.0 | **+ SAVIER (ours)** | **0.725** | 67.9 | 16.8 | 24 / 12 / **2** |
| FLUX.1-dev | I0 · refined prompt | 0.696 | 59.1 | 15.0 | – |
| FLUX.1-dev | + reference (refs-only) | 0.713 | 65.3 | 16.8 | 19 / 13 / 8 |
| FLUX.1-dev | + keep-only | 0.711 | **66.1** | 17.2 | 22 / 13 / 5 |
| FLUX.1-dev | **+ SAVIER (ours)** | **0.767** | 66.0 | **13.7** | 19 / 16 / **5** |

Paired SAVIER − refs-only: SDXL ΔVQA +0.01 [−0.03, +0.07], ΔVCFS −4.4 [−9.9, +0.9], ΔCCR +2.7 [−0.7, +6.7];
FLUX ΔVQA +0.05 [−0.02, +0.13], ΔVCFS +0.7 [−5.6, +7.0], ΔCCR −3.2 [−8.8, +2.2]. Reading: the reference photos
carry most of the VCFS gain over I0 on both generators; the Observer/Refiner pair adds prompt fidelity
(VQAScore, clearest on FLUX: +0.05) and fewer regressions (NRG < 0: 2 vs 3 on SDXL, 5 vs 8 on FLUX) and, on FLUX,
lower confusable rate (CCR 13.7 vs 16.8). On SDXL the Refiner's edits cost VCFS (−4.4, CI touching 0) — the
repairs move the image away from the reference-conditioned layout without the evaluator crediting the change.
These are the honest numbers for one seed; none of the SAVIER-vs-refs-only differences is significant at 95%.

## Repository layout

```
savier/
  llm.py          shared Mistral-Small-3.1 backbone, JSON extraction
  wiki.py         per-page Wikipedia extracts, passage cutting
  agents.py       C · O · R, machine checks, P1 construction
  generators.py   SDXLGen, FluxGen (no negative prompt, optional IP-Adapter)
  pipeline.py     arms A / I0 / I1, transcript, contact sheet
scripts/
  run_savier.py   end-to-end run
  fetch_wiki.py   build data/wiki_curated from the contract URLs
  eval_metrics.py VCFS · CCR · NRG · VQAScore (+ CAIRE merge)
  grid_models.py  multi-generator comparison grid
data/
  prompts_simple.json      50 prompts (vi / en, category, entities)
  contracts_v2.json        frozen visual contracts (evaluation only)
  culture_trip/            Culture-TRIP refined prompts (P_ct) for the 50 prompts
  wiki_curated/            curated Wikipedia extracts used by the Curator
```

## Running

```bash
pip install -r requirements.txt
python scripts/fetch_wiki.py --ids S001,S002,S003 -o data/wiki_curated/smoke.json
python scripts/run_savier.py --ids S001,S002,S003 --backend sdxl --refs-dir /path/to/reference_images \
       --wiki data/wiki_curated/smoke.json --out runs/sdxl
python scripts/run_savier.py --ids S001,S002,S003 --backend flux --refs-dir /path/to/reference_images \
       --wiki data/wiki_curated/smoke.json --out runs/flux
python scripts/eval_metrics.py --run SDXL=runs/sdxl --run FLUX.1-dev=runs/flux -o runs/metrics
python scripts/grid_models.py --run SDXL=runs/sdxl --run FLUX.1-dev=runs/flux -o runs/grid.png --cell 1024
```

`--refs-dir` expects `selected/<prompt_id>/` (photos used for conditioning) and `candidates/<prompt_id>/`
(held-out photos used only by the evaluator). Mistral-Small-3.1-24B (~48 GB bf16) and SDXL fit together on
one 80 GB GPU; FLUX.1-dev runs with model CPU offload.

## Data and licensing notes

- Prompts, contracts, Culture-TRIP outputs and Wikipedia extracts are released here. Wikipedia text is
  CC BY-SA 4.0.
- The reference photographs used for conditioning and held-out evaluation are third-party web images and
  are **not** redistributed; the pipeline runs without them (no reference conditioning).
- Culture-TRIP (NAACL 2025) is used as the prompt-refinement baseline; its repository is not included.

## Citation

```
@misc{savier2026,
  title  = {SAVIER: Source-Anchored Visual Inspection and Evidence-Guided Repair for Vietnamese Cultural Text-to-Image Generation},
  author = {Thai Gia Huy},
  year   = {2026}
}
```
