# CS781 — Capstone Project

A notebook-based legal NLP pipeline built around three language models trained from scratch. The project reuses their representations across entity extraction, rhetorical role prediction, judgment prediction, statute identification, summarization, and prior-case retrieval.

**Stack:** PyTorch · Custom byte-level BPE · Kaggle T4 · FP16  
**Data:** [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)

## Navigation

- [Models and adaptation](#models-and-adaptation)
- [Phase 1: Foundation model pretraining](#phase-1-foundation-model-pretraining)
- [Phase 2: Legal named entity recognition](#phase-2-legal-named-entity-recognition)
- [Phase 3: Rhetorical role prediction](#phase-3-rhetorical-role-prediction)
- [Phase 4: Court judgment prediction](#phase-4-court-judgment-prediction)
- [Phase 5: Legal statute identification](#phase-5-legal-statute-identification)
- [Phase 6: Abstractive summarization](#phase-6-abstractive-summarization)
- [Phase 7: Prior-case retrieval](#phase-7-prior-case-retrieval)
- [Running the notebooks](#running-the-notebooks)
- [Outputs and validation](#outputs-and-validation)
- [References](#references)

## Models and adaptation

| Backbone | Architecture | Parameters | Main components |
|---|---|---:|---|
| `LegalLM` | 12-layer causal decoder | 121.3M | Width 768, GQA, RoPE, RMSNorm, SwiGLU, tied embeddings |
| `LegalEncDec` | 6-layer encoder + 6-layer decoder | 123.8M | Width 768, encoder–decoder attention, sentinel-token support |
| `LegalHybrid` | 20 layers arranged as `MMMA × 5` | 111.0M | 15 Mamba-2 SSD layers and 5 GQA attention layers |

All three use a 1,280-token pretraining context. `LegalEncDec` is a project-defined T5-style architecture, not an imported T5 model. `LegalHybrid` implements chunked SSD scans in PyTorch; its recurrent SSM state is fixed-size during decoding, but its attention KV cache still grows with sequence length.

Downstream notebooks start from the relevant **Phase-1 checkpoint and its matching tokenizer**. They do not require training all preceding downstream phases. The full encoder–decoder is retained for summarization; classification and retrieval use its encoder.

### Experiment selection

Each downstream notebook exposes an `EXPERIMENT` selector and a `PRESETS` dictionary. Available techniques vary by phase:

| Technique | Implementation approach |
|---|---|
| Full fine-tuning | Update the task backbone and head |
| Partial / progressive fine-tuning | Train selected upper layers or gradually release frozen layers |
| Adapters | Insert trainable bottleneck residual modules |
| LoRA | Add a low-rank update to frozen linear weights |
| LORMA | Apply a multiplicative low-rank update |
| QLoRA-style adaptation | Locally implemented NF4 quantization with trainable low-rank updates |
| Prompt / prefix tuning | Learn input embeddings or attention key/value prefixes |
| IA3 / BitFit | Train scaling vectors or eligible bias parameters |
| Knowledge distillation | Train a reduced-depth student using task supervision and teacher outputs |

For output-side LORMA:

$$W'=(I+sBA)W_0,\qquad s=\alpha/r.$$

Earlier notebooks may select input- or output-side multiplication through `side="auto"`. Phase 7 fixes output-side multiplication with zero low-rank dropout. The NF4 implementation is QLoRA-style rather than a reproduction of the complete QLoRA training stack.

Preset names are local to each notebook: for example, `M2_partial` in Phase 5 and `M2_lora` in Phase 6 are different configurations. A listed preset describes available code, not necessarily a completed experiment.

## Phase 1: Foundation model pretraining

**Goal:** learn legal language representations from the competition corpus and generate legal text continuations.

**Notebooks:** [Decoder](phase-1/cs781-capstone-phase1-attempt1.ipynb) · [Encoder–decoder](phase-1/phase-1-encdec-attempt-1.ipynb) · [Hybrid](phase-1/capstone-phase1-mamba-2-attention-hybrid.ipynb)

### Approach

1. Sample documents from Acts, High Courts, Supreme Court judgments, and Tribunals.
2. Normalize punctuation and whitespace, filter unsuitable records, remove duplicates, and cap document length.
3. Reserve a deterministic hash-based validation split and train a custom `ByteBPE` tokenizer on sampled text.
4. Pack tokenized documents into fixed-length blocks for efficient training.
5. Train the selected architecture, retain checkpoints, and compare decoding settings on held-out legal prompts.

The decoder and hybrid optimize next-token cross-entropy. The encoder–decoder combines **prefix continuation and span corruption in a 75:25 ratio**, with corruption density 0.15 and mean span length 3. Its decoder start token reuses EOS.

Training uses AdamW, FP16 autocast, gradient scaling, warmup, and learning-rate decay. A microbatch of four blocks with accumulation over 16 steps allows training within T4 memory. Peak learning rates are `6e-4` for decoder/hybrid and `5e-4` for encoder–decoder.

The generation pipeline supports greedy or sampled decoding, repetition penalties, and stopping constraints. It converts continuations into the required ID-keyed JSON format.

**Saved assets:** model checkpoints, architecture configuration, `bpe_merges.json`, training history, `prediction.json`, and `submission.zip`.

**Evaluation dependency:** Phase-1 encoder–decoder and hybrid exports use pretrained RoBERTa-large through BERTScore for local evaluation only. Task-model weights are trained from scratch; the evaluation dependency is separate.

## Phase 2: Legal named entity recognition

**Goal:** identify legal entities as typed character spans in the original document.

**Notebook:** [Hybrid NER](phase-2/capstone-phase-2-mamba.ipynb)

### Approach

The hybrid backbone generates contextual token representations. A **BiGRU and BIO classification head** predict 25 tags corresponding to 12 entity types plus the outside label.

Long documents are processed through overlapping 1,024-token windows. Inference combines overlapping emissions before constrained decoding. BIO constraints help prevent invalid tag sequences, and decoded token spans are mapped back to raw character offsets.

Offset handling is central to this phase. Normalized text retains an alignment map to the original document, and predictions are checked against the raw substring. Endpoint conventions are explicitly controlled rather than assuming inclusive and exclusive offsets are interchangeable.

Model selection uses one labelled fold for training and the other for validation. The selected configuration is then retrained on both folds before test prediction.

**Presets:** `M1_full`, `M2_lora`, `M3_adapter`. The saved notebook selects `M3_adapter`, using a bottleneck size of 64.

**Output:** a document-ID mapping containing entity spans and labels; optional entity text follows the configured serializer.

## Phase 3: Rhetorical role prediction

**Goal:** assign a rhetorical role to every supplied sentence while using surrounding legal context.

**Notebooks:** [Decoder](phase-3/capstone-phase3-dec%20%281%29.ipynb) · [Hybrid](phase-3/capstone-phase3-mamba%20%281%29.ipynb)

### Approach

Supplied sentence boundaries and ordering are preserved. Each sentence is capped at 96 tokens, and sentences are packed into overlapping 1,024-token document windows.

Token pooling and separator representations produce sentence features. These pass through a **512-dimensional projection, a BiGRU with hidden size 256, domain-aware features, and a sentence classifier**. The contextual head helps distinguish similar language used as facts, arguments, or judicial reasoning.

Training uses the official Corporate Law and IT splits. Class-aware supervision addresses uneven role frequencies. Evaluation keeps the label-averaging convention explicit because averaging over gold-present classes differs from averaging over all configured roles.

| Backbone | Available presets | Selected in export |
|---|---|---|
| Decoder | `D1_lorma`, `D2_ia3`, `D3_kd`, `D4_lora` | `D4_lora` |
| Hybrid | `M1_lora`, `M2_prompt`, `M3_kd`, `M4_lorma` | `M4_lorma` |

KD uses a half-depth student and cached teacher sentence distributions. The teacher label vocabulary and document keys must match the student data.

**Output:** one ordered list of role labels per document, with exactly one prediction for each input sentence.

## Phase 4: Court judgment prediction

**Goal:** predict a binary judgment outcome from a long legal document.

**Notebooks:** [Decoder](phase-4/phase-4-dec-only-implementation.ipynb) · [Encoder](phase-4/phase-4-encoder-decoder-implementation.ipynb) · [Hybrid](phase-4/phase-4-mamba-implementation.ipynb)

### Approach

The loader audits duplicate records and label consistency. Documents are represented by **up to four 1,024-token head/tail windows**, retaining both introductory facts and concluding passages.

Each backbone encodes the selected windows. An attention pooling head combines their representations into a case-level vector, followed by a binary classifier. The encoder–decoder notebook uses only its encoder.

Training uses class weighting, gradient accumulation, and development-based threshold selection. The prediction threshold is tuned for Macro-F1 rather than fixed at 0.5. Context probes compare first, last, uniform, and head/tail selections to examine where useful evidence appears.

| Backbone | Available presets | Selected in export |
|---|---|---|
| Decoder | `D1_lorma`, `D2_bitfit`, `D3_kd`, `D4_unfreeze` | `D4_unfreeze` |
| Encoder | `E1_full`, `E2_unfreeze`, `E3_kd`, `E4_lorma` | `E3_kd` |
| Hybrid | `M1_adapter`, `M2_partial`, `M3_kd`, `M4_lorma` | `M4_lorma` |

Progressive decoder unfreezing releases 0%, 50%, and 100% of backbone layers across three epochs. KD initializes a half-depth student and combines supervised loss with teacher probabilities; cached teacher coverage determines which training cases are available to that student.

**Output:** a JSON list containing original case IDs and integer predictions `0` or `1`.

## Phase 5: Legal statute identification

**Goal:** predict all applicable statute labels for a case.

**Notebook:** [Hybrid statute identification](phase-5/phase-5-mamba-implementation.ipynb)

### Approach

This phase is treated as **multi-label classification**, allowing several statutes per document. The loader cleans duplicate cases, checks split overlap, and constructs the 100-label target space.

The hybrid processes up to four uniformly selected 1,024-token windows. **Label-wise attention** produces a separate representation for each statute, allowing different labels to focus on different evidence within the same case.

Weighted binary cross-entropy handles unequal positive frequencies, with positive weights capped at 20. Prediction uses global or label-specific thresholds; per-label thresholds are shrunk toward a global value when evidence is limited. The configured output policy predicts at least one statute per case.

**Presets:** `M1_adapter`, `M2_partial`, `M3_kd`, `M4_lorma`. The selected `M2_partial` configuration trains the top five hybrid layers, final normalization, and task head, using backbone LR `1e-4` and head LR `2e-3`.

A throughput benchmark fits the number of training cases per epoch to the configured GPU budget.

**Output:** a JSON list mapping each original case ID to a list of unique integer statute IDs.

## Phase 6: Abstractive summarization

**Goal:** generate a concise legal summary while respecting the backbone's context limit.

**Notebooks:** [Decoder](phase-6/phase6-decoder-implementation.ipynb) · [Encoder–decoder](phase-6/phase6-encdec-implementation.ipynb) · [Hybrid](phase-6/capstone-phase6-mamba-implementation.ipynb)

### Approach

The loader validates document–summary pairs, audits and skips unusable training summaries, and creates a grouped, deduplicated holdout. Selected document regions are packed into a **960-token source budget**.

| Backbone | Summarization formulation |
|---|---|
| Decoder | Prompt-conditioned continuation using the document and a summary cue; loss is restricted to summary target tokens |
| Encoder–decoder | Full encoder, decoder, and cross-attention with separate source and target masks |
| Hybrid | Causal continuation with SSD recurrent state updates and attention KV caching |

Generation reuses cached state to avoid recomputing the complete prefix at every step. The active configuration uses greedy decoding, repetition penalty `1.1`, no-repeat trigrams, and a 64–256-token output range. EOS is banned before the minimum length.

The three exports select **decoder prompt tuning (`D4_prompt`)**, **encoder–decoder prefix tuning (`E2_prefix`)**, and **hybrid LORMA (`M4_lorma`)**. Their preset menus also provide alternative full fine-tuning, low-rank adaptation, adapters, quantized adaptation, and KD configurations where supported.

Training and generation are benchmarked separately so the budget reserves time for validation and test summaries. Local checkpoint selection uses ROUGE-L on the configured development subset.

**Output:** `[{"id": "case_id", "summary": "Generated summary ..."}]`.

## Phase 7: Prior-case retrieval

**Goal:** rank prior cases by relevance and return exactly seven candidates for each query.

**Notebooks:** [Decoder](phase-7/capstone-phase-7-dec-implementation.ipynb) · [Encoder](phase-7/capstone-phase-7-encdec-implementation.ipynb) · [Hybrid](phase-7/capstone-phase-7-mamba-implementation.ipynb)

### Approach

Query and candidate text is supplied as lists of sentences. After flattening and tokenization, cases are divided into 1,024-token windows. The selected decoder and hybrid configurations use up to three windows; the encoder uses four.

A **shared bi-encoder** processes queries and candidates independently. Valid token states and chunk representations are mean-pooled, projected into 256 dimensions, and L2-normalized. The encoder–decoder implementation uses only its encoder.

Similarity is the dot product of normalized embeddings, equivalent to cosine similarity:

$$S(Q,C)=E(Q)^\top E(C).$$

Training uses **multi-positive InfoNCE** with temperature `0.07`, in-batch candidates, a hard negative, and a random negative per query. A positive mask prevents a known relevant candidate from being incorrectly treated as a negative.

Initial hard negatives come from BM25 over the training pool. Later refreshes use the current dense encoder. Development and test relevance labels are not used for mining. Gradient replay and block checkpointing reduce activation memory.

The selected presets are **decoder NF4/LoRA (`D2_qlora`)**, **encoder prefix tuning (`E2_prefix`)**, and **hybrid LORMA (`M4_lorma`)**. Additional presets cover full/partial fine-tuning, IA3, prompt tuning, adapters, and KD as supported by each backbone. KD teaches a half-depth student using a task-trained full-depth retrieval teacher.

### Indexing and ranking

Candidate embeddings are computed once per indexing pass and reused across queries. Exact dense scoring ranks the complete candidate pool, with deterministic ID tie-breaking. Evaluation uses Micro-Precision@7, Micro-Recall@7, and Micro-F1@7 over the complete development pool.

**Output:** a query-ID mapping to exactly seven unique candidate IDs, ordered by relevance. Candidate IDs must belong to the test candidate pool.

## Running the notebooks

1. Import the required notebook into Kaggle and enable a GPU accelerator.
2. Run its setup cells and provide access to the official dataset. Use Kaggle Secrets for authentication where needed.
3. For downstream tasks, attach the correct Phase-1 checkpoint and `bpe_merges.json`, then update `CKPT_PATH` and `TOK_PATH`.
4. Select an available `EXPERIMENT` and review its batch, context, chunk, and budget settings.
5. Restart the kernel and run cells in order. Earlier notebooks execute sequentially; Phases 6 and 7 expose a pipeline entry point.
6. Validate predictions and download the final submission archive from the printed output path.

```python
# Example: select a Phase-7 hybrid preset in its configuration cell.
EXPERIMENT = "M4_lorma"
RUN_PIPELINE = True

# After the definition and helper cells have run:
if RUN_PIPELINE:
    RESULTS = main()
```

### Implementation notes

- **Precision:** use FP16 autocast and gradient scaling on T4; do not switch these notebooks to BF16.
- **Tokenizer pairing:** matching vocabulary size is insufficient. Preserve the exact merges used for the checkpoint and verify asset hashes. Mounted checkpoint/tokenizer provenance needs checking before exact reproduction.
- **Long documents:** multiple windows are separate model inputs, not an extension of the backbone's maximum context.
- **Time budgets:** reserve time for validation, teacher work, candidate indexing, and generation. Rebenchmark when trainable layers change.
- **KD dependencies:** earlier KD presets require a compatible `TEACHER_PKL`. Retrieval KD requires a task-trained teacher including its projection/pooling state, or the train-then-cache workflow.
- **Caches:** invalidate token or embedding caches when their tokenizer, preprocessing, chunk policy, or model dependencies change.
- **Execution records:** the Phase-5 export stops after benchmarking; Phase-3 hybrid refitting and Phase-6 decoder/hybrid runs also have partial saved logs. Run through the final output cell to produce a fresh archive.

Dependencies include PyTorch, NumPy, `huggingface_hub`, and notebook utilities; Phase-1 BERTScore evaluation adds its optional metric dependencies. Follow each notebook's setup cells rather than assuming a pinned environment is supplied.

`README.md` and the folders `phase-1/` through `phase-7/` are located directly at the repository root. All notebook links use paths relative to this README.

## Outputs and validation

| Phase | `prediction.json` structure | Essential check |
|---|---|---|
| 1 | Document ID → continuation string | Required IDs and generation format |
| 2 | Document ID → typed span lists | Raw offsets, endpoint convention, valid entity types |
| 3 | Document ID → ordered role labels | Exactly one label per supplied sentence |
| 4 | List of `{id, pred}` objects | Integer prediction in `{0, 1}` |
| 5 | List of `{id, pred}` objects | Unique valid statute IDs in each prediction list |
| 6 | List of `{id, summary}` objects | Complete test coverage and valid summary strings |
| 7 | Query ID → ranked candidate IDs | Exactly seven unique IDs from the correct pool |

Every archive places **`prediction.json` directly at its root**. Phase 1 uses `submission.zip`; downstream notebooks use `submission_<EXPERIMENT>.zip`.

Preserve the selected checkpoint, tokenizer, effective configuration, thresholds or decoding settings, and generated archive together. Output directories are printed by the notebooks, with later phases using backbone/preset/run-specific paths under `/kaggle/working/`.

## References

- [Transformer — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [T5 — Exploring the Limits of Transfer Learning](https://jmlr.org/papers/v21/20-074.html)
- [Mamba-2 — Transformers Are SSMs](https://proceedings.mlr.press/v235/dao24a.html)
- [LoRA — Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LoRMA — Low-Rank Multiplicative Adaptation for LLMs](https://doi.org/10.18653/v1/2025.findings-acl.527)
- [Adapters — Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html)
- [QLoRA — Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Prefix-Tuning](https://doi.org/10.18653/v1/2021.acl-long.353) and [Prompt Tuning](https://doi.org/10.18653/v1/2021.emnlp-main.243)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Contrastive Predictive Coding / InfoNCE](https://arxiv.org/abs/1807.03748)
- [BERTScore](https://arxiv.org/abs/1904.09675) and [ROUGE](https://aclanthology.org/W04-1013/)
