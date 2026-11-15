# Higguf#7 2B pilot model card

## Identifier and status

**Higguf#7 2B pilot** is an experimental decoder-only text-model architecture
with exactly **2,063,854,080 parameters** when embeddings are tied. It is
initialized from random weights and is not based on a pretrained model.

This is a training-pipeline pilot, not a released pretrained model. As of the
latest recorded run, it has completed six small updates, so it must not be
represented as capable of reliable chat, coding, factual question answering,
file generation, image generation, or general language understanding.

## Architecture

| Component | Value |
| --- | ---: |
| Layers | 24 |
| Hidden size | 2,560 |
| Attention heads | 20 |
| KV heads (GQA) | 4 |
| MLP intermediate size | 8,704 |
| Vocabulary | 32,000 |
| Maximum sequence length | 2,048 |

The implementation uses a causal decoder, RMSNorm, RoPE, grouped-query
attention, SwiGLU, and tied input/output embeddings. The design configuration
is [`configs/higguf-2b.yaml`](../configs/higguf-2b.yaml).

## Pilot run

| Item | Observed value |
| --- | --- |
| Hardware | Two NVIDIA T4 GPUs in a free Kaggle session |
| Distributed method | FSDP full sharding with CPU offload |
| Precision | FP16 initialization |
| Optimizer | Adafactor |
| Dataset stream | Approved WikiText-2 text pilot |
| Successful updates | 6 |
| First / latest logged loss | 1572.7500 / 1189.4375 |

The pilot established that this architecture can perform a forward/backward
update and save resumable sharded checkpoints in that environment. It does not
establish model quality or readiness for any practical use.

## Data and licensing

The pilot's corpus provenance is recorded in
[`data/corpus-manifest.jsonl`](../data/corpus-manifest.jsonl). WikiText is
Wikipedia-derived and listed there under CC BY-SA 3.0. No raw corpus or model
weights are included in this source release. Any future model-weight release
needs a separately completed provenance, attribution, share-alike, and
redistribution review.

## Intended future work

The next legitimate milestones are corpus preparation, substantially more
pretraining, evaluation, safety testing, reproducible checkpoint export, and a
full model card. The separate 18B configuration is a long-term architecture
target, not an available model.
