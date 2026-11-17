# Higguf#7 training log

## Seed pilot

| Item | Observed result |
| --- | --- |
| Initialization | Random weights, seed 7; no pretrained checkpoint |
| Model | 94,615,040 parameters |
| Hardware | Kaggle T4 x2 |
| Dataset | Approved WikiText pilot (`wikitext-2-raw-v1`) |
| Tokenizer | Fresh 8,192-token BPE tokenizer |
| Training stream | 2,714,645 tokens |
| Latest saved Kaggle checkpoint | 5,020 updates (Kaggle Version #6) |
| Last logged loss | 4.4879 |

The initial 20-update run logged loss 9.1679 at its first update and 7.3498 at
its final update. Later continuation batches reduced the last logged loss to
5.9899 at 520 total updates, 5.5488 at 1,520 updates, and 4.3785 at 4,020
updates. A further 1,000-update continuation restored the 4,020-update
checkpoint and ended at loss 4.4879 at 5,020 updates. Kaggle Version #6 saved
that continuation successfully.

## Persistence

Kaggle Version #6 was saved successfully with the 5,020-update Seed output.
It is a reproducible research checkpoint, not a general-purpose chatbot or a
2B model.

## 2B pilot

| Item | Observed result |
| --- | --- |
| Model | 2,063,854,080 parameters |
| Hardware | Kaggle T4 x2, FSDP full sharding with CPU offload |
| Optimizer | Adafactor, FP16 initialization |
| Dataset | Approved WikiText-2 stream with a fresh 32,000-token BPE tokenizer |
| Checkpoint format | FSDP `SHARDED_STATE_DICT` (resumable) |
| Completed updates | 6, with each later step restored from the prior checkpoint |
| Losses | 1,572.7500 (step 1), 1,489.1250 (step 2), 1,408.5000 (step 3), 1,331.5000 (step 4), 1,258.7500 (step 5), 1,189.4375 (step 6) |

The first attempt to save step 3 exhausted Kaggle's 19.5 GiB output quota.
The obsolete, non-resumable checkpoint pair, incomplete files, and later the
step-1, step-2, and step-3 shard pairs were removed with user approval. The
resumable step-4, step-5, and step-6 pairs are the authoritative 2B pilot
artifacts. Future runs must retain only the latest two checkpoint pairs.

## Local real-data Seed verification

| Item | Observed result |
| --- | --- |
| Model | 106,980,096-parameter Seed architecture |
| Hardware | Local CPU only |
| Dataset | WikiText-103 raw training split, CC BY-SA 3.0; provenance manifest recorded |
| Tokenizer | Fresh 8,192-token tokenizer trained from the approved local corpus |
| Training stream | 136,340,605 tokens |
| Completed updates | 1 |
| Loss | 9.2922 |

This one-step run verified the real local data, tokenizer, token stream,
forward pass, backward pass, optimizer, and checkpoint path together. Its
checkpoint is a 1.28 GB local artifact and is not a useful released model.
It must not be described as a trained 2B checkpoint or a chat-ready AI.

## Scope note

This is an early text-only Seed pilot, not the 4B, 6B, or 10B target model. It
must not be described as a trained 10B model or as having image/file-generation
capabilities.
