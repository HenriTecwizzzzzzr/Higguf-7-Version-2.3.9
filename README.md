# Higguf#7

Higguf#7 is a decoder-only language model designed to be trained **from random
initialisation**. It has no pretrained base model and makes no inference API
calls. Its longer-term goals include text/file generation and a separately
trained image-generation capability.

The current practical milestone is the exact 2,063,854,080-parameter text-model
pilot. The long-term architecture target is 18B parameters. Neither is a
finished general-purpose model yet: the 2B run has completed only six pilot
updates on WikiText-2, and the 18B configuration has not been launched.

## Community help and future contributor beta

We welcome reproducible training ideas, useful software or code, experiments,
compute guidance, and testing help for Higguf#7. People who make a useful
contribution can be considered for a future beta when there is a genuinely
usable release. See [the community request issue](https://github.com/HenriTecwizzzzzzr/Higguf-7-Version-2.3.9/issues/1) to contribute or register interest. There is no public beta or trained 4B checkpoint yet, and beta access is not guaranteed.

## Project stages

1. Build a lawful, documented training corpus.
2. Train Higguf#7's tokenizer using only that corpus.
3. Validate the data and run the tiny-model smoke test.
4. Launch and evaluate the 2B distributed pilot from random weights.
5. Establish useful-scale 2B training and evaluation, then validate scale-up
   milestones toward the 18B architecture.
6. Add validated text/file generation, then develop image generation as a
   separate multimodal project.

## Local setup

```powershell
py -m venv .venv
py -m pip install -e ".[dev]"
python scripts/create_synthetic_smoke_data.py
py -m higguf.train --config configs/smoke-test.yaml
```

If PowerShell blocks virtual-environment activation, you do not need to change
Windows security settings: use `py -m pip` and `py -m ...` as above. To use the
project virtual environment specifically, replace `py` with
`.\.venv\Scripts\python.exe`.

## Terminal research interface

The repository includes a terminal sampler for **compatible checkpoints created
by this project's trainer**. It deliberately requires an acknowledgement flag,
because an interface does not turn an early checkpoint into a trained chatbot.
The local one-update Seed checkpoint can be inspected with:

```powershell
py -m pip install -e ".[data]"
py -m higguf.chat --allow-untrained `
  --checkpoint checkpoints/higguf-7-seed-107m-cpu-pilot/latest.pt `
  --config configs/higguf-seed-107m-cpu-pilot.yaml `
  --tokenizer artifacts/tokenizer-seed-8192/tokenizer.json
```

This command may be slow on CPU and its output is not expected to be useful:
the supplied local checkpoint has only one real training update. The separately
saved Kaggle Seed checkpoint uses a different pilot architecture, so it cannot
be loaded by this command without a dedicated compatible loader.

## Next scale target: 1.5B

`configs/higguf-1.5b.yaml` defines a fresh **1,514,181,888-parameter** Higguf#7
model. It cannot reuse the 94.6M Seed weights: widening or adding layers creates
new, untrained parameters. Its minimum AdamW state is about 22.8 GiB in total
before activations, so it requires real parameter sharding (FSDP or DeepSpeed
ZeRO-3) plus CPU offload across both GPUs, activation checkpointing, and short
checkpointed sessions. The existing single-process trainer must not be used for it.


This will run only after you supply tokenized data. The smoke-test config is
intentionally small; it is a code-path validation, not a reduced-quality
version of the model.

## 4B target feasibility

`configs/higguf-4b.yaml` defines a fresh **3,960,655,872-parameter** target.
The configuration is valid and has been checked without allocating its weights.
Its BF16 weights alone are about 7.38 GiB; a conventional AdamW training state
is about 59.02 GiB before activations. It is therefore not trainable by the
single-process trainer or this PC, and it is not correct to call it a trained
4B model. A future 4B run needs FSDP/ZeRO-3 sharding, CPU/NVMe offload,
activation checkpointing, a lawful tokenized corpus, and checkpoint recovery.
The repository now includes an FSDP research launcher for this class of run;
it materialises target weights per shard and writes per-rank sharded recovery
files. A November 2026 exact-architecture, one-update FSDP probe on Kaggle T4
x2 exhausted the notebook memory and restarted the kernel while directly
materialising the full model before sharding. A second run with the
repository's corrected non-recursive meta-device FSDP launcher also restarted
from memory pressure before completing its single update. A third run removed
the full-model initialisation broadcast and used the smallest configured
eight-token micro-batch; it also restarted before one update. These are failed
implementation probes, not completed 4B runs; there is no 4B checkpoint.

```powershell
python scripts/estimate_resources.py --config configs/higguf-4b.yaml
# On a compatible multi-GPU CUDA host only:
torchrun --standalone --nproc_per_node=2 -m higguf.train_fsdp `
  --config configs/higguf-4b.yaml --steps 1 --cpu-offload
# Resume only from matching shards produced by that launcher:
torchrun --standalone --nproc_per_node=2 -m higguf.train_fsdp `
  --config configs/higguf-4b.yaml --steps 1 --cpu-offload `
  --resume-dir checkpoints/higguf-7-4b --resume-step 1
```

`configs/higguf-4b-pilot.yaml` keeps the same 3.96B parameter architecture,
but uses an 8-token context and one update solely to test a free-GPU FSDP
launch. A successful pilot would prove only that a single update and sharded
checkpoint can run; it would not be a trained 4B model.

## Long-term 18B architecture

| Component | Value |
| --- | ---: |
| Parameters | ~18B target configuration |
| Layers | 48 |
| Hidden size | 6,144 |
| Attention heads | 48 |
| KV heads (GQA) | 8 |
| MLP intermediate size | 15,040 |
| Context length | 4,096 |
| Vocabulary | 65,536 |

The target needs an initial training pilot on a distributed CUDA environment.
It will not fit or train on this PC's integrated GPU. Its 32 GB of system RAM
is still enough for local data preparation and tiny-model validation.

The exact configuration contains 17,937,537,024 parameters: BF16 weights need
about 33.41 GiB and a conventional AdamW state needs about 267.29 GiB before
activations. These are planning estimates, not evidence of an 18B run.

The free Kaggle T4 x2 session can perform tiny FSDP pilot updates for the 2B
configuration, but it cannot fully pretrain either the 2B model or the 18B
target. The 18B target requires a carefully configured multi-node sharded run,
rather than a normal single-GPU or DataParallel launch.

Check a configuration without allocating its weights:

```powershell
python scripts/estimate_resources.py --config configs/higguf-18b.yaml
```

Before starting a training job, run the setup-only preflight. It checks that
the configuration is coherent and that the token file exists and is large
enough. `--require-cuda` also checks the requested number of GPUs without
allocating model weights:

```powershell
python scripts/preflight_training.py --config configs/smoke-test.yaml
python scripts/preflight_training.py --config configs/higguf-2b.yaml --require-cuda --minimum-gpus 2
```

## First training milestone: Higguf#7 Seed

`configs/higguf-seed-107m.yaml` is a roughly 107M-parameter model initialized
from scratch. It is the first runnable milestone and deliberately uses the
same RMSNorm, RoPE, GQA, SwiGLU, and causal-loss code as the 18B configuration.
It is not an 18B substitute and should never be presented as Higguf#7 18B.

The supplied synthetic-data generator validates the pipeline only; it does not
produce meaningful language training. A real run must start from a documented,
lawfully sourced corpus.

## Future image and file generation

Text/file generation and image generation are goals, not current claims. File
generation requires later instruction tuning and output validation. Image
generation needs a separate lawfully sourced image-and-caption dataset and a
separate multimodal architecture. See
[the capability roadmap](docs/capability-roadmap.md) for the exact plan.

## Public end plan

The intended public end plan is an 18B Higguf#7 text model, plus separately
trained image, file, and video-generation tools. None of those future tools is
available publicly yet. Each must first have lawful data, reproducible training,
evaluation, safety testing, and a release review.

The observed Seed-pilot results are tracked in
[the training log](docs/training-log.md).

## Release status

Before publishing, read [the release status](docs/release-status.md). It lists
what can truthfully be released now (source, configurations, and pilot logs)
and what must not be claimed (a complete 2B checkpoint, useful image generation,
or a trained 18B model).

The accompanying [2B pilot model card](docs/model-card-2b-pilot.md) describes
the architecture, verified run conditions, data provenance, and limitations.

## Prepare a corpus

Place UTF-8 `.txt` files that you have the right to use beneath `data/raw/`,
then run:

```powershell
python scripts/train_tokenizer.py --input data/raw --output artifacts/tokenizer
python scripts/tokenize_corpus.py --input data/raw --tokenizer artifacts/tokenizer --output data/tokenized
```

No corpus is bundled with this repository. Before either command, record and
validate its provenance as described in [the data-governance rules](docs/data-governance.md):

```powershell
python scripts/validate_corpus_manifest.py
```

### WikiText pilot corpus

For the Seed model, the repository can download WikiText-103, a Wikipedia-derived
English pilot corpus. Its dataset card identifies the licence as CC BY-SA 3.0.
That licence requires attribution and share-alike compliance, so the downloader
requires the reviewer who approves those terms to be named explicitly:

```powershell
pip install -e ".[data]"
python scripts/download_wikitext_pilot.py --approved-by "Your name"
python scripts/validate_corpus_manifest.py
```

