# Higguf#7: honest 2B pilot update

## Suggested title

Building Higguf#7 From Scratch — 2B Pilot Update (Not a Finished AI Yet)

## Thumbnail

Use [`assets/youtube/higguf7-2b-pilot-thumbnail-background.png`](../assets/youtube/higguf7-2b-pilot-thumbnail-background.png)
as the background. Add these text overlays in an editor rather than treating
the generated artwork as a factual label:

- `HIGGUF#7` — upper left, white, bold
- `2B PILOT` — middle left, amber, bold
- `NOT FULLY TRAINED` — lower left, white on a dark translucent strip

The background contains no logos or text. Its SHA-256 is
`73837E5D313CF33740E78A167F76C277DDA35348C106CB852A1DD2C884502385`.

## Description

This video documents the Higguf#7 from-scratch language-model project. The
current experiment uses an exact 2,063,854,080-parameter text-model
architecture. On a free Kaggle session with two NVIDIA T4 GPUs, we completed
six small FSDP pilot updates using an approved WikiText-2 dataset stream.

This is not a full pretraining run, not a released model checkpoint, and not an
image generator or finished file-generation tool. The next work is to build a
lawful larger training corpus, run substantially more training, evaluate it,
and only then consider a model release. The long-term architecture target is
18B parameters.

Source code and the training log: add the GitHub release URL here after the
repository is published.

## Full 90-second narration

Hi, this is Higguf number seven. This is an honest project update, not a claim
that we have finished a new AI.

The current Higguf#7 milestone is an exact 2,063,854,080-parameter,
decoder-only text-model architecture. It starts from random initialisation; it
is not a renamed pretrained model and it is not an API wrapper.

We ran a small technical pilot on two free NVIDIA T4 GPUs in Kaggle. Fully
sharded training with CPU offload let the training setup fit and run. Using the
approved WikiText-2 pilot stream, the experiment completed six small updates.
The logged loss went from 1572.75 on update one to 1189.44 on update six.

Here is the important part: that is not full training. Six tiny pilot updates
only show that the architecture and distributed-training path can run. They do
not make Higguf#7 a useful assistant, and there is no public model checkpoint
in this release.

The public repository contains the source, configuration, training log, and
resource estimates so progress can be checked. The next real work is to obtain
lawful training data, document its provenance, train for vastly more tokens,
run evaluations, and publish a model card before considering any release.

The long-term target is an 18B text model. Image generation, file generation,
and video generation are separate future tools. They each need their own
lawful data, engineering, safety work, testing, and release decision. They are
not features of the current 2B pilot.

Follow the repository for verified progress. Higguf#7 is a research project in
an early pilot stage, and this video is the honest starting point.

## On-screen text plan

1. `Higguf#7 — From Scratch` (0–5 s)
2. `2,063,854,080 parameters` (5–15 s)
3. `Kaggle T4 × 2 • FSDP + CPU offload` (15–30 s)
4. `Six pilot updates • loss 1572.75 → 1189.44` (30–45 s)
5. `Pilot only — not fully trained` (45–60 s)
6. `Pilot only — no public checkpoint` (45–60 s)
7. `Next: lawful data • more training • evaluation` (60–75 s)
8. `End plan: 18B text AI + separate image, file & video tools` (75–90 s)

Use this final narration sentence:

> The end plan is a public 18B Higguf#7 text model and separate image, file,
> and video-generation tools—but only after each has been trained, evaluated,
> and is ready for a responsible public release.

## Accessibility asset

Upload [`assets/youtube/higguf7-2b-pilot.en.srt`](../assets/youtube/higguf7-2b-pilot.en.srt)
with the finished video as the English subtitle track. Update the subtitle
timing before upload because the narration is now 90 seconds.
