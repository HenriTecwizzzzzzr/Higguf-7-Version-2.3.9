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

## 75-second narration

Hi, this is Higguf number seven. We are building it from random initialization,
not from a pretrained model or an API. The current milestone is an exact
2.064-billion-parameter decoder-only text model.

We used two free NVIDIA T4 GPUs on Kaggle with fully sharded training and CPU
offload. That let us verify the architecture and complete six small pilot
updates on an approved WikiText-2 stream. The loss moved from 1572.75 on the
first update to 1189.44 on update six.

That does not mean Higguf#7 is fully trained or ready to use. Six tiny pilot
updates are evidence that the distributed training path works, not evidence of
a useful model. There is no public checkpoint in this release, and image and
file generation are future separate milestones.

The next goal is careful, lawful corpus preparation, much more training,
evaluation, and a reproducible model-card release. The long-term architecture
target is 18B parameters. Follow the source repository for the honest progress
log.

## On-screen text plan

1. `Higguf#7 — From Scratch` (0–5 s)
2. `2,063,854,080 parameters` (5–15 s)
3. `Kaggle T4 × 2 • FSDP + CPU offload` (15–30 s)
4. `Six pilot updates • loss 1572.75 → 1189.44` (30–45 s)
5. `Pilot only — not fully trained` (45–60 s)
6. `End plan: public 18B text AI + separate image, file & video tools` (60–75 s)

Use this final narration sentence:

> The end plan is a public 18B Higguf#7 text model and separate image, file,
> and video-generation tools—but only after each has been trained, evaluated,
> and is ready for a responsible public release.

## Accessibility asset

Upload [`assets/youtube/higguf7-2b-pilot.en.srt`](../assets/youtube/higguf7-2b-pilot.en.srt)
with the finished video as the English subtitle track. The timing matches the
75-second narration above.
