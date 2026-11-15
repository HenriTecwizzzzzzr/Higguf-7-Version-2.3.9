# Higguf#7 release status

Last verified: 2026-11-16 (Australia/Brisbane)

## Suitable for a source release now

- From-scratch model and training source code.
- Configurations for the Seed, 1.5B, 2B, 4B, 6B, 10B, and 18B design targets.
- Data-governance documents and corpus-manifest validation tools.
- The 2B pilot training log: an exact 2,063,854,080-parameter architecture,
  six successful FSDP pilot updates on approved WikiText-2, and resumable
  sharded checkpoints retained in the Kaggle working session.

## Not suitable to claim or publish as a model release

- A fully pretrained or generally useful 2B Higguf#7 model.
- A downloadable 2B checkpoint: the retained shards are temporary Kaggle
  pilot outputs and have not been independently exported, integrity-checked,
  evaluated, or licensed for public redistribution.
- File generation, image generation, or a trained 18B model.

## Publication checklist

1. Create an empty GitHub repository under the owner's account and add it as
   this repository's `origin` remote.
2. Review the tree to ensure no checkpoints, user data, credentials, or private
   Kaggle outputs are tracked.
3. Commit the source release and push it to GitHub.
4. Publish the first video only with the truthful title and description in
   `docs/youtube-video-01.md`.
5. Do not attach model weights until they have a reproducible export,
   evaluation report, model card, and confirmed redistribution rights.
