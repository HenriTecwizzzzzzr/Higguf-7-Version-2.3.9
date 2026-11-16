# Higguf#7 capability roadmap

## Current capability

Higguf#7 Seed is a text-only, random-initialized language-model pilot. It is
being trained on approved WikiText data. It cannot yet generate useful images
or deliver finished files.

## Goal: text and file generation

The 10B text model will be developed to create useful text-based files from a
user's instructions, including Markdown, plain text, JSON, CSV, HTML, CSS,
JavaScript, Python, and configuration files. A file is only considered
generated after its content is validated for the requested format and saved to
the user's chosen location.

This needs a later instruction-tuning stage using lawfully sourced examples of
requests paired with high-quality file contents, plus format-specific tests.

## Goal: image generation

Image generation is a separate multimodal capability, not a feature that can
be obtained by continuing text-only language-model pretraining. The intended
first scope is prompt-to-image generation of common raster formats such as PNG,
JPEG, and WebP.

It requires its own lawfully sourced, documented image-and-caption dataset, an
image tokenizer or latent encoder/decoder, a generative image model, safety
evaluation, and substantially more GPU memory and compute than the current
free T4 x2 Seed run provides.

## Order of work

1. Continue validating the from-scratch text Seed pilot.
2. Train and evaluate the 4B and 6B milestones, then the 10B text model, on
   documented text data.
3. Add supervised tuning and validation for text/file creation.
4. Design a separate image-generation pilot and approve its dataset before
   training it.

No pretrained model or inference API is substituted for these stages.

## Future mixture-of-experts research

Five unrelated 4B models cannot simply be joined together to create a trained
20B model. Their parameters, tokenizers, representations, and learned skills
will generally be incompatible. Higguf#7 instead has an experimental
from-scratch mixture-of-experts component in `higguf/moe.py`. It provides a
learned top-k router and compatible SwiGLU experts for future experiments.

This is not part of the released 2B dense pilot, is not trained, and does not
make image, file, or video generation appear automatically. A practical MoE
run would require a shared tokenizer and architecture, router training,
distributed expert placement, evaluation of routing quality, and separate
safety work for every additional modality.
