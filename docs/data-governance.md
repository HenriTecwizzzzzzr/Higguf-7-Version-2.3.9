# Higguf#7 training-data rules

Higguf#7 is trained from random weights, but that does not permit copying data
without a right to use it. Every corpus contribution must appear in
`data/corpus-manifest.jsonl` before tokenisation.

Each line records the source, licence or public-domain basis, retrieval date,
hash, language, and the person who verified the rights. Do not add private
messages, passwords, personal data, pirated books, paywalled articles, or data
whose terms prohibit model training.

The first executable milestone uses only synthetic tokens. That validates the
software but is not language training and must not be represented as a trained
assistant.

## Required review before a real run

1. Verify the source permits this use and record the precise licence.
2. Remove duplicate documents and personally identifying data.
3. Write the metadata row and calculate the source file's SHA-256 hash.
4. Obtain a reproducible token count after training the tokenizer exclusively
   on the approved corpus.
5. Keep held-out evaluation data separate from the training corpus.
