---
title: Pet Breed Identification and Product Recommender
emoji: 🐾
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Pet Breed Identification and Product Recommender

A multimodal pet-care assistant with three modules:

- **Breed identification** — a Swin Transformer (Swin-Tiny) with three heads
  covering 151 dog and cat breeds, 118 fish species and 200 bird species.
- **Food quality check** — a reference lookup backed by a fine-tuned BERT
  classifier that labels ingredients Safe, Moderate or Unsafe.
- **Product recommendation** — Sentence-BERT embeddings ranked by cosine
  similarity against a product catalogue.

Guidance only. Consult a veterinarian for medical or dietary concerns.
