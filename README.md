# Tracing Semantic Change in Historical Corpora: A Reproducible Word-Embedding Workflow for Intellectual History

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdh-observer/bEaZZLcieZcS/main?filepath=article.ipynb)



This article demonstrates a reproducible workflow for measuring semantic change in historical corpora using transformer-based language models, with a demonstration on 160,000 documents from the American Founders' writings. Two findings emerge from the demonstration. *Republican* moved downward: from a theoretical framework holding the founding vocabulary together toward a credential of partisan belonging. *Liberty* moved outward: from a single ideological tradition toward multiple competing ones simultaneously — republican virtue, liberal economics, and antislavery argument all reaching for the same word. Together they describe the disaggregation of the founding ideological vocabulary, now measurable at corpus scale.

The workflow departs from standard diachronic practice by training a language model directly on the historical corpus rather than adapting a modern pretrained model, ensuring the meanings it encodes are period-native. Results are cross-validated across three instruments built on entirely different mathematical principles; a finding is treated as evidential only where independent methods agree. The republican finding reframes the Appleby–Pocock–Wood debate: both sides were tracking real changes in different registers — the word's semantic content was not displaced, but its social function was transformed. Every step is documented in open, runnable code; the trained model is released on Hugging Face.
