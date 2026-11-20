# Research report: Compare retrieval-augmented generation and fine-tuning for adding knowledge to LLMs

## Summary

- Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. [6]
- While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. [7]
- In this study, we compare two common approaches: unsupervised fine-tuning and retrieval-augmented generation (RAG). [11]
- Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs) to retrieve and incorporate new information from external data sources. [1]
- However, existing pipelines for retrieval-augmented code generation (RACG) employ static knowledge bases with a single source, limiting the adaptation capabilities of Large Language Models (LLMs) to domains they have insufficient knowledge of. [4]

## Background

**Retrieval-augmented generation**: Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs) to retrieve and incorporate new information from external data sources. With RAG, LLMs first refer to a specified set of documents, then respond to user queries. [1]

**Knowledge cutoff**: In machine learning, a knowledge cutoff (or data cutoff) is the point in time beyond which a large language model has not been trained on new data. Since large language models are pretrained, any model's knowledge is fixed at what it was trained on before deployment; information about events after this date is absent from the model's training data. [2]

**Fine-tuning (deep learning)**: In deep learning, fine-tuning is the process of adapting a computational model trained for one task (the upstream task) to perform a different, usually more specific, task (the downstream task). It is considered a form of transfer learning, as it reuses knowledge learned from the original training objective. [3]

## Recent research

- **Guardian Crawler: Retrieval-First Knowledge Discovery with Bounded LLM Augmentation for Noisy Web Intelligence** (2026, Joshua Castillo et al.): Retrieving relevant evidence from noisy web data is challenging, particularly in sensitive domains containing incomplete reports, heterogeneous language, and irrelevant content. [8]
- **Expert Mind: A Retrieval-Augmented Architecture for Expert Knowledge Preservation in the Energy Sector** (2026, Diego Ezequiel Cervera): The departure of subject-matter experts from industrial organizations results in the irreversible loss of tacit knowledge that is rarely captured through conventional documentation practices. [5]
- **FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation** (2025, Mohammad Aghajani Asl et al.): While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. [7]
- **Demasking Fine-Tuning for Knowledge Injection in Autoregressive and Diffusion LLMs** (2025, Xu Pan et al.): Large language models (LLMs) are often used in environments where facts evolve, yet factual knowledge updates via fine-tuning on unstructured plain text often suffer from 1) reliance on paraphrasing augmentation, without which training does not generalize to downstream QA tasks, and 2) the reversal curse. [12]
- **Explaining Fine Tuned LLMs via Counterfactuals A Knowledge Graph Driven Framework** (2025, Yucheng Wang et al.): The widespread adoption of Low-Rank Adaptation (LoRA) has enabled large language models (LLMs) to acquire domain-specific knowledge with remarkable efficiency. [13]
- **SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation** (2025, Rafiq Kamel et al.): Large Language Models (LLMs) are increasingly applied to tasks involving structured inputs such as graphs. [10]
- **Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation** (2025, Ali Naseh et al.): Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. [6]
- **Topic Modeling with Fine-tuning LLMs and Bag of Sentences** (2024, Johannes Schneider): Large language models (LLMs) are increasingly used for topic modeling, outperforming classical topic models such as LDA. [9]

## Sources

1. [Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
2. [Knowledge cutoff](https://en.wikipedia.org/wiki/Knowledge_cutoff)
3. [Fine-tuning (deep learning)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning))
4. [EVOR: Evolving Retrieval for Code Generation](https://arxiv.org/abs/2402.12317v2) (2024)
5. [Expert Mind: A Retrieval-Augmented Architecture for Expert Knowledge Preservation in the Energy Sector](https://arxiv.org/abs/2603.14541v1) (2026)
6. [Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation](https://arxiv.org/abs/2502.00306v2) (2025)
7. [FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.22344v1) (2025)
8. [Guardian Crawler: Retrieval-First Knowledge Discovery with Bounded LLM Augmentation for Noisy Web Intelligence](https://arxiv.org/abs/2608.08994v1) (2026)
9. [Topic Modeling with Fine-tuning LLMs and Bag of Sentences](https://arxiv.org/abs/2408.03099v2) (2024)
10. [SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation](https://arxiv.org/abs/2507.13381v2) (2025)
11. [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/abs/2312.05934v3) (2023)
12. [Demasking Fine-Tuning for Knowledge Injection in Autoregressive and Diffusion LLMs](https://arxiv.org/abs/2510.09885v7) (2025)
13. [Explaining Fine Tuned LLMs via Counterfactuals A Knowledge Graph Driven Framework](https://arxiv.org/abs/2509.21241v1) (2025)
