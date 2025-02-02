# ACD
Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts (Findings of EMNLP 2024)

## Prepare
**Prepare datasets and contexts you want to use.**
- Datasets (nq, triviaqa)
- Retriever & Retrieved contexts
- Directory paths (<code>dir</code>) in <code>configs/{task}.yaml</code>

**In this paper, we use ...**
- Datasets: NaturalQuestions (NQ) and TriviaQA data downloaded from [FiD](https://github.com/facebookresearch/fid) (others from huggingface)
- Context: wiki dump from [DPR](https://github.com/facebookresearch/DPR)
- Retriever: [Contriever-msmarco](https://github.com/facebookresearch/contriever) as a retriever

## Run
Run <code>sh scripts/run.sh {gpu number} {task}</code>
- <code>{gpu number}</code>: e.g. <code>'0'</code>
- <code>{task}</code>: <code>'acd'</code> or <code>'naive'</code> (greedy decoding)

## Reference
```bib
@inproceedings{kim-etal-2024-adaptive,
    title = "Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts",
    author = "Kim, Youna  and
      Kim, Hyuhng Joon  and
      Park, Cheonbok  and
      Park, Choonghyun  and
      Cho, Hyunsoo  and
      Kim, Junyeob  and
      Yoo, Kang Min  and
      Lee, Sang-goo  and
      Kim, Taeuk",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.136/",
    doi = "10.18653/v1/2024.findings-emnlp.136",
    pages = "2421--2431",
    abstract = "When using large language models (LLMs) in knowledge-intensive tasks, such as open-domain question answering, external context can bridge the gap between external knowledge and the LLMs' parametric knowledge.Recent research has been developed to amplify contextual knowledge over the parametric knowledge of LLMs with contrastive decoding approaches.While these approaches could yield truthful responses when relevant context is provided, they are prone to vulnerabilities when faced with noisy contexts.We extend the scope of previous studies to encompass noisy contexts and propose adaptive contrastive decoding (ACD) to leverage contextual influence effectively.ACD demonstrates improvements in open-domain question answering tasks compared to baselines, especially in robustness by remaining undistracted by noisy contexts in retrieval-augmented generation."
}
```
