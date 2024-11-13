# DecLimSup: Decoding with Limited Teacher Supervision Requires Understanding When to Trust the Teacher

[![Paper](https://img.shields.io/badge/Paper-arxiv.2406.18002-red)](https://arxiv.org/abs/2406.18002)
[![Conference](https://img.shields.io/badge/EMNLP-2024-orange)](##Citation)

This repository contains the code of our work.
Refactoring code and code for HPU will be released soon.

You can run the code by referring *script.sh* file 

## Coming Soon
  - [ ] Code refactoring and details for easy to use
  - [ ] Release our code for HPU

## Introduction
DecLimSup is our work that **empirically analyzes contrastive decoding settings in a limited supervision scenario of teacher LLM**. We find that it is essential to adaptively overtrust or disregard the LLM prediction based on the confidence of the small-scale LLM. Our experiments on a wide range of models and datasets demonstrate that our method consistently improves over conventional decoding strategies.


## Citation
If you use this code, please cite the following paper:
```
@inproceedings{ok-etal-2024-decoding,
    title = "Decoding with Limited Teacher Supervision Requires Understanding When to Trust the Teacher",
    author = "Ok, Hyunjong  and
      Ryu, Jegwang  and
      Lee, Jaeho",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.693",
    pages = "12460--12476",
}
