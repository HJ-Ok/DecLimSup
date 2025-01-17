# DecLimSup: Decoding with Limited Teacher Supervision Requires Understanding When to Trust the Teacher

[![Paper](https://img.shields.io/badge/Paper-arxiv.2406.18002-red)](https://arxiv.org/abs/2406.18002)
[![Conference](https://img.shields.io/badge/EMNLP-2024-orange)](https://aclanthology.org/2024.emnlp-main.693/)


You can run the code by referring to the *script.sh* file  
Also our code supports **Gaudi HPU**s. You can enable it by adding the `--use_hpu`.

## How to Run the Code

You can run the code using the following command:

```bash
python inference.py
```
### Arguments

Below is a description of the key arguments:

#### `--benchmark` (str)
- **Description**: Specifies the benchmark to evaluate the model on.  
- **Supported Benchmarks**:  
    - `gsm8k`, `strategyqa`, `multiarith`, `math`, `arc_c`, `arc_e`, `svamp`  

#### `--N` (int)
- **Description**: Determines how many tokens will receive knowledge from the teacher model.  

#### `--multi_exec` (bool)
- **Description**: Enables running multiple alpha values in a single execution. 

  - ##### `--alpha_start` The starting value of alpha.

  - ##### `--alpha_end` The ending value of alpha.

  - ##### `--alpha_step` The step size for iterating over alpha values.


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
