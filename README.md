# Overview

This repository is an implementation of the End-to-End Spoof-Aggregated Spoofing-Aware Speaker Verification System.
The model performance is tested on the ASVSpoof 2019 Dataset.

## Setup 

### Environment

<details><summary>Show details</summary>
<p>

* speechbrain==0.5.7
* pandas
* torch==1.9.1
* torchaudio==0.9.1
* nnAudio==0.2.6
* ptflops==0.6.6

</p>
</details>

-   Create a conda environment with  `conda env create -f environment.yml`.
-   Activate the conda environment with  `conda activate `.


``

### Data preprocessing
    .
    ├── data                       
    │   │
    │   ├── PA                  
    │   │   └── ...
    │   └── LA           
    │       ├── ASVspoof2019_LA_asv_protocols
    │       ├── ASVspoof2019_LA_asv_scores
    │       ├── ASVspoof2019_LA_cm_protocols
    │       ├── ASVspoof2019_LA_train
    │       ├── ASVspoof2019_LA_dev
    │       
    │
    └── ARawNet

1. Download dataset. Our experiment is trained on the Logical access (LA) scenario of the ASVspoof 2019 dataset. Dataset can be downloaded [here](https://datashare.is.ed.ac.uk/handle/10283/3336).
2. Unzip and save the data to a folder  `data`  in the same directory as  `ARawNet` as shown in below.

    
3. Run ``python preprocess_SASASV.py``  Or you can use our processed data directly under "/processed_data".

### Train 

`python train_SASASV.py yaml/SASASV.yaml --data_parallel_backend -data_parallel_count=2`

### Evaluate
  `python eval_SASASV.py`






## Cite Our Paper

If you use this repository, please consider citing:

@article{teng2022sa,
title={SA-SASV: An End-to-End Spoof-Aggregated Spoofing-Aware Speaker Verification System},
author={Teng, Zhongwei and Fu, Quchen and White, Jules and Powell, Maria E and Schmidt, Douglas C},
journal={arXiv preprint arXiv:2203.06517},
year={2022}
}
