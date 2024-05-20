# CF-KAN

## KAN (Kolmogorov-Arnold Network)-based Collaborative Filtering (CF)

![그림1](https://github.com/jindeok/CF-KAN/assets/35905280/a9b4ca1d-07e4-497b-9ec1-57454475f431)

**CF-KAN** is a **recommendation system (collaborative filtering) based on the Kolmogorov-Arnold Network (KAN) approach**. This project leverages and explores the power of KAN for CF-based recommendation.


## Installation
To get started with CF-KAN, clone the repository and install the required dependencies (pytorch and scipy, etc)

If you are ready, test it with the notebook file (**CF-KAN.ipynb**)


## Priliminary results

Dataset : ML-1M

We found only 10 epochs training on MovieLens-1M produces **pretty convincing results!**
- Recall@20: **0.2013**
- NDCG@20: **0.1315**


!! We will keep update this project! open to new collaboration :)

!! If you found this project is helpful for your research, please STAR this repository

!! For KAN implementation, I've used this repo: https://github.com/Blealtan/efficient-kan
