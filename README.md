# CF-KAN

## KAN (Kolmogorov-Arnold Network)-based Collaborative Filtering (CF)

<img src="https://github.com/user-attachments/assets/4387006d-48ab-4f84-a6a3-0ed83982c442" alt="overview_v2" width="400" height="300">


**CF-KAN** is a **recommendation method (collaborative filtering) based on the Kolmogorov-Arnold Network (KAN, https://arxiv.org/abs/2404.19756) approach**. This project leverages and explores the power of KAN for CF-based recommendation.



## Running 
To get started with CF-KAN, clone the repository and install the required dependencies (pytorch and numpy, etc)

If you are ready, test it with

`python main.py`




## The Results (Recommendation Accuracy)


We comprehensively compare CF-KAN with the following categories of recommendation models: 
**Matrix factorization**-based methods: MF-BPR, NeuMF, and DMF;
**Autoencoder**-based methods: CDAE, Multi-DAE, and RecVAE;
**GCN**-based methods: SpectralCF, NGCF ,LightGCN, SGL, and NCL;
**Generative model**-based methods: CFGAN, RecVAE, and DiffRec.
| Method      | ML-1M                |                  |                  |                  | Yelp                 |                  |                  |                  | Anime                |                  |                  |                  |
|-------------|----------------------|------------------|------------------|------------------|----------------------|------------------|------------------|------------------|----------------------|------------------|------------------|------------------|
|             | R@10                 | R@20             | N@10             | N@20             | R@10                 | R@20             | N@10             | N@20             | R@10                 | R@20             | N@10             | N@20             |
| MF-BPR      | 0.0876               | 0.1503           | 0.0749           | 0.0966           | 0.0341               | 0.0560           | 0.0210           | 0.0341           | 0.1521               | 0.2449           | 0.2925           | 0.3153           |
| NeuMF       | 0.0845               | 0.1465           | 0.0759           | 0.0965           | 0.0378               | 0.0637           | 0.0230           | 0.0308           | 0.1531               | 0.2442           | 0.3277           | 0.3259           |
| DMF         | 0.0799               | 0.1368           | 0.0731           | 0.0921           | 0.0342               | 0.0588           | 0.0208           | 0.0282           | 0.1386               | 0.2161           | 0.3277           | 0.3122           |
| CDAE        | 0.0991               | 0.1705           | 0.0829           | 0.1078           | 0.0444               | 0.0703           | 0.0280           | 0.0360           | 0.2031               | 0.2845           | 0.4652           | 0.4301           |
| MultiDAE    | 0.0975               | 0.1707           | 0.0820           | 0.1046           | 0.0531               | 0.0876           | 0.0316           | 0.0421           | 0.2022               | 0.2882           | 0.4577           | 0.4125           |
| RecVAE      | 0.0835               | 0.1422           | 0.0769           | 0.0963           | 0.0493               | 0.0824           | 0.0303           | 0.0403           | 0.2137               | 0.3068           | 0.4105           | 0.4068           |
| SpectralCF  | 0.0751               | 0.1291           | 0.0740           | 0.0909           | 0.0368               | 0.0572           | 0.0201           | 0.0298           | 0.1633               | 0.2564           | 0.3102           | 0.3236           |
| NGCF        | 0.0864               | 0.1484           | 0.0805           | 0.1038           | 0.0417               | 0.0708           | 0.0255           | 0.0346           | 0.2071               | 0.3043           | 0.3937           | 0.3827           |
| LightGCN    | 0.0824               | 0.1419           | 0.0793           | 0.1006           | 0.0505               | 0.0858           | 0.0312           | 0.0417           | 0.2071               | 0.3043           | 0.3937           | 0.3827           |
| SGL         | 0.0885               | 0.1575           | 0.0802           | 0.1029           | 0.0564               | 0.0944           | 0.0346           | 0.0462           | 0.1994               | 0.2918           | 0.3748           | 0.3652           |
| NCL         | 0.0878               | 0.1471           | 0.0819           | 0.1011           | 0.0535               | 0.0906           | 0.0326           | 0.0438           | 0.2063               | 0.3047           | 0.3919           | 0.3819           |
| CFGAN       | 0.0684               | 0.1181           | 0.0663           | 0.0828           | 0.0206               | 0.0347           | 0.0129           | 0.0172           | 0.1946               | 0.2889           | 0.4601           | 0.4289           |
| DiffRec     | 0.1021               | 0.1763           | 0.0877           | 0.1131           | 0.0554               | 0.0914           | 0.0345           | 0.0452           | 0.2104               | 0.3012           | 0.5047           | 0.4649           |
| **CF-KAN**  | **0.1065**           | **0.1831**       | **0.0894**       | **0.1152**       | **0.0590**           | **0.0961**       | **0.0359**       | **0.0471**       | **0.2287**           | **0.3261**       | **0.5256**       | **0.4875**       |



If you found this project is helpful for your research, please STAR this repository


## Citation
If our work was helpful for your project, cite our work :)

** Once the paper is uploaded to arXiv soon, we will provide instructions on how to cite it.


## Etc.
For the implementation of KANs, I've referred to the following repos: [EfficentKAN](https://github.com/Blealtan/efficient-kan), [Visualization API](https://github.com/KindXiaoming/pykan)

