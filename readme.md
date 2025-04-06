# DeepSurv.pytorch

This repository is an unofficial pytorch implementation of 
[DeepSurv: Personalized Treatment Recommender System Using
A Cox Proportional Hazards Deep Neural Network](https://link.springer.com/article/10.1186/s12874-018-0482-1).
We reimplement the experiments in the paper, which is followed by [Github](https://github.com/jaredleekatzman/DeepSurv), and the detailed understanding is available on my [Blog](https://www.cnblogs.com/CZiFan/p/12674144.html).

## Requirements
- Install Pytorch>=0.4.0
- Run:
```
pip install requirements.txt
```
  
## Usage
Run:
```
python main.py
```

## Replication Results
| | Simulated Linear | Simulated Nonlinear |   WHAS   | SUPPORT  | METABRIC | Simulated Treatment | Rotterdam & GBSG |
|---|---|---|---|---|---|---|---|
| Paper | 0.774019 | 0.648902 | 0.862620 | 0.618308 | 0.643374 | 0.582774 | 0.668402 |
|Our implements|     0.778607     |       0.652048      | 0.841484 | 0.618107 | 0.643453 |       0.552648      |     0.673290     |

## Citation
```
@article{Katzman2016DeepSurv,
  title={DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network},
  author={Katzman, Jared and Shaham, Uri and Bates, Jonathan and Cloninger, Alexander and Jiang, Tingting and Kluger, Yuval},
  journal={Bmc Medical Research Methodology},
  volume={18},
  number={1},
  pages={24},
  year={2016},
}
```
