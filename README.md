# ActiveHARNet-On-Device-Deep-Bayesian-Active-Learning-for-HAR

✔ This repository contains code to our research work/publication, ***[ActiveHARNet: Towards On-Device Deep Bayesian Active Learning for Human Activity Recognition](https://arxiv.org/abs/1906.00108)*** which was presented at [ACM MobiSys 2019 (3rd International Workshop on Embedded and Mobile Deep Learning, EMDL '19)](https://www.sigmobile.org/mobisys/2019/workshops/deepmobile19/).

✔ Various health-care applications such as assisted living, fall detection etc., require modeling of user behavior through Human Activity Recognition (HAR). HAR using mobile- and wearable-based deep learning algorithms have been on the rise owing to the advancements in pervasive computing.

✔ However, there are two other challenges that need to be addressed: first, the deep learning model should support on-device incremental training (model updation) from real-time incoming data points to learn user behavior over time, while also being resource-friendly; second, a suitable ground truthing technique (like Active Learning) should help establish labels on-the-fly while also selecting only the most informative data points to query from an oracle.

✔ Hence, in this paper, we propose ActiveHARNet, a resource-efficient deep ensembled model which supports on-device Incremental Learning and inference, with capabilities to represent model uncertainties through approximations in Bayesian Neural Networks using dropout. This is combined with suitable acquisition functions for active learning.

✔ Empirical results on two publicly available wrist-worn HAR and fall detection datasets indicate that ActiveHARNet achieves considerable efficiency boost during inference across different users, with a substantially low number of acquired pool points (at least 60% reduction) during incremental learning on both datasets experimented with various acquisition functions, thus demonstrating deployment and Incremental Learning feasibility.

## Acknowledgements
Our code structure is inspired by [Deep Bayesian Active Learning](https://github.com/Riashat/Deep-Bayesian-Active-Learning).
