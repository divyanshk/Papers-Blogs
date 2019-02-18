---
layout: default
---

[Mathematics of Deep Learning](#vidal)  
[Wasserstein GAN](#wgan)   
[Why and How of Nonnegative Matrix Factorization](#nmf)   

---

## <a name="vidal"></a>Mathematics of Deep Learning

* The paper talks about the principles of mathematics which underline the field
* Most of the progress of Deep Learning is empirically stated, but the paper sheds light on numerous aspects of ongoing research to theoretically justify the progress
* There are three main factors in deep learning, namely the architectures, regularization techniques and optimization algorithms
* Statistical learning theory suggests that the number of training examples needed to achieve good generalization grows polynomially with the size of the network
* In practice, this is not the case
* One possible explanatino is that deeper architectures produce an embedding of the input data that approximately preserves the distance between data points in the same class
* Key challenge in deep learning is the lack of a convex loss function to minimize
* Due to non-convextity, the set of critical points includes local minimas, saddle points, plateaus, etc. How the model is initialized and the details of the optimization algorithm play a crucial role
* Common strategy to tackle the non-convexity: Randomly initializing the weights, update the weights using local descent, check if the training error decreases sufficiently fast, and if not, choose another initialization
* Geometrical priors: stationarity and local deformations
* Translation invariance and equivariance
* CNNs leverage these priors
* Convolution + Max-pooling = translation invariance
* The relationship between the structure of the data and the deep network is important
* Research shows that networks with random weights preserve the metric structure of the data as they propogate along the layers
* Each layer of a deep network distorts the data proportional to the angle between the two inputs: the smaller the angle, the stronger the shrinkage of the distance
* The addition of ReLU makes the system sensitive to the angles between points
* Information Bottleneck Lagrangian: minimize the empirical cross-entropy and regularize (KL divergence) to minimize the amount of information stored

References
* [Paper](https://arxiv.org/abs/1712.04741)

---

## <a name="wgan"></a>Wasserstein GAN

* Wasserstein GAN (2017) was one of the first papers of GANs trying to solve the mode collapse problem which is pretty evident in most classical GAN architectures
* Mode Collapse: the network is able to learn only a certain modes (distributions) and hence not generalise properly; the generator learn only limited samples
* Modelling $$ P_{\theta} $$ is challenging because the distribution is a low dimensional manifold embedded in a very high dimensional space; the true and model manifold's distribution may not have negligible intersection (singluar support), hence KL divergence won't work well
* The paper talks about a distance measure which is weak so as to make it easier for a sequence of distributions to converge
* Section 2 introduces and evaluates different types of distance metrics, showing Earth-Mover (EM) distance is the better choice of them all
* The WGANs are based on the dual form of the optimal transport problem
* Because of the continuity and differentiality of the loss function, the critic is trained to optimality eliminating mode collapse
* The better the critic, the higher the gradients we use to train the disciminator
* Importantly, the authors report that no experiment suffered from mode collapse
* Lipschitz continuity plays a major role in the success of the dual form of the EMD

References
* [Paper](https://arxiv.org/abs/1701.07875)
* [Blog](https://vincentherrmann.github.io/blog/wasserstein/)

---

## <a name="nmf"></a>Why and How of Nonnegative Matrix Factorization

* NMF is a highly important tool used to extract meaningful and sparse features from a high dimensional matrix
* Most useful applications of NMF are in data mining, document processing, image processing, collaborative filtering, recommendation systems, etc
* Low-rank matrix factorization is representing p-dimensional data points in a r-dimensional linear subspace spanned by basis elements $$w_k$$ and the weights $$h_j$$

$$ x_j \approx \sum_{k=1}^{r}\ w_kh_j(k) \;\text{for some weights}\; h_j \in \mathbb{R}^{r}$$

* The noise model is crucial in the choice of the measure to access the quality of approximation
* NMF aims at decomposing a given nonnegative data matrix X as X ≈ WH where W ≥ 0 and H ≥ 0 (meaning that W and H are component-wise nonnegative)
* NMF is NP-Hard; the unconstrained version of the problem can be solved with SVD though
* NMF is ill-posed; there can be multiple solution to the same problem
* Choice of r is crucial; it can be estimated using SVD, or use trial and error to choose one with lowest error
* Most NMF algorithms follow the general framework of Two-Block Coordinate Descent; eg Alternating Least Squares
* Initialization of the matrices W and H is a challenge; options at hand - randomization, clustering techniques, SVD with $$ [x]_{+} = max(x, 0) $$

References
* [Paper](https://arxiv.org/abs/1401.5226)

---
