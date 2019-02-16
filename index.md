---
layout: default
---

[Mathematics of Deep Learning](#vidal)   

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
