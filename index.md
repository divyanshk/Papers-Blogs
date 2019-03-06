---
layout: default
---

[Mathematics of Deep Learning](#vidal)  
[Wasserstein GAN](#wgan)   
[Why and How of Nonnegative Matrix Factorization](#nmf)   
[DenseNet](#dense)  
[Learning Generative Models with Sinkhorn Divergences](#sinkhorn)   
[Improving GANs Using Optimal Transport](#otgan)   

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

## <a name="dense"></a>Densely Connected Convolutional Networks

* Densenets are a type of deep convolutional networks which try to solve the problem of ResNets
* Instead of having crazy deep nets with identity connections (ResNets), these nets have smaller depth but connect all previous layers to a deeper layer (n * n+1 / 2)
* The feature maps from previous layers are concatenated at every layer
* This way of connection ensures maximum information flow between laters in the netowrk
* This also helps in solving the vanishing gradient problem, and reduces the parameters required as every layer has the information from the previous layers and does not need learn the gradients for it again (feature reuse)
* The authors organise a typical deep DenseNet as a collection of dense blocks (whose all layers are connected to all its previous layers), bottleneck and compression layers
* Having dense blocks allows the use of pooling in between as transitions
* A crucial hyperparameter in DenseNets is growth rate, which is the number of feature maps coming out of a layer; the paper show relatively a small growth rate is sufficient to obtain state-of-the-art results

References
* [Paper](https://arxiv.org/abs/1608.06993)

---

## <a name="sinkhorn"></a>Learning Generative Models with Sinkhorn Divergences

* Training generative models using OT suffers from the computational inefficiency, lack of smoothness and instability
* This paper provides a method to train such models using an OT-based loss called Sinkhorn, based on entropic regularization and utilizing automatic differentiation
* The primal form of the Earth-Mover Distance (EMD) is highly intractable
* Introducing entropic regularization leads to an approximation (Sinkhorn distance) to the full EMD loss

$$ d^{\lambda}_M(r,c) = min_{P\in U(r,c)}\ \sum_{i\,j}\ P_{i\,j} \ M_{i\,j} - \frac{1}{\lambda} h(P)$$

* For $$\lambda \to \infty$$ the solution converges to the original EMD and for $$\lambda \to 0$$ it forms a homogenous solution (MMD)
* The paper proposes a plan to use minibatch sampling loss and Sinkhorn-Knopp's iterative algorithm to solve for $$P^{\lambda}_{i,j}$$
* A neural network is used to learn a cost function (acts as a discriminator)


References
* [Paper](http://proceedings.mlr.press/v84/genevay18a/genevay18a.pdf)
* [Sinkhorn Paper](https://arxiv.org/pdf/1306.0895.pdf)
* [Blog](https://michielstock.github.io/OptimalTransport/)
* [Blog](https://regularize.wordpress.com/2015/09/17/calculating-transport-plans-with-sinkhorn-knopp/)

---

## <a name="otgan"></a>Improving GANs Using Optimal Transport


* This paper presents OTGAN which minimizes aa new metric measuring the distance between the generator distribution and data distribution
* This distance called the mini-batch energy distaance combines OT (primal form) with an energy distance
* The dual form of OT resembles GANs formulation, but primal form allows for closed form solutions
* The authors point out the disadvantage of the method used in the paper above, that it leads to biased estimators of the gradients (because of the mini-batch)
* Salimans et al 2016 suggests using distributions over mini-batches
* The distance suggested

$$ \mathcal{D}^2_{MED}(p, g) = 2\mathbb{E}[\mathbb{W}_c(\mathbf{X}, \mathbf{Y})] - \mathbb{E}[\mathbb{W}_c(\mathbf{X}, \mathbf{X'})] - \mathbb{E}[\mathbb{W}_c(\mathbf{Y}, \mathbf{Y'})]$$

where $$\mathbf{X}, \mathbf{X'}$$ aare individually sampled mini-bathces from distribution $$\textit{p}$$ and $$\mathbf{Y}, \mathbf{Y'}$$ are independent mini-bathces from $$\textit{g}$$

* $$\mathbb{W}$$ is the entropy regularized Wasserstein distance, or the Sinkhorn distance; and the energy function is inspired from Bellemare et al 2017 (CramerGAN)

* The cost function $$c$$ is learned adversarially, as the cosine distance between two latent vectors (a neural network maps the images into a learned latent space)
* They do not backpropogate gradients through the Sinkhorn algorithm
* Downside: requires large amount of computation and memeory

References
* [Paper](https://arxiv.org/abs/1803.05573)

---
