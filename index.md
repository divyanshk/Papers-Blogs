---
layout: default
---

[Mathematics of Deep Learning](#vidal)  
[Wasserstein GAN](#wgan)   
[Why and How of Nonnegative Matrix Factorization](#nmf)   
[DenseNet](#dense)  
[Learning Generative Models with Sinkhorn Divergences](#sinkhorn)   
[Improving GANs Using Optimal Transport](#otgan)   
[Mask R-CNN](#maskrcnn)   
[Fully Convolutional Networks for Semantic Segmentation](#fcn)   
[Improving Sequence-To-Sequence Learning Via Optimal Transport](#seq2seqot)   
[Memory-Efficient Implementation of DenseNets](#memdense)   
[Attention Is All You Need](#attn)  
[Analyzing and Improving Representations with the Soft Nearest Neighbor Loss](#soft)    
[Optimal Transport for Domain Adaptation](#otda)   
[Large Scale Optimal Transport and Mapping Estimation](#largescaleot)   
[Autoencoding Variational Bayes](#vae)   

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
* This way of connection ensures maximum information flow between layers in the network
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

## <a name="maskrcnn"></a>Mask R-CNN

* Extension to Faster R-CNN; adds a masking branch which works in parallel to the classification branch
* RoIPool causes a misalignment; RoIAlign fixes that by having no quantizations
* Predict a binary mask for each class independently, without competition among classes, and rely on the network’s RoI classification branch to predict the category
* The first stage, called a Region Proposal Network (RPN), proposes candidate object bounding boxes
* The second layer extracts features using RoIPool from each candidate box and performs classification and bounding-box regression
* In the second stage, in parallel to predicting the class and box offset, Mask R-CNN also outputs a binary mask for each RoI

$$ L = L_{cls} + L_{box} + L_{mask} $$

* RoIPooling is simple and nice; but it causes the target cells to not be of the same size
* RoIAlign prevents this by using bilinear interpolation over the input feature maps; this causes significant improvement in accuracy
* Non-maximum suppression groups highly overlapped boxes for the same class and selects the most confidence prediction only; this avoids duplicates for the same object

References
* [Helpful image](https://cdn-images-1.medium.com/max/2600/1*M_ZhHp8OXzWxEsfWu2e5EA.png)
* [Mark R-CNN Paper](https://arxiv.org/abs/1703.06870)
* [FPN](https://arxiv.org/pdf/1612.03144.pdf)
* [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
* [Blog](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)

---

## <a name="fcn"></a>Fully Convolutional Networks for Semantic Segmentation
* The paper details training a fully convolutional network (FCN) end-to-end, pixels-to-pixels on semantic segmentation
* Upsampling is performed in-network for end-to-end learning by backpropagation from the pixelwise loss; upsampling is backwards strided convolution
* The skip architecture is learned end-to-end to refine the semantics and spatial precision of the output
* For final predictions, the net combines the predictions from earlier pooling layers to capture finer details
* Fine-tuning from classification to segmentation gave reasonable predictions for each net

References
* [Paper](https://arxiv.org/abs/1411.4038)

---

## <a name="seq2seqot"></a>Improving Sequence-To-Sequence Learning Via Optimal Transport
* For word prediction tasks, MLE suffers from [exposure bias problem](https://divyanshk.github.io/Mini-Blogs/#bias)
* Attempts to alleviate the issues are i) RL based ii) Adversarial learning based
* OT loss allows end-to-end supervised training and acts asa an effective sequence-level regularization to the MLE loss
* The authors' novel strategy considers not only the OT distance between the generated sentence and ground truth references, but also the OT distance between the generated sequence and its corresponding input
* The OT approach is derived from soft bipartite matching (see Figure 1 in paper)
* The authors use the IPOT algorithm to solve the OT optimiation problem; though the Sinkhorn algorithm can be used, IPOT was empirially found to be better
* The soft copying mechanism considers semantic similarity in the embedding space
* MLE loss is still require to capture the syntactic structure
* OT losses aare basically used as efficient regularizers

$$
\mathcal{L} = \mathcal{L}_{MLE} + \gamma_1 \ \mathcal{L}_{copy} + \gamma_2 \ \mathcal{L}_{seq} 
$$

* The authors use Wasserstein Gradient Flows (WGF) to descibe how the model approximately leaarns to match the ground-truth sequence distribution
* In WGF, the Wasserstein distaance describes the local geometry of aa trajectory in the space of probability measures converging to a target distribution

References
* [Paper](https://arxiv.org/abs/1901.06283)

---

## <a name="memdense"></a>Memory-Efficient Implementation of DenseNets

* DenseNets are cool; they save up a lo of parameteres by resuse, but their naive implementation can be quadratic with depth in memory
* If not properly managed, pre-activation batch normaliza- tion [7] and contiguous convolution operations can produce feature maps that grow quadratically with network depth
* The quadratic memory dependency w.r.t. feature maps originates from intermediate feature maps generated in each layer, which are the outputs of batch normalization and concatenation operations
* The intermediate feature maps responsible for most of the memory consumption are relatively cheap to compute; frameworks keep these intermediate feature maps allocated in GPU memory for use during back-propagation
* The authors propose reducing the memory cost by using memory pointers, and shared memory locations; also recomputing the intermediate layers output
* The recomputation comes at an additional cost, but reduces the memory cost from qudratic to linear
* Check out the discussion on why pre-activation batch normalization is necessary and useful; and utility of continguous memory allocations for performing convolutions
* They propose two pre-allocated Shared Memory Storage locations to avoid the quadratic memory growth. During the forward pass, we assign all intermediate outputs to these memory blocks. During back-propagation, we recompute the concatenated and normalized features on-the-fly as needed
* In the code this is implemented using checkpointing

References
* [Paper](https://arxiv.org/pdf/1707.06990.pdf)
* [Code](https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py)

---

## <a name="attn"></a>Attention Is All You Need

* Generally, language models are comprised of RNNs. Recently CNNs have been used for such tasks (ByteNet, ConvS2S). This paper prexents a novel model architecture called Transformer, which is based entirely on self aattention
* RNNs are inherently serial, not parallelizable; and computatinoaly expensive
* The number of operations required to relate signals from two arbitrary input and output positions grows in the distane between postitions, linearly for ConvS2S and logarithmically for ByteNet; Transformer based model have constant number of operations
* Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representaiton fo the sequence
* The arch is made up to encoder and decoder stacks, each having modules for sel-attention and point-wise, fully connected layers (similar to 1x1 convolution layers)
* The decoder takes inputs from the encoder's outputs and the different positions on the query it is decoding (see animation in the Google blog)
* The output of the attention function is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key
* The attention weights are the relevance of the encoder hidden states (values) in processing the decoder state (query) and are calculated based on the encoder hidden states (keys) and the decoder hidden state (query)
* The query is the 'word' being decoded; keys are the input sequence, and the output is the sequence with relevance weights, with higher weight on the correct translation for that 'word'
* The encoder uses source sentence’s embeddings for its keys, values, and queries, whereas the decoder uses the encoder’s outputs for its keys and values and the target sentence’s embeddings for its queries 
* Multi-head attention lets the architecure parallelise the attention mechanism; jointly attend to information from different representation subspaces at different positions
* Encoder uses self attention, where it can attend to all positions in the previous layer of the encoder
* The architecture includes masking in the decoder to prevent leftward information flow
* Weight tying is used in the embeddigns and pre-softmax linear transformation layers
* Positional encoding injects some informaation about the relative or absolute position of the token in the sequence

References
* [Paper](https://arxiv.org/pdf/1706.03762.pdf)
* [Blog](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
* [Google Blog](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

---

## <a name="soft"></a>Analyzing and Improving Representations with the Soft Nearest Neighbor Loss

* The paper studies internal representations with a loss function, to measure the lack of separation of class manifolds in representation space—in other words, the entanglement of different classes
* They focus on the effect of deliberately maximizing the entanglement of hidden representations in a classifier. Surprisingly, unlike the penultimate layer, hidden layers that perform feature extraction benefit from being entangled, ie, they should not be forced to disentangle data from different classes
* The entangled representations form class-independent clusters which capture other kinds of similarity that is helpful for eventual discrimination
* Since entangled representations exhibit a similar- ity structure that is less class-dependent, entangled models more coherently project outlier data that does not lie on the training manifold
* In particular, data that is not from the training distribution has fewer than the normal number of neighbors in the predicted class
* The entanglement of class manifolds characterizes how close pairs of representations from the same class are, relative to pairs of representations from different classes. 
* If we have very low entanglement, then every representation is closer to representations in the same class than it is to representations in different classes; if entanglement is low then a nearest neighbor classifier based on those representations would have high accuracy
* The soft nearest neighbor loss is the negative log probability of sampling a neighboring point j from the same class as i
* Cross entropy loss is minimized and entanglement (soft nearest neighbor loss) is maximised as a regularizer
* This not only promotes spread-out intraclass representations, but also turns out to be good for recognizing data that is not from the training distribution by observing that in the hidden layers, such data has fewer than the normal number of neighbors from the predicted class


References
* [Paper](https://arxiv.org/pdf/1902.01889.pdf)

---

## <a name="otda"></a>Optimal Transport for Domain Adaptation

* This paper introduces the theory of optimal transport with application to the problem of domain adaptation, specifically dealing with unsupervised learning
* The domain adaptation problem boils down to i) finding a transformation of the input data matching the source and target distributions, aand ii) learning a new classifier from the transformed source samples
* OT distance make a strong case for such problems because i) they can be evaluated directly on empirical estimates of the distruibutions, and ii) they exploit the geometry of the underlying metric space
* A common stratget to tackle unsupervised domain adaptation is to propose methods that aim at finding representations in which domains match in some sense
* The paper introduces the theory of OT with the intuition to applying it to domain adaptation
* Regulaarization is key in solving OT linear problems, as it induces some properties of the solution; reduces overfitting
* Entropic regularization introduced by Cuturi (and using Sinkhorm computation) is cruical
* Intuition for such a regularization: since most elements of the transport should be zero with high probability, one can look for a smoother version of the transport, thus lowering its sparsity, by increasing its entropy
* As the parameter controlling the entropic regularization term increases, the sparsity of the plan decreases aand source points tend to distribute their porobablity masses toward more taarget points
* The paper provides regularization to the barycentric mapping solution of the transformation mapping source to taaret domains
* But these only cover basic affine transformations, the paper doesn't provide solutions that can be directly applied to modern deep learning problems

References
* [Paper](https://arxiv.org/abs/1507.00504)

---

## <a name="largescaleot"></a>Large Scale Optimal Transport and Mapping Estimation

* This paper proesents a novel two step approach for the fundamental problem og learning an optimal map from one distribution to another
* The technique is tested on generative modelling and domain adaptation
* Cuturi's work on entropic regularization does not scale well, while Wasserstein GANs scale but have to satisfy the non-trivial 1-Lipschitz condition
* First, the authors propose an algorithm to compute the optimal traansport plan using dual stochastic gradient for solving regularizied dual form of OT
* Second, they learn an optimal map (Monge map) as a neural network by approximating the barycentric projection of the OT plan obtained in the first step
* Parameterizing using a neural network allows efficient learning and provides generalization outside the support of the input measure
* The authors use the regularized OT dual formulation; and provide convergence proofs
* The barycentric projection wrt the squared Euclidean cost is oftern used as a simple way to recover optimal maps from optimal transport plans
* The authors claim the restricted application of [Courty et al](#otda) is maily due to the fact that tey consider the primal formulation of the OT problem
* Source saamples are mapped to the target set through the barycentric projection; a classifier is then learned on the mapped source samples
* In all experiments conducted, the squared Euclidean distance is used as the ground cost and the barycentric projection is comptued wrt that cost

References
* [Paper](https://arxiv.org/abs/1711.02283)

---

## <a name="vae"></a>Autoencoding Variational Bayes

* Encoder encodes x to latent variable z
* Decoder decodes z to x (reconstruction)
* At test time, when we want to generate new samples, we simply input values of z ∼ N (0, I) into the decoder; we remove the “encoder"
* The key idea behind the variational autoencoder is to attempt to sample values of z that are likely to have produced X, and compute P(X) just from those
* We need a new function Q(z|X) which can take a value of X and give us a distribution over z values that are likely to produce X
![equation1](/images/vae_eq1.png)
* This equation serves is the core of the variational autoencoder
* We want to maximise logP(X) and the error term, which makes Q produce z’s that can reproduce a given X
* The Q function takes the form of a Gaussian with mean aand covariance as outputs used to model the distribution
![equation2](/images/vae_eq2.png)
* This equation refers to the last term of the VAE equation, where k is the dimensionality of the distribution
* As is standard in stochastic gradient descent, we take one sample of z and treat P(X\|z) for that z as an approximation of the first term on the right
* To prevent the sample from Q(z|X) in the first term on the right during the backprop stage, we use the reparamaterization trick
![equation3](/images/vae_eq3.png)
* The second term on the left can be viewed as a regularization term, much like the sparsity regularization in sparse autoencoders
* Refer to figure 4 in the tutorial linked below for a pictorial explanation

References
* [Paper](https://arxiv.org/abs/1312.6114)
* [Tutorial on VAE](https://arxiv.org/abs/1606.05908)
* Deep Learning lab slides

---