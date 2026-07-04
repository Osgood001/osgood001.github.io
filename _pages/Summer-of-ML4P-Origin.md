---
layout: post
date: 2033-07-19 20:50:59 +0800
title: "Machine Learning & Statistical Physics"
tags: [Research]
---

*This note is for the summer school of Machine Learning and statistical physics held at YNU, 2023.*

Written by Shigang Ou (oushigang19@mails.ucas.ac.cn)

> *This note is under rapid change, and may contain mis-information or redundent content, including some extended information not mentioned in the summer school*

- [What is Neural Network?](#what-is-neural-network)
  - [MLP](#mlp)
  - [Hopfield Network](#hopfield-network)
  - [Curie-Weiss Model](#curie-weiss-model)
  - [Network Phase Diagram](#network-phase-diagram)
  - [NN dynamics and it's weight spectrum](#nn-dynamics-and-its-weight-spectrum)
  - [Network](#network)
- [Unsupervised Learning](#unsupervised-learning)
  - [Boltzman Machine](#boltzman-machine)
  - [自动重构](#自动重构)
  - [GAN](#gan)
  - [Contrast Learning](#contrast-learning)
  - [Score-Based Diffusion Model](#score-based-diffusion-model)
- [Statistical Physics of Learning](#statistical-physics-of-learning)
  - [玻璃态](#玻璃态)
    - [Replica Method](#replica-method)
  - [Replica Symmetry Breaking](#replica-symmetry-breaking)
- [Cavity Method and 2-layer Perceptron](#cavity-method-and-2-layer-perceptron)
  - [Cavity Method](#cavity-method)
  - [大偏差理论](#大偏差理论)
- [Quantum Many-body Physics Introduction](#quantum-many-body-physics-introduction)
  - [Quantum Many-body Physics](#quantum-many-body-physics)
  - [Quantum Many-body Problem](#quantum-many-body-problem)
  - [Quantum Many-body Problem in 1D](#quantum-many-body-problem-in-1d)
  - [Quantum Many-body System introduction](#quantum-many-body-system-introduction)
  - [Superconductivity](#superconductivity)
    - [BCS Theory](#bcs-theory)
      - [Cooper Pair](#cooper-pair)
  - [Superfluidity](#superfluidity)
    - [Bose-Hubbard Model for Superfluidity and Mott Insulator](#bose-hubbard-model-for-superfluidity-and-mott-insulator)
- [Spin Glass: Theory and applications](#spin-glass-theory-and-applications)
  - [Setting](#setting)
- [Connections between ML and SP](#connections-between-ml-and-sp)
  - [Statistical Physics and Statistical Inference is the same thing](#statistical-physics-and-statistical-inference-is-the-same-thing)
  - [Information Compression](#information-compression)
  - [Why NNs?](#why-nns)
  - [What I cannot Create, I do not Understand](#what-i-cannot-create-i-do-not-understand)
  - [Quantum Supremacy](#quantum-supremacy)
  - [Statistical Physics of Learning](#statistical-physics-of-learning-1)
  - [Curie-Weiss Model](#curie-weiss-model-1)
  - [Network Phase Diagram](#network-phase-diagram-1)
  - [Implicit vs Explicit Generative Models](#implicit-vs-explicit-generative-models)
  - [Statistical Physics or Statistics?](#statistical-physics-or-statistics)
  - [Formal Definition of Statistical Physics](#formal-definition-of-statistical-physics)
  - [Learning Machine](#learning-machine)
  - [When to use linear and when non-linear?](#when-to-use-linear-and-when-non-linear)
  - [Occam's Razor](#occams-razor)
  - [NN dynamics and it's weight spectrum](#nn-dynamics-and-its-weight-spectrum-1)
  - [Other Model-free methods](#other-model-free-methods)
  - [Dynamic Systems: A Prior](#dynamic-systems-a-prior)
  - [Reservoir Computing](#reservoir-computing)
  - [Systemetic Prediction based on Periodic Orbits](#systemetic-prediction-based-on-periodic-orbits)
- [Monte Carlo Methods](#monte-carlo-methods)
- [Quantum Annealing](#quantum-annealing)
  - [An problem: Phase transition of Ising model](#an-problem-phase-transition-of-ising-model)
- [Dynamics of cell state transition](#dynamics-of-cell-state-transition)
- [Human population behavior and propagation dynamics](#human-population-behavior-and-propagation-dynamics)
  - [information spreading dynamics](#information-spreading-dynamics)
  - [Disease spreading dynamics](#disease-spreading-dynamics)
- [Spiking Neural Networks](#spiking-neural-networks)
- [Probalistic inference reformulated as tensor networks](#probalistic-inference-reformulated-as-tensor-networks)
  - [Reformulate probalistc inference as tensor networks](#reformulate-probalistc-inference-as-tensor-networks)
- [Variational Autoregressive Network](#variational-autoregressive-network)
  - [Chemical reaction simulation with VAN](#chemical-reaction-simulation-with-van)
- [Machine learning and chaos](#machine-learning-and-chaos)
  - [Feature of chaos](#feature-of-chaos)
  - [How ML is doing better than traditional methods?](#how-ml-is-doing-better-than-traditional-methods)
  - [Chaos synchronization](#chaos-synchronization)
- [Definition and Characterization of Chaos](#definition-and-characterization-of-chaos)
  - [Feature of chaos](#feature-of-chaos-1)
- [Prediction of Chaos](#prediction-of-chaos)
  - [Problem Formulation](#problem-formulation)
    - [Generate Data](#generate-data)
    - [Create a Reservoir](#create-a-reservoir)
  - [A (very) short tutorial on Reservoir Computing](#a-very-short-tutorial-on-reservoir-computing)
- [AI for Physics](#ai-for-physics)
  - [Wave System](#wave-system)
    - [How recurrent NN can be mapped to wave function](#how-recurrent-nn-can-be-mapped-to-wave-function)
    - [How quantum grover algorithm can be mapped to ML](#how-quantum-grover-algorithm-can-be-mapped-to-ml)
  - [Diffusion System](#diffusion-system)
    - [Manifold learning](#manifold-learning)
    - [Diffusion mapping](#diffusion-mapping)
  - [1.1 网络拓扑结构](#11-网络拓扑结构)
  - [1.2 马尔可夫转移矩阵](#12-马尔可夫转移矩阵)
  - [1.3 网站排序](#13-网站排序)
- [2 稀疏矩阵的幂运算](#2-稀疏矩阵的幂运算)
  - [2.1 运算的化简](#21-运算的化简)
    - [Langevin diffusion](#langevin-diffusion)
  - [Topological Phonon Hall Effect](#topological-phonon-hall-effect)
  - [Recommendation system](#recommendation-system)
  - [Machine learning stochastic dynamics](#machine-learning-stochastic-dynamics)
    - [Marvovian stochastic dynamics](#marvovian-stochastic-dynamics)
  - [Chemical master equation](#chemical-master-equation)
    - [A solvable example: birth-death process](#a-solvable-example-birth-death-process)
    - [Gillespie algorithm](#gillespie-algorithm)
  - [Stochastic reaction networks with $M$ species (each with count $N$)](#stochastic-reaction-networks-with-m-species-each-with-count-n)
- [Learning nonequilibrium statistical mechanics and dynamical phase transitions](#learning-nonequilibrium-statistical-mechanics-and-dynamical-phase-transitions)
  - [Dynamic partition function and observable](#dynamic-partition-function-and-observable)
  - [Track the distribuiton oin Ornstein-Uhlenbeck process](#track-the-distribuiton-oin-ornstein-uhlenbeck-process)
- [Machine Learning, Statistical Physic, and Complex System](#machine-learning-statistical-physic-and-complex-system)
  - [Long-range connected 2-D network percolation](#long-range-connected-2-d-network-percolation)
- [厄尔尼诺预测](#厄尔尼诺预测)
- [Variational Bayesian Method](#variational-bayesian-method)
  - [Fluctuation theorem](#fluctuation-theorem)
  - [Diffusion Model](#diffusion-model)
  - [(Jiaming Song et al, ICLR 2021)  Denoising Diffusion Implicit Model](#jiaming-song-et-al-iclr-2021--denoising-diffusion-implicit-model)
  - [AI4Materials](#ai4materials)
- [Close speech](#close-speech)


## What is Neural Network?

> 2023-07-19,  Huang: "Statistical physics of Neural Network"

| Network Name | Developer Name | Year | Introduction |
| ------------ | -------------- | ---- | ------------ |
| Multilayer Perceptron (MLP) | Frank Rosenblatt | 1958 | A simple and widely used type of neural network that consists of an input layer, one or more hidden layers, and an output layer of artificial neurons. |
|  Network | John  | 1982 | A type of recurrent neural network that can store and retrieve patterns as stable states. It consists of a single layer of fully connected neurons with symmetric weights and binary activations. |
| Recurrent Neural Network (RNN) | David Rumelhart et al. | 1986 | A type of neural network that can process sequential data such as text, speech, and  series. It consists of a hidden layer of neurons that have recurrent connections to themselves, forming a loop. |
| Convolutional Neural Network (CNN) | Yann LeCun | 1989 | A type of feedforward neural network that can process high-dimensional data such as images, videos, and speech. It consists of multiple layers of neurons that perform convolution operations on the input data, followed by pooling layers and fully connected layers. |
| Spiking Neural Network (SNN) | Eugene Izhikevich et al. | 2003 | A type of neural network that mimics the behavior of biological neurons more closely than other types. It consists of spiking neurons that communicate with each other using discrete pulses or spikes. |

> This list can go on and on, along with the history of winters and springs of AI. But how to understand the neural network in a more general way?
> 
> **Some well-established theories in the history can still be used today** 

### MLP

MLP is defined as:

$$
x \mapsto W_D \sigma_{D-1}(W_{D-1} \sigma_{D-2}(\dots \sigma_1(W_1 x))),
$$

where $W_i$ is the weight matrix of the $i$-th layer, $\sigma_i$ is the activation function of the $i$-th layer, and $D$ is the depth of the network.

```mathematica
WHERE IS MY BIAS?
Assume a layer with input x, weight W, bias b, and activation sigma
A typical layer is defined as:
y = \sigma(Wx + b) 

Append 1 to x and b to W to get new input x_tilde and new weight W_tilde
x_tilde = (x, 1)^\top
W_tilde = (W, b)

Rewrite the output of the layer using x_tilde and W_tilde
y = \sigma(W_tilde x_tilde) 

This has same output as before, but no bias term
```

> This function is unreasonably effective in many tasks, but why?
> 
> **Over-parameterization** is the key.

https://arxiv.org/pdf/2008.06786.pdf: Triple Descent and a Multi-Scale Theory of Generalization

![Image](https://pic4.zhimg.com/80/v2-fdf1aee4b947708887673d610b14c2c3.png)

> Traditional idea: more parameters, more overfitting.
> 
> **This is not the full story**: $p$ is the number of parameters, $m$ is the number of training samples.
>
> Explained in 2019 by [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/pdf/1912.02292.pdf):

### Hopfield Network

[Neural Networks and Physical Systems with Emergent Collective Computational Abilities](
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/pdf/pnas00447-0135.pdf):

Hopfield network is defined as:

$$
V_i(t+1) = \begin{cases}
1 & \text{if } \sum_{j=1}^N T_{ij}V_j(t) > U_i \\
0 & \text{if } \sum_{j=1}^N T_{ij}V_j(t) < U_i \\
V_i(t) & \text{otherwise}
\end{cases}
$$

where $V_i(t)$ is the state of neuron $i$ at time $t$, $T_{ij}$ is the weight of the synapse from neuron $j$ to neuron $i$, and $U_i$ is the threshold value for neuron $i$.

![Image](https://pic4.zhimg.com/80/v2-129568210c5345b72d3ee192890d05d3.png)

> This definition mimics the biological neurons like Fig.1, anything cool?
>
> Computational ability that can **generalize, categorize, correct errors, and recognize familiarity.**


If we want to store a binary vector $V^s\in \{0,1\}^n$ to it, we can set the weight matrix as:

$$
T_{ij} = \sum_{s=1}^n (2V_i^s - 1)(2V_j^s - 1)
$$

<!-- This is a **Hebbian learning rule**: neurons that tend to fire together have stronger positive connections, and neurons that tend to fire oppositely have stronger negative connections. -->

> Here $2V-1 \in [-1, 1]$ is the binary to bipolar transformation.
>
> 同种状态相互促进，不同状态相互抑制
>
> Here weight matrix is **fixed**, unlike the MLP.

To retrieve a stored memory from an initial state $V(0)$, the neurons are then updated randomly and asynchronously by the Hopfield network dynamics until the network reaches a stable state.

> The update rule is **asynchronous** and **random**? Why not synchronous and deterministic?
>
> **The update rule is not important, the energy function is.**

We can define the energy function of the network as:

$$
E = -\frac{1}{2}\sum_{i,j=1}^N T_{ij}V_iV_j + \sum_{i=1}^N U_iV_i
$$

The first term is the **interaction energy** between neurons, and the second term is the **external energy** of the neurons.

> It's proved that the energy function is a **Lyapunov function** of the Hopfield network dynamics, which means that the energy function will decrease or remain constant at each step, until it reaches a minimum value that corresponds to a stable state of the network.

Proof:

$$
\begin{aligned}
E(t+1) - E(t) &= -\frac{1}{2}\sum_{i,j=1}^N T_{ij}(V_i(t+1) - V_i(t))(V_j(t+1) - V_j(t)) \\
&= -\frac{1}{2}\sum_{i,j=1}^N T_{ij}(V_i(t+1) - V_i(t))^2 \\
&\leq 0
\end{aligned}
$$

> Do the stable states acutally correspond to the stored memories?
>
> **Not necessarily.** The stable states of the Hopfield network are the local minima of the energy function, which may not correspond to the stored memories. **Ising Model** and **Spin Glass** provided the answer for different $T_{ij}$.

![Image](https://pic4.zhimg.com/80/v2-f442222df7e3303e15cef152316fd64a.png)

> This is the MC simulation result, local minima, or noise, exists. How to reduce it?
>
> This paper find **clipping the weights increase noise.**

- Generalize: Restore the stored memories from a corrupted version of the memories in a consistant way.
- Categorize: Classify the input into one of the stored memories.
- Correct errors: Correct the corrupted memories to the stored memories.
- Recognize familiarity: Guide the initial state to nearest stored memories (hamming distance).

> What cool application can be made out of this? In principle: MNIST digit recognition, image denoising, and even generative models.

See:

- [Text2Image with Hopfield Networks](https://github.com/YonghaoXu/Txt2Img-MHN)
- [Hopfield Networks is All You Need](https://arxiv.org/pdf/2008.02217.pdf)

> Btw, Restricted Boltzmann Machine (RBM) is a special case of Hopfield network.

---
>Hands-on session!

For coding convenience, we can define the Hopfield network as:

Hopfield network is a Graph $G=(V, T)$, where $V$ is the set of vertices and $W$ is the set of edges. The state of the network is a binary vector $V(t)\in \{-1,1\}^n$, where $n$ is the number of vertices. The weight matrix $T$ is defined as: 

$$
T_{ij} = \sum_{s=1}^n V_i^sV_j^s
$$

where $V_i^s$ is the $i$-th element of the $s$-th stored memory $V^s$.

```
<!-- In Progress  -->
```
### Curie-Weiss Model

> Curie-Weiss model is a mean-field model of ferromagnetism, which was introduced by Pierre Curie and Pierre Weiss in 1906. It is used to describe the behavior of ferromagnetic materials above the Curie temperature, where the magnetic moments of the atoms are randomly oriented. The model assumes that the magnetic moments are aligned in the same direction, and it predicts that the magnetization of the material is proportional to the applied magnetic field.

Energy:

$$
\mathcal{H} = -\frac{1}{2} \sum_{i<j} Js_i s_j 
$$

> This contains only 1 parameter, $J$.

The solution is

$$
m = \tanh{(\beta (n-1) J m)}
$$

where $m$ is the magnetization, $n$ is the number of particles, $J$ is the interaction strength, and $\beta$ is the inverse temperature.

This equation has two solutions:

> 1 parameter saves 2 data points.

<!-- why? -->

###  Network Phase Diagram

![Image](https://pic4.zhimg.com/80/v2-822fdf7e141228673104056b90f26ada.png)

Taken from [Dreaming Neural Networks: Forgetting spurious memories and reinforcing pure ones](https://arxiv.org/pdf/1810.12217.pdf)

To learn from data, the weights of the network weights $J_{ij}$ are chosen as:

$$
J_{ij} = \argmin_{J_{ij}} \Big(-\beta \sum_{\mu=1}^m \sum_{i<j} x_i^\mu x_j^\mu J_{ij} - \log Z\Big)
$$

Which is derived from the maximum likelyhood estimation.

To minimize it, we take derivative of it with respect to $J_{ij}$:

$$
\frac{\partial \mathcal{L}}{\partial J_{ij}} = -\beta \sum_{\mu=1}^m x_i^\mu x_j^\mu + \frac{\partial \log Z}{\partial J_{ij}}\\
= -\beta \sum_{\mu=1}^m x_i^\mu x_j^\mu + \beta \sum_{\mu=1}^m \lang x_i^\mu x_j^\mu \rang_J\\
= -\beta ( \lang x_i x_j \rang_\text{data} - \lang x_i x_j \rang_J)
$$

where $\lang x_i x_j \rang_\text{data}$ is the average of $x_i x_j$ over the data, and $\lang x_i x_j \rang_J$ is the average of $x_i x_j$ over the distribution defined by the weights $J_{ij}$.

> This is inverse Ising problem. Because we want to find $J_{ij}$ from $x_i$. i.e. design Hamiltonian to fit the data.

>Phase diagram is so cool, it has been applied in NN like here:
![Image](https://pic4.zhimg.com/80/v2-bd0f2e85916efbdbe37afb8c8e5ef0a3.png)
Taken from [Phase diagram for two-layer ReLU neural networks and infinite-width limit](https://arxiv.org/pdf/2007.07497.pdf)

> The above model only considers two body interaction, but in reality, there's higher order interaction. (Not discussed yet)

Consider Ising model with hidden variable:

$$
E = \sum_{i, a}s_i\sigma_a J_{ia} + \sum_{i<j} s_i s_j J_{ij} + \sum_{a<b} \sigma_a \sigma_b J_{ab}
$$

where $s_i$ is the spin of the $i$-th particle, $\sigma_a$ is the spin of the $a$-th hidden variable, $J_{ia}$ is the interaction strength between particle $i$ and hidden variable $a$, $J_{ij}$ is the interaction strength between particles $i$ and $j$, and $J_{ab}$ is the interaction strength between hidden variables $a$ and $b$.

But we can only observe $s_i$, so we need to marginalize over $\sigma_a$:

$$
p(s) = \sum_{\sigma} p(s,\sigma) = \sum_{\sigma} \frac{e^{-\beta E(s,\sigma)}}{Z}
$$

The loss is 

$$
\mathcal{L} = \sum_{\mu=1}^m \log \sum_{\sigma} e^{-\beta E(x^\mu,\sigma)}/Z
$$

> This is the same as the above.

Similar to the above, we take derivative of it with respect to $J_{ia}$:

$$
\frac{\partial \mathcal{L}}{\partial J_{ia}} = -\beta \sum_{\mu=1}^m x_i^\mu \sigma_a^\mu + \beta \sum_{\mu=1}^m \lang x_i^\mu \sigma_a^\mu \rang_J\\
= -\beta ( \lang x_i \sigma_a \rang_\text{data + hidden} - \lang x_i \sigma_a \rang_J)
$$

> This is slow using MCMC. Hinton proposed a faster way: RBM. Which is a bipartite graph where hidden variable is independent of each other, and visible variable is independent of each other too.

Hinton proposed Contrastive Divergence (CD) algorithm to solve this problem.

### NN dynamics and it's weight spectrum

![Image](https://pic4.zhimg.com/80/v2-b358b12f04ee39b907472f343436474f.png)
Taken from [Correlation Between Eigenvalue Spectra and Dynamics
of Neural Networks](https://nba.uth.tmc.edu/homepage/cnjclub/2009Fall/NecoNNSpectra2009.pdf)

The main contributions of the paper are:

- It provides a clear criterion to distinguish between the chaos phase and the memory phase in asymmetric neural networks with associative memories, based on the eigenvalue spectra of the synaptic matrices.

- It reveals a novel phenomenon of eigenvalue splitting in the memory phase, and shows that the number and positions of the split eigenvalues are related to the number and stability of the memory attractors.

- It develops a mean-field theory to derive analytical expressions for the eigenvalue spectra and the dynamical properties of the neural network, and verifies them with numerical simulations.

> A new proposal:
>
> find different phases and order parameters of NN. Integrate existing works.
>
> The NN can be MLP, Hopfield, RNN, CNN, SNN, etc.

One work relating to the grokking behavior in NNs:

![Image](https://pic4.zhimg.com/80/v2-7026cbe28c84f492961bce3e2a0410fc.png)
Taken from [Towards Understanding Grokking:
An Effective Theory of Representation Learning](https://arxiv.org/pdf/2205.10343.pdf)
> but it does not offer any meaningful insight into the phase diagram.

<!-- ![Image](https://pic4.zhimg.com/80/v2-521e12d49b4c0a8d3ff19656608a8a6b.png) -->

![Image](https://pic4.zhimg.com/80/v2-243cea4a68fa4383d46796b460517091.png)

> This so-called phase diagram/phase transition is an abuse of terms in physics. Limited theoretical insights. More discussion see [OpenReview of this paper](https://openreview.net/forum?id=6at6rB3IZm)
>
> A reviewer claim that:
> There has been a couple of papers that use tools from physics and provide phase diagrams for understanding the generalization of neural networks:

- Generalisation error in learning with random features and the hidden manifold model by Gerace et al
- Multi-scale Feature Learning Dynamics: Insights for Double Descent by Pezeshki et al
- The Gaussian equivalence of generative models for learning with two-layer neural networks by Goldt et al
- [Statistical mechanics for neural networks with continuous-time dynamics by Kuhn et al](https://arxiv.org/pdf/2012.04728.pdf)

> Here the last paper apply Noether's theorem to NN, yielding predictions to the training dynamics of NNs.

###  Network

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/pdf/pnas00447-0135.pdf: Neural Networks and Physical Systems with Emergent Collective Computational Abilities

## Unsupervised Learning

### Boltzman Machine

set

$$
p(x)=\frac{e^{-E(x)}}{Z}
$$
where $Z=\sum_x e^{-E(x)}$ is the normalization factor.
we can learn $E(x)$ from the data by minimzie
$$
\mathcal{L}=-\frac{1}{|D|}\sum_{x\in D}\ln{p(x)}
$$

for example, if we set $E(x)=-\frac{1}{2}W_{ij}x_ix_j$, then all we need is the gradient of $\mathcal{L}$ w.r.t. $W_{ij}$:
$$
\frac{\partial \mathcal{L}}{\partial W_{ij}} = -\frac{1}{2}(\lang x_ix_j \rang_{x\thicksim D} - \lang x_ix_j \rang_{x \thicksim p})
$$

and we can use gradient descent to minimize it
$$
W_{ij}'=W_{ij} + \eta (\lang x_ix_j \rang_{x\thicksim D} - \lang x_ix_j \rang_{x \thicksim p})
$$

What makes the algorithm costy is the generation of $x\thicksim p$, which is also updated and generated every step using MCMC. To deal with this, one may use the samples from $D$, for example the last batch, to replace $p$.

In practice, when we trained the energy function, we often treat it as the following form
$$
E(x)=-\sum_{i}a_ix_i - \sum_j\ln{(1 + e^{\sum_{i}W_{ij}x_i + b_j})}
$$
which is derived from the following hypothesis
$$
p(x)=\frac{1}{Z}\sum_h e^{-E(x,h)}
\text{~(hidden variable)}\\
E(x,h)=-\sum_i a_ix_i -\sum b_j h_j - \sum_{i,j}x_iW_{ij}h_j
\text{~(energy paramterization)}\\
p(h|x)=\Pi_j p(h_i|x)\text{~(bipartite graph)}\\
p(x|h)=\Pi_i p(x_i|h)\text{~(bipartite graph)}\\
$$

The gradient in general would be
$$
\frac{\partial \mathcal{L}}{\partial \theta}
= \lang \frac{\partial E}{\partial \theta} \rang _{x\thicksim D, h\thicksim p(h|x)} - \lang \frac{\partial E}{\partial \theta} \rang_{x \thicksim p(x),h \thicksim p(h|x)}
$$

### 自动重构

input equals output.

Variational Autoencoder, which works as follows:

1. encode the input into a distribution $q(z|x)$
2. sample $z$ from $q(z|x)$
3. decode $z$ into a distribution $p(x|z)$
4. sample $x$ from $p(x|z)$
5. minimize the negative log likelyhood of $x$ w.r.t. $q(z|x)$ and $p(x|z)$

### GAN


set

$$
\max_{G} \min_{D} \mathcal{L}(D,G) = \mathbb{E}_{x\thicksim p_{data}(x)}[\log{D(x)}] + \mathbb{E}_{z\thicksim p_{z}(z)}[\log{(1-D(G(z)))}]
$$

where $G$ is the generator, $D$ is the discriminator, $p_{data}(x)$ is the data distribution, $p_{z}(z)$ is the noise distribution, and $G(z)$ is the generated data.

### Contrast Learning

Contrast Learning is a general framework for unsupervised learning. The idea is to learn a representation that makes the data distribution and the noise distribution distinguishable.

<!-- Under Consideration -->

### Score-Based Diffusion Model

![Image](https://pic4.zhimg.com/80/v2-4b7f04222de9d0252579dc826dc33b01.png)

Jacot e.t.a. 2018

## Statistical Physics of Learning

<!-- The following will be written in Chinese, Simplified -->

### 玻璃态

非磁金属中的磁性杂质：AuFe 导致自旋磁矩方向无规，长程无序。

**Glass state is a non-equilibrium, non-crystalline condensed state of matter that exhibits a glass transition when heated towards the liquid state.**

Edwards-Anderson model: Ising spin glass.

<!-- `![Image](https://pic4.zhimg.com/80/v2-dfc0b389dcc54f376cd6061ff86f9ae5.png)` -->

Sherrington-Kirkpatrick model: Heisenberg spin glass.

<!-- This is solved using replica method -->

#### Replica Method

<!-- Detailed derivation is to be continued -->

Goal: compute free energy $F=-\frac{1}{\beta}\ln{Z}$

1. write out Z: $Z=\sum_{\{s_i\}}e^{-\beta H(s_1,\dots,s_N)}$
2. use limit property: $-\ln{Z}=\lim_{n\to 0}\frac{Z^n-1}{n}$
3. expand it: $Z^n=\sum_{\{s_i^a\}}e^{-\beta H(s_1^1,\dots,s_N^1)}\dots e^{-\beta H(s_1^n,\dots,s_N^n)}$
4. linearize the exponent quadratic term: $e^{-\beta H(s_1^a,\dots,s_N^a)}e^{-\beta H(s_1^b,\dots,s_N^b)}=e^{-\beta H(s_1^a,\dots,s_N^a,s_1^b,\dots,s_N^b)}$
5. integrate out the Gaussian variables: $Z^n=\int \prod_{a=1}^n \prod_{i=1}^N \frac{ds_i^a}{\sqrt{2\pi}}e^{-\frac{1}{2}(\sum_{a=1}^n \sum_{i=1}^N s_i^a)^2}e^{-\beta \sum_{a=1}^n H(s_1^a,\dots,s_N^a)}$
6. Laplace approximation: $Z^n=\int \prod_{a=1}^n \prod_{i=1}^N \frac{ds_i^a}{\sqrt{2\pi}}e^{-\frac{1}{2}(\sum_{a=1}^n \sum_{i=1}^N s_i^a)^2}e^{-\beta \sum_{a=1}^n H(s_1^a,\dots,s_N^a)}\approx \int \prod_{a=1}^n \prod_{i=1}^N \frac{ds_i^a}{\sqrt{2\pi}}e^{-\frac{1}{2}(\sum_{a=1}^n \sum_{i=1}^N s_i^a)^2}e^{-\beta \sum_{a=1}^n H(s_1^a,\dots,s_N^a)}e^{-\frac{\beta^2}{2}\sum_{a,b=1}^n \sum_{i,j=1}^N \lang s_i^a s_j^b \rang H_{ij}}$


### Replica Symmetry Breaking

<!-- Detailed derivation is to be continued -->

The replica method is a technique to deal with quenched disorder in statistical physics, such as in spin glass models. The idea is to compute the average of the logarithm of the partition function by introducing replicas of the system and taking the limit of zero replicas. The partition function of n replicas can be written as:

$$Z^n = \int \prod_{i<j} dJ_{ij} P(J_{ij}) \sum_{\{\sigma\}} \exp\left(\beta \sum_a \sum_{i<j} J_{ij} \sigma_i^a \sigma_j^a + \beta h \sum_a \sum_i \sigma_i^a\right)$$

where $\sigma_i^a$ is the spin variable of the $i$-th site and the $a$-th replica, $J_{ij}$ is the random coupling between sites $i$ and $j$, $P(J_{ij})$ is the probability distribution of the couplings, $\beta$ is the inverse temperature, and $h$ is the external magnetic field.

The replica method assumes that there is a unique analytic function that interpolates the values of $Z^n$ for integer $n$, and that this function can be analytically continued to real values of $n$. Then, one can write:

$$\lim_{n\to 0} \frac{Z^n - 1}{n} = \log Z$$

The average free energy per spin can then be obtained by:

$$f = -\lim_{n\to 0} \frac{1}{\beta n N} \log Z^n$$

where $N$ is the number of spins in the system.

The main difficulty in applying the replica method is to find a suitable representation for the order parameter that describes the correlations between replicas. This order parameter is usually an $n\times n$ symmetric matrix $Q_{ab}$, where $Q_{ab}$ is the overlap between replicas $a$ and $b$, defined as:

$$Q_{ab} = \frac{1}{N} \sum_i \sigma_i^a \sigma_i^b$$

The matrix $Q_{ab}$ encodes the structure of the phase space of the system, and how it is partitioned into different states or clusters.

The simplest assumption for the matrix $Q_{ab}$ is that it is invariant under permutations of replicas, meaning that all replicas are equivalent. This is called the replica symmetric (RS) Ansatz, and it implies that:

$$Q_{aa} = 0$$
$$Q_{ab} = q \quad a\neq b$$

where $q$ is a constant that measures the average overlap between replicas.

However, it turns out that the RS Ansatz is not valid for some systems, such as the Sherrington-Kirkpatrick (SK) model of spin glasses. In these systems, there are many metastable states that are separated by large energy barriers, and different replicas can explore different regions of phase space. This leads to a more complicated structure for the matrix $Q_{ab}$, which requires breaking replica symmetry.

Replica symmetry breaking (RSB) is a way to generalize the RS Ansatz by allowing for different values of overlaps between replicas, depending on how they are grouped or clustered. The most general form of RSB is called full RSB, and it involves an infinite hierarchy of breaking levels. However, for some systems, such as the SK model, a simpler form of RSB is sufficient, called one-step RSB (1-RSB).

In 1-RSB, one divides the replicas into $n/m$ groups of size $m$, where $m$ is a real parameter between 0 and 1. Then, one assumes that:

$$Q_{aa} = 0$$
$$Q_{ab} = q_1 \quad a,b \in \text{same group}$$
$$Q_{ab} = q_0 \quad a,b \in \text{different groups}$$

where $q_1 > q_0$ are constants that measure the intra-group and inter-group overlaps, respectively.

The 1-RSB Ansatz captures the idea that there are clusters of states that have a higher overlap within them than between them. The parameter $m$ controls how probable it is to find two replicas in the same cluster.

Using the 1-RSB Ansatz, one can compute the free energy per spin as:

$$f(q_1,q_0,m) = -\frac{\beta}{4}(1 + mq_0^2 + (1-m)q_1^2 - 2q_1) -\frac{1}{m\beta}\int Dz \log\left[\int Dy\sqrt{\frac{q_1-q_0}{2\pi}} \exp\left(-\frac{y^2}{2(q_1-q_0)}\right) (2\cosh(\beta(z+y)))^m\right]$$

where $Dz = dz \sqrt{\frac{q_0}{2\pi}} \exp\left(-\frac{z^2}{2q_0}\right)$ is a Gaussian measure.

The self-consistency equations for $q_1$, $q_0$, and $m$ can be obtained by extremizing the free energy with respect to these parameters. The solution of these equations gives the correct description of the low-temperature phase of the SK model, as proven by Parisi [6].


A general procedure of the replica method is as follows: 

![Image](https://pic4.zhimg.com/80/v2-64b8d88b05cce1ea88ce05214cd59fd7.png)

(taken from [Boazbarak](https://www.boazbarak.org/Papers/replica.pdf))

> The crazy part of Replica method is that: We want to calculate $\log Z$, but turns out be evaluating a diferent value, and they equals each other magically! (for only SK model, no proof yet for other methods)

## Cavity Method and 2-layer Perceptron

### Cavity Method

The cavity method is a technique to compute the average of a function of a random variable in a large system, by using a recursive equation that describes the correlations between the variable and its neighbors. It is used in statistical physics to compute the free energy of spin glass models, and in computer science to analyze the performance of message passing algorithms on random graphs.

### 大偏差理论

ML algorithms don't stop at the saddle points that exists indepdently, but stay on the large (although it might seem rare) basins of attraction.

Subdominant dense clusters allow for simple learning and high computational performance in Neural netowkrs with dicere synaps.

## Quantum Many-body Physics Introduction

### Quantum Many-body Physics

> Quantum many-body physics is the study of the behavior of systems made of many interacting particles, where quantum mechanics plays an essential role. It is used to describe a wide range of physical phenomena, from the behavior of electrons in metals and semiconductors, to the superfluidity of liquid helium, the Bose-Einstein condensation of ultracold atoms, and the superconductivity of certain materials at low temperatures.

### Quantum Many-body Problem

> The quantum many-body problem is the problem of finding the ground state of a quantum system with many interacting particles. It is one of the most challenging problems in physics, due to the exponential complexity of the Hilbert space of the system. The difficulty of the problem depends on the dimensionality of the system, the type of interactions, and the symmetries of the Hamiltonian.

Define

$$
\mathcal{H} = \sum_{i=1}^N \frac{p_i^2}{2m} + \sum_{i<j} V(r_i - r_j)
$$

where $r_i$ is the position of the $i$-th particle, $p_i$ is its momentum, $m$ is its mass, and $V(r_i - r_j)$ is the interaction potential between particles $i$ and $j$.

The ground state of the system is the state with the lowest energy, and it is the state that the system will tend to occupy at zero temperature.

### Quantum Many-body Problem in 1D

> In one dimension, the quantum many-body problem can be solved exactly using the Bethe ansatz. The Bethe ansatz is a method to find the eigenstates of a system with integrable Hamiltonian, by writing them as a linear combination of plane waves. It was first introduced by Hans Bethe in 1931 to solve the Heisenberg model of ferromagnetism, and it was later generalized to other models such as the Hubbard model of interacting electrons.

---

### Quantum Many-body System introduction

> A quantum many-body system is a system with many interacting particles, where quantum mechanics plays an essential role. It is used to describe a wide range of physical phenomena, from the behavior of electrons in metals and semiconductors, to the superfluidity of liquid helium, the Bose-Einstein condensation of ultracold atoms, and the superconductivity of certain materials at low temperatures.

- 铁基超导体
- 锰氧化物

competing order: different order parameters compete with each other.

Macroscopic quantum phenomena: superconductivity, superfluidity, Bose-Einstein condensation.

Nobel Prize related: 1972, 2003, 2016, 2017, 2019

Bosion: superfluidity, Bose-Einstein condensation

Fermion: superconductivity

Exponential complexity of the Hilbert space of the system: $2^N$, where $N\thicksim 10^{23}$.

### Superconductivity

The superconductivity can be defined as:

$$
\begin{aligned}
\text{Meissner effect:} &\quad \text{perfect diamagnetism} \\
\text{Zero resistance:} &\quad \text{perfect conductivity} \\
\text{Cooper pair:} &\quad \text{bound state of two electrons} \\
\text{BCS theory:} &\quad \text{Cooper pair condensation}
\end{aligned}
$$

> During the writing of this note, LK-99, a material that claims to have superconductivity, is attracting public attention.
> 
> Can theory really guide the dicovery of novel material? The answer seems to be *NO* until now.

Experiment facts:

- $T_c$ is the critical temperature of the superconducting phase transition.
- Specific heat jump at $T_c$.
- isotopic effect: $T_c \propto M^{-\alpha}$, where $M$ is the mass of the atom, and $\alpha \thicksim 0.5$.

Theory explaining isotopic effect: BCS theory.

BCS theory basics:

- Cooper pair: bound state of two electrons.
- Cooper pair condensation: the ground state of the system is a condensate of Cooper pairs.
- BCS wavefunction: $\ket{\Psi} = \prod_{k>0} (u_k + v_k c_k^\dagger c_{-k}^\dagger) \ket{0}$, where $c_k^\dagger$ is the creation operator of an electron with momentum $k$, and $u_k$ and $v_k$ are the coefficients of the wavefunction.
- BCS Hamiltonian: $H = \sum_k \xi_k c_k^\dagger c_k + \sum_{k,k'} V_{kk'} c_k^\dagger c_{-k}^\dagger c_{-k'} c_{k'}$, where $\xi_k$ is the kinetic energy of an electron with momentum $k$, and $V_{kk'}$ is the interaction potential between electrons with momenta $k$ and $k'$.
- BCS ground state: $\ket{\Psi} = \prod_{k>0} (u_k + v_k c_k^\dagger c_{-k}^\dagger) \ket{0}$, where $u_k$ and $v_k$ are the coefficients of the wavefunction.
- BCS gap equation: $\Delta_k = \sum_{k'} V_{kk'} \frac{\Delta_{k'}}{2E_{k'}} \tanh\left(\frac{\beta E_{k'}}{2}\right)$, where $\Delta_k$ is the gap parameter, $E_k = \sqrt{\xi_k^2 + \Delta_k^2}$ is the quasiparticle energy, and $\beta = 1/k_B T$ is the inverse temperature.
- The critical temperature is given by: $k_B T_c = 1.14 \omega_D e^{-1/\lambda}$, where $\omega_D$ is the Debye frequency and $\lambda$ is the coupling constant.

It explains the isotopic effect by :

- The mass of the atom affects the interaction potential between electrons.
- The interaction potential affects the gap parameter.
- The gap parameter affects the critical temperature.

#### BCS Theory

Three key points:

- Electrons are attracted to each other due to electron-phonon interaction.
- Electrons form Cooper pairs.
- Cooper pairs condense into a superconducting state.

##### Cooper Pair

- Definition: A Cooper pair is a pair of electrons with opposite spins and wave vectors $\pm \frac{\hbar}{2}\mathbf{k}_F$, where $\mathbf{k}_F$ is the Fermi wave vector, that are bound together by an attractive interaction mediated by phonons.

- Definition: The binding energy of a Cooper pair is the difference between the energy of two free electrons and the energy of a bound pair.

- Theorem: Under the assumptions of a weak electron-phonon coupling and a screened Coulomb interaction, the binding energy of a Cooper pair is given by:

$$
E_b = -g^2(k_F) n^2(\omega_D) V^{-1}
$$

where $g(k_F)$ is the coupling constant at zero momentum transfer, $n(\omega_D)$ is the Bose-Einstein distribution function for phonons at frequency $\omega_D$, which is the Debye frequency, and $V$ is the volume of the system.

### Superfluidity


> PWAnderson: Imagine yourself in a room full of people, each of whom is wearing a pair of roller skates. You are not wearing roller skates. You want to move across the room, but you cannot walk. What do you do? You grab onto the hand of the person nearest you, and the two of you move together across the room. Then you let go, grab onto the hand of the next person, and so on. In this way, you can move across the room without ever putting on a pair of roller skates. This is how electrons move through a superconductor.

Quantum Simulation using cold atom in optical lattice: [Nature 452, 854 (2008)](https://www.nature.com/articles/nature06838)

#### Bose-Hubbard Model for Superfluidity and Mott Insulator

define:

$$
\mathcal{H} = -t \sum_{\langle i,j \rangle} (b_i^\dagger b_j + b_j^\dagger b_i) + \frac{U}{2} \sum_i n_i (n_i - 1) - \mu \sum_i n_i
$$

where $b_i^\dagger$ is the creation operator of a boson at site $i$, $n_i = b_i^\dagger b_i$ is the number operator at site $i$, $t$ is the hopping amplitude, $U$ is the on-site interaction strength, and $\mu$ is the chemical potential.

The ground state of the system is the state with the lowest energy, and it is the state that the system will tend to occupy at zero temperature.

The ground state of the system can be either a superfluid or a Mott insulator, depending on the values of the parameters $t$, $U$, and $\mu$.

The superfluid phase is characterized by the presence of long-range phase coherence, which means that the phase of the wavefunction is the same at all sites. This allows the bosons to move freely through the lattice, and it leads to a finite superfluid density.

The Mott insulator phase is characterized by the absence of long-range phase coherence, which means that the phase of the wavefunction is different at different sites. This prevents the bosons from moving freely through the lattice, and it leads to a zero superfluid density.

Solve for the ground state of the system using the Gutzwiller ansatz, which is a variational ansatz that assumes that the ground state is a product state of the form:

$$
\ket{\Psi} = \prod_i \ket{\psi_i}
$$

where $\ket{\psi_i}$ is the state of site $i$.

The Gutzwiller ansatz is exact for the Mott insulator phase, and it gives a good approximation for the superfluid phase.

## Spin Glass: Theory and applications

### Setting

$N$ element, with property $\sigma_i$, energy $E_i(\sigma_i)$.

element will interact.

## Connections between ML and SP

>2023-07-23 09:11:11 Pan Zhang, SP, ML, QC

ML is "essentially" fitting the joint probability distribution of the data.

Simulation methods in one field can be used in the other fields.

> Do nature essentially computing?
>
> Is the universe a computer?
>
> Computation is defined as the process of transforming information from one form to another. It is a fundamental process that occurs in nature, and it is the basis of all physical phenomena.
>
> Can we mimic nature's computation?
>
> Can we harness nature's computation?

> Why there's different phase and phase transition?
>
> Free energy: Consider all possible configurations.
>
> At low T, liquid phase is possible, but with vanishing probability.
>
> Nature choose the phase with lowest free energy.
>
> Will nature fail?
>
> Super-cooled liquid: liquid phase with lower free energy than solid phase.
>
> It might rest at some local minimum.

OK, somes no solid, but super-cooled liquid and glass. (This is insane, but reasonable, solved some partial problem of me)

<!-- We claim to produce engineers than scientist, but turns out these engineering wonders happen in  -->

> More is different: exponential complexity space, but only a few phases.

> Statistical mechanics from theoretical computer science perspective: computational complexity.
>

|Task | computational complexity |
|---|---|
| Partition function | #P-complete |
| Free energy | #P-complete |
| Observable | NP-complete |
| Sampling | NP-complete |
| Optimization | NP-complete |
| Energy of one configuration | P |

Complexity ladders: P, NP, #P, PSPACE, BQP, BPP, ...

|Complexity class | Problem | Example |
|---|---|---|
| P | Polynomial  | Matrix multiplication |
| NP | Non-deterministic polynomial  | Traveling salesman problem |
| #P | Counting problem | Partition function |
| PSPACE | Polynomial space | Quantum circuit simulation |

### Statistical Physics and Statistical Inference is the same thing

Given a data set $D=\{x\}$, where each sample $x\in D$ has a label $y$. If we can learn the joint distribution $p(x,y)$ from the dataset, and generate unseen samples based on the label, then it's called **Generative Model**.
> Learn: given data $x$, label $y$, learn $p(x,y)$
> 
> Usage:
>
> 1. discrimitive task: $p(y|x)=p(x,y)/p(x)$
> 2. generative task: $p(x|y)=p(x,y)/p(y)$

This seems to be a trivial Bayesian estimation, but problem will arise when we are dealing with high dimensional distribution (i.e., $\dim{x}>>1$) , since we need to fit a high dimensional curve (which will lead to curse of dimensionality).

To deal with this, we introduce some models that give a prior distribution function $p(x)$ and learn the parameters to obtain the correct distribution.

>This seems to be a trivial practice as it makes no difference from just using NNs to replace $p(x,y)$

The loss function is the difference between $p(x)$ and sample distribution $\pi ({x})$, we minimize
$$
\mathbb{KL}(\pi||p)=\sum_{x\in D}\pi(x)\ln{\Big[\frac{\pi(x)}{p(x)}\Big]} = \lang \ln{\frac{\pi}{p}} \rang_\pi
$$

we may simplify it as
$$
\mathbb{KL}(\pi||p)=\lang \ln{\frac{\pi}{p}} \rang_\pi =\lang \ln{\pi} \rang_\pi - \lang \ln{p} \rang_\pi
$$

in most cases, we treat $\pi(x)=\frac{1}{|D|}\sum_{x'\in D}\delta(x-x')$, and all we need is to minimize $- \lang \ln{p} \rang_\pi$, which can be simplified as
$$
\mathcal{L}=- \lang \ln{p} \rang_\pi=-\frac{1}{|D|}\sum_{x\in D}\ln{p(x)}
$$

this is the **Negative Log Likelyhood** (what's this?), to minimize this is to maximize the likelyhood.

---

Another way to understand:

Given data $\bold{x}$, we want to learn the distribution $p(\bold{x})$.

Define it as a partition function:

$$
Z=p(\bold{x})=\sum_{\bold{s}} p(\bold{x}|\bold{s})p_0(\bold{s})
$$

where $\bold{s}$ is the hidden variable, $p_0(\bold{s})$ is the prior distribution of $\bold{s}$, and $p(\bold{x}|\bold{s})$ is the conditional distribution of $\bold{x}$ given $\bold{s}$.

If we given $p(\bold{x}|\bold{s})$ and $p_0(\bold{s})$, we can compute $p(\bold{x}|s)$, and then $p(\bold{x})$.

$$
p(\bold{x}|s)=\frac{p(\bold{x},s)}{p(s)}=\frac{p(\bold{x}|s)p_0(s)}{p(s)}
$$

### Information Compression

Given $y=f(x)$, we want to restore $x$ from $y$. Where $f(x)=Fx$, F is a $m \s n$ matrix, $m<n$.

In principle, we can't restore $x$ from $y$, but we can find a $x'$ that is close to $x$. And if $x$ is sparse, given enough $y$, we can restore $x$.

A phase diagram of the problem:

![Image](https://pic4.zhimg.com/80/v2-33a963afdea2f9ac8e46db8627fe1389.png)

Taken from [Krzakala et al. 2012](https://www.marcmezard.fr/wp-content/uploads/2019/03/2012-Krzakala_etal_PRX.pdf)

> How is this diagram related to physics?
>
> Why phase diagram plays central role in physics?

### Why NNs?

Many more parameter efficient methods than NN, but NN is most suitable for GPU.

> 魑魅魍魉 Chi Mei Wang Liang
>
> The difficulty of Generative > Discriminative

### What I cannot Create, I do not Understand

> Quantum Mechanics want to find $p(x)=\frac{|\Psi(x)|^2}{\int dx |\Psi(x)|^2}$

> Statistical Physics want to find $p(x)=\frac{e^{-\beta E(x)}}{\int dx e^{-\beta E(x)}}$

Parameterization of $\Psi(x)$ on Quantum hardware is not attainable. So we use Tensor Network to approximate it.

- Forward problem: Given energy, find minimum energy state.
- Inverse problem: Given state, find energy that generates it.

> Trivial case for RBM.

> Generalize ability matters. How this overcome the problem from physics?
>
> Seemly no solution yet.

Physicist attack to Learning and Inference: [An Introduction to Learning and Generalisation](https://link.springer.com/chapter/10.1007/978-94-011-1068-6_6)

> GPU is all you need: Current Boom in ML is due to GPU. NNs are not the best, but the most suitable for GPU.
>
> Success of ML = Data + Algorithms + NNs + GPU

> Given that computing hardware is so important, what is the currrent solution to moore law failure?

- Quantum Computing
  
> "Nature isn't classical, dammit, and if you want to make a simulation of nature, you'd better make it quantum mechanical, and by golly it's a wonderful problem, because it doesn't look so easy." - Richard Feynman

- Neuromorphic Computing

> "The brain is a computer made of meat." - Marvin Minsky

- Optical Computing

> "The future of computing is light." - John Hennessy

- DNA Computing

> "DNA is like a computer program but far, far more advanced than any software ever created." - Bill Gates

> The above are analog computing, which have inherent problems of noise and precision. But the future of AGI might be analog computing.

### Quantum Supremacy

> Quantum Supremacy is the demonstration of a quantum computer performing a calculation that is beyond the reach of the most powerful supercomputers today.

Ising model with complex-number interaction.

Google: 10,000 years(Summit)-> 200 seconds(Sycamore)

The quantum state in the random circuit sampling problem is a superposition of all possible states, and it is given by:

$$
\ket{\psi} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} \ket{x}
$$

where $n$ is the number of qubits, and $\ket{x}$ is the computational basis state with binary representation $x$.

We apply random single-qubit gates to the initial state, and we measure the final state in the computational basis. The probability of obtaining a given outcome $x$ is given by:

$$
p(x) = |\bra{x} U \ket{\psi}|^2
$$

where $U$ is the unitary operator that represents the random circuit.

In the paper of Quantum Supremacy using a programmable superconducting processor, we have a circuit with 53 qubits, the classical computation cost can be estimated as:

$$
2^{53} \s 4 \times 4
$$

in storage and computation.

> Tensornetwork: solve the problem on a hardware.

### Statistical Physics of Learning

How Ising model be used for learning?

Ising model can be defined :

$$
\mathcal{H} = -\sum_{i<j} J_{ij} s_i s_j
$$

where $s_i$ is the spin of the $i$-th particle, and $J_{ij}$ is the interaction strength between particles $i$ and $j$.

> Energy function must be Polynomial- computable. Otherwise, this model won't work. (But why)

It satisfies boltzmann distribution:

$$
p(s) = \frac{e^{-\beta \mathcal{H}(s)}}{Z}
$$

where $\beta$ is the inverse temperature, and $Z$ is the partition function.

> Why boltzmann distribution?

Consider $m$ microstates, with $n$ spins.

We want to maximize the entropy:

$$
S = -\sum_{s} p(s) \ln{p(s)}
$$

subject to the constraints:

$$
<s_i> = \frac{1}{m} \sum_{s} s_i p(s)\\
<s_is_j> = \frac{1}{m} \sum_{s} s_i s_j p(s)
$$

> Which means the two variable equals the sample average.

Using Lagrange multiplier, we have:

$$
\mathcal{L} = -\sum_{s} p(s) \ln{p(s)} + \alpha \Big(\sum_{s} p(s) - 1\Big) 
+ \sum_{i} \beta_i \Big(\sum_{s} s_i p(s) - \lang s_i \rang\Big)
+ \sum_{i<j} \beta_{ij} \Big(\sum_{s} s_i s_j p(s) - \lang s_i s_j \rang\Big)
$$

where $\alpha$, $\beta_i$, and $\beta_{ij}$ are the Lagrange multipliers.

Solve for $p(s)$, we take derivative of $\mathcal{L}$ with respect to $p(s)$, and set it to zero:

$$
\frac{\partial \mathcal{L}}{\partial p(s)} = -\ln{p(s)} - 1 + \alpha + \sum_{i} \beta_i s_i + \sum_{i<j} \beta_{ij} s_i s_j = 0
$$

which gives:

$$
p(s) = \frac{e^{-\alpha - \sum_{i} \beta_i s_i - \sum_{i<j} \beta_{ij} s_i s_j}}{Z}
$$

where $Z$ is the partition function.

The Lagrange multipliers then be solved as:

$$
\alpha = -\ln{Z} - \sum_{i} \beta_i \lang s_i \rang - \sum_{i<j} \beta_{ij} \lang s_i s_j \rang\\
\beta_i = \frac{1}{\beta} \ln{\frac{\lang s_i \rang}{1 - \lang s_i \rang}}\\
\beta_{ij} = \frac{1}{\beta} \ln{\frac{\lang s_i s_j \rang}{1 - \lang s_i s_j \rang}}
$$

where $\beta$ is the inverse temperature.

> That's why we use Boltzmann distribution.

### Curie-Weiss Model

> Curie-Weiss model is a mean-field model of ferromagnetism, which was introduced by Pierre Curie and Pierre Weiss in 1906. It is used to describe the behavior of ferromagnetic materials above the Curie temperature, where the magnetic moments of the atoms are randomly oriented. The model assumes that the magnetic moments are aligned in the same direction, and it predicts that the magnetization of the material is proportional to the applied magnetic field.

Energy:

$$
\mathcal{H} = -\frac{1}{2} \sum_{i<j} Js_i s_j 
$$

> This contains only 1 parameter, $J$.

The solution is

$$
m = \tanh{(\beta (n-1) J m)}
$$

where $m$ is the magnetization, $n$ is the number of particles, $J$ is the interaction strength, and $\beta$ is the inverse temperature.

This equation has two solutions:

> 1 parameter saves 2 data points.

```
<!-- why? -->
```

###  Network Phase Diagram

![Image](https://pic4.zhimg.com/80/v2-822fdf7e141228673104056b90f26ada.png)

Taken from [Dreaming Neural Networks: Forgetting spurious memories and reinforcing pure ones](https://arxiv.org/pdf/1810.12217.pdf)

To learn from data, the weights of the network weights $J_{ij}$ are chosen as:

$$
J_{ij} = \argmin_{J_{ij}} \Big(-\beta \sum_{\mu=1}^m \sum_{i<j} x_i^\mu x_j^\mu J_{ij} - \log Z\Big)
$$

Which is derived from the maximum likelyhood estimation.

To minimize it, we take derivative of it with respect to $J_{ij}$:

$$
\frac{\partial \mathcal{L}}{\partial J_{ij}} = -\beta \sum_{\mu=1}^m x_i^\mu x_j^\mu + \frac{\partial \log Z}{\partial J_{ij}}\\
= -\beta \sum_{\mu=1}^m x_i^\mu x_j^\mu + \beta \sum_{\mu=1}^m \lang x_i^\mu x_j^\mu \rang_J\\
= -\beta ( \lang x_i x_j \rang_\text{data} - \lang x_i x_j \rang_J)
$$

where $\lang x_i x_j \rang_\text{data}$ is the average of $x_i x_j$ over the data, and $\lang x_i x_j \rang_J$ is the average of $x_i x_j$ over the distribution defined by the weights $J_{ij}$.

> This is inverse Ising problem. Because we want to find $J_{ij}$ from $x_i$. i.e. design Hamiltonian to fit the data.

>Phase diagram is so cool, it has been applied in NN like here:
![Image](https://pic4.zhimg.com/80/v2-bd0f2e85916efbdbe37afb8c8e5ef0a3.png)
Taken from [Phase diagram for two-layer ReLU neural networks and infinite-width limit](https://arxiv.org/pdf/2007.07497.pdf)

> The above model only considers two body interaction, but in reality, there's higher order interaction. (Not discussed yet)

Consider Ising model with hidden variable:

$$
E = \sum_{i, a}s_i\sigma_a J_{ia} + \sum_{i<j} s_i s_j J_{ij} + \sum_{a<b} \sigma_a \sigma_b J_{ab}
$$

where $s_i$ is the spin of the $i$-th particle, $\sigma_a$ is the spin of the $a$-th hidden variable, $J_{ia}$ is the interaction strength between particle $i$ and hidden variable $a$, $J_{ij}$ is the interaction strength between particles $i$ and $j$, and $J_{ab}$ is the interaction strength between hidden variables $a$ and $b$.

But we can only observe $s_i$, so we need to marginalize over $\sigma_a$:

$$
p(s) = \sum_{\sigma} p(s,\sigma) = \sum_{\sigma} \frac{e^{-\beta E(s,\sigma)}}{Z}
$$

The loss is

$$
\mathcal{L} = \sum_{\mu=1}^m \log \sum_{\sigma} e^{-\beta E(x^\mu,\sigma)}/Z
$$

> This is the same as the above.

Similar to the above, we take derivative of it with respect to $J_{ia}$:

$$
\frac{\partial \mathcal{L}}{\partial J_{ia}} = -\beta \sum_{\mu=1}^m x_i^\mu \sigma_a^\mu + \beta \sum_{\mu=1}^m \lang x_i^\mu \sigma_a^\mu \rang_J\\
= -\beta ( \lang x_i \sigma_a \rang_\text{data + hidden} - \lang x_i \sigma_a \rang_J)
$$

> This is slow using MCMC. Hinton proposed a faster way: RBM. Which is a bipartite graph where hidden variable is independent of each other, and visible variable is independent of each other too.

Hinton proposed Contrastive Divergence (CD) algorithm to solve this problem.

### Implicit vs Explicit Generative Models

Given a parameterized distribution $p_\theta(x)$, if we can explicitly compute the probability of a given sample $x$, then it is called an explicit generative model. Otherwise, it is called an implicit generative model.

> Is RBM explicit or implicit?

Since $p_\theta(x) = \frac{e^{-\beta E(x)}}{Z}$, we cannot explicitly compute the probability of a given sample $x$, so it is an implicit generative model.

> Is Flow model explicit or implicit?

Since $p_\theta(x) = p_0(z) \prod_{i=1}^n f_i(z_i)$, we can explicitly compute the probability of a given sample $x$, so it is an explicit generative model.

> Is VAE explicit or implicit?

Since $p_\theta(x) = \int dz p_\theta(x|z) p_0(z)$, we cannot explicitly compute the probability of a given sample $x$, so it is an implicit generative model.

> Is Autoregressive model explicit or implicit?

Since $p_\theta(x) = \prod_{i=1}^n p_\theta(x_i|x_{<i})$, we can explicitly compute the probability of a given sample $x$, so it is an explicit generative model.

GPT uses autoregressive model, so it is an explicit generative model.

> Should the parameter be shared within the model?

---

> Hong Zhao, XMU, 2023-07-24 09:00:21

> "Statistical Physics is not of much use in Machine Learning, Statistical Physics maximize the entropy, but ML minimize it."

> Serious? 

### Statistical Physics or Statistics?

Main topics in Statistical Physics:

- Boltzmann distribution
- Maxwell-Boltzmann distribution
- Darwin-Fowler distribution
- Gibbs distribution

> Statistical Physics is yet to be fully developed compared to classical mechanics.
No clear boundaries of application between these theories

> No physics theory is possible to describe the world we mainly see. That's why ML rises.

> 水流，天上的云，吹刮的风，身边的一切都处在远离平衡态的状态。

Statistics have the following topics:

- Random variable
- Probability distribution

### Formal Definition of Statistical Physics

Given data $x = (x_1, x_2, \dots, x_n, p_1, p_2, \dots, p_n)$ in $\Gamma$
, where $x_i$ is the coordinate of the $i$-th particle, $p_i$ is the momentum of the $i$-th particle, $n$ is the number of particles.

(Micro-canonical ensemble) The probability of a given microstate $x$ is given by:

$$
p(x) = \frac{1}{\Omega(E)} \delta(E - E(x))
$$

where $\Omega(E)$ is the number of microstates with energy $E$, and $\delta(E - E(x))$ is the Dirac delta function.

(Canonical ensemble) The probability of a given microstate $x$ is given by:

$$
p(x) = \frac{1}{Z} e^{-\beta E(x)}
$$

where $Z$ is the partition function, $\beta$ is the inverse temperature, and $E(x)$ is the energy of the microstate $x$.

(Macro-canonical ensemble) The probability of a given microstate $x$ is given by:

$$
p(x) = \frac{1}{\Xi} e^{-\beta E(x)} e^{-\alpha N(x)}
$$

where $\Xi$ is the grand partition function, $\beta$ is the inverse temperature, $\alpha$ is the chemical potential, $E(x)$ is the energy of the microstate $x$, and $N(x)$ is the number of particles in the microstate $x$.

(Bose-Einstein distribution) The probability of a given microstate $x$ is given by:

$$
p(x) = \frac{1}{Z} \prod_{i=1}^n \frac{1}{e^{\beta (E_i - \mu)} - 1}
$$

where $Z$ is the partition function, $\beta$ is the inverse temperature, $\mu$ is the chemical potential, $E_i$ is the energy of the $i$-th particle, and $n$ is the number of particles.

(Fermi-Dirac distribution) The probability of a given microstate $x$ is given by:

$$
p(x) = \frac{1}{Z} \prod_{i=1}^n \frac{1}{e^{\beta (E_i - \mu)} + 1}
$$

where $Z$ is the partition function, $\beta$ is the inverse temperature, $\mu$ is the chemical potential, $E_i$ is the energy of the $i$-th particle, and $n$ is the number of particles.

> Now we have a distribution over $\Gamma$, but what we see is an average over $t$. How to resolve this?
>
> Ergodicity:  average = ensemble average

Ergodicity hypothesis: the  average of a physical quantity is equal to the ensemble average.

$$
\lim_{t \to \infty} \frac{1}{t} \int_0^t A(x(t')) dt' = \lang A \rang
$$

where $A(x(t'))$ is the value of the physical quantity $A$ at  $t'$, and $\lang A \rang$ is the ensemble average of the physical quantity $A$.

Problem related to this hypothesis:

- Poincare recurrence theorem: the system will eventually return to a state arbitrarily close to its initial state.
- Loschmidt's paradox: the -reversed dynamics of a system is not the same as the original dynamics of the system.
- Boltzmann's H-theorem: the entropy of an isolated system will increase over .

Difference from statistics:

- Data generated by Hamiltonian dynamics.
- A prior distribution is given by the Hamiltonian.
- Do not assume ergodicity: Boltzman invented $\Gamma$ space, but don't know how to proceed. "Boltzmann School" (which throws away the ergodicity hypothesis), add assume dynamic mixing. (No proof yet)
- Work strictly when $N\to\infty$.
- Can only be applied to little systems.

> Why statistical physics have no mechanics in it? (i.e. no dynamic equation)
>
> Liouville's theorem: the phase space volume is conserved under Hamiltonian dynamics.
>
> But it cannot provide proof for ergodicity or equa-partition theorem.

> Consider Noisy-Dynamics?
>
> "Random School" developed by Einstein, Smoluchowski, Langevin, Fokker-Planck, Kolmogorov, etc.

Boltzmann equation:

$$
\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f = \frac{\partial f}{\partial t} = \mathcal{C}[f]
$$

where $f$ is the probability distribution function, $\mathbf{v}$ is the velocity, and $\mathcal{C}[f]$ is the collision operator.

### Learning Machine

> If the output can feedback to the system, then it's a dynamic system.

Algorithm:

- 1986: Empirical Risk Minimization (ERM): $\min_{\theta} \sum_{i=1}^m \mathcal{L}(y_i, f(x_i;\theta))$
- 1990: Structural Risk Minimization (SRM): $\min_{\theta} \sum_{i=1}^m \mathcal{L}(y_i, f(x_i;\theta)) + \lambda \Omega(\theta), \Omega(\theta) = \frac{1}{2} \theta^T \theta$
- 2000: Design Risk Minimization (DRM): A committee vote on the output of each model.
<!-- - $\min_{\theta} \sum_{i=1}^m \mathcal{L}(y_i, f(x_i;\theta)) + \lambda \Omega(\theta), \Omega(\theta) = \frac{1}{2} \theta^T \theta + \frac{1}{2} \theta^T \mathbf{K} \theta, \mathbf{K}_{ij} = k(x_i, x_j), k(x_i, x_j) = \exp{\Big(-\frac{1}{2\sigma^2} \|x_i - x_j\|^2\Big)}$ -->

### When to use linear and when non-linear?

> Can all data be linearly separable under high dimension?

> Linear transformation won't span the space dimension. But non-linear could.

<!-- This remains vagueing , my udnerstanding include the following points:

1. Non-linear maps to higher space , while linear won't
2. Non-linear functions makes the solution space wider (i.e. have many possible weights more than linear ones.
3. No clue what the gaussian kernel is doing
-->

### Occam's Razor

> 如无必要，勿增实体

> Can we identify the shape of the drum film given the sound we here?
>
> Math proof: No

Learning Machine does not obey Occam's Razor.

### NN dynamics and it's weight spectrum

![Image](https://pic4.zhimg.com/80/v2-b358b12f04ee39b907472f343436474f.png)
Taken from [Correlation Between Eigenvalue Spectra and Dynamics
of Neural Networks](https://nba.uth.tmc.edu/homepage/cnjclub/2009Fall/NecoNNSpectra2009.pdf)

The main contributions of the paper are:

- It provides a clear criterion to distinguish between the chaos phase and the memory phase in asymmetric neural networks with associative memories, based on the eigenvalue spectra of the synaptic matrices.

- It reveals a novel phenomenon of eigenvalue splitting in the memory phase, and shows that the number and positions of the split eigenvalues are related to the number and stability of the memory attractors.

- It develops a mean-field theory to derive analytical expressions for the eigenvalue spectra and the dynamical properties of the neural network, and verifies them with numerical simulations.

### Other Model-free methods

- Phase Space Reconstruction
- Reservoir Computing Approach
- Long Short-Term Memory (LSTM)
- Time Delay Dynamical Learning

> Which one is the best?
>
> No camparison found in literature.

### Dynamic Systems: A Prior

Lorentz system:

$$
\frac{dx}{dt} = \sigma (y - x)\\
\frac{dy}{dt} = x (\rho - z) - y\\
\frac{dz}{dt} = xy - \beta z
$$

where $x$, $y$, and $z$ are the state variables, $\sigma$, $\rho$, and $\beta$ are the parameters, and $t$ is the .

It exhibits chaotic behavior for $\sigma = 10$, $\rho = 28$, and $\beta = 8/3$.

Phase space reconstruction:

$$
\mathbf{x}_i = (x_i, x_{i-\tau}, x_{i-2\tau}, \dots, x_{i-(m-1)\tau})
$$

where $x_i$ is the $i$-th data point, $\tau$ is the , and $m$ is the embedding dimension.

> Benchmark is the key to solid fundation.

> Do phase space reconstruction apply to problems without model but only data?
>
> Yes, because it's a model-free method. It can predict the future of the system.

### Reservoir Computing

> Reservoir computing is a machine learning framework for training recurrent neural networks. It was introduced by Herbert Jaeger in 2001, and it is inspired by the echo state network approach of Jürgen Schmidhuber.

Theory:

$$
\mathbf{r}(t) = \tanh{(\mathbf{W}^\text{in} \mathbf{u}(t) + \mathbf{W} \mathbf{r}(t-1))}
$$

where $\mathbf{r}(t)$ is the state vector at  $t$, $\mathbf{W}^\text{in}$ is the input weight matrix, $\mathbf{u}(t)$ is the input vector at time $t$, $\mathbf{W}$ is the recurrent weight matrix, and $\tanh$ is the hyperbolic tangent function.

Training procedure:

- Initialize $\mathbf{W}^\text{in}$ and $\mathbf{W}$ randomly.
- For each training sample $\mathbf{u}(t)$, compute $\mathbf{r}(t)$ using the above equation.
- Compute $\mathbf{W}^\text{out}$ using ridge regression: $\mathbf{W}^\text{out} = \mathbf{Y} \mathbf{R}^T (\mathbf{R} \mathbf{R}^T + \lambda \mathbf{I})^{-1}$, where $\mathbf{Y}$ is the target output matrix, $\mathbf{R}$ is the state matrix, $\lambda$ is the regularization parameter, and $\mathbf{I}$ is the identity matrix.

The target output matrix $\mathbf{Y}$ is computed as follows:

$$
\mathbf{Y} = \mathbf{R} \mathbf{W}^\text{out} (\mathbf{W}^\text{out})^T (\mathbf{R} \mathbf{R}^T + \lambda \mathbf{I})^{-1}
$$

where $\mathbf{R}$ is the state matrix, $\mathbf{W}^\text{out}$ is the output weight matrix, $\lambda$ is the regularization parameter, and $\mathbf{I}$ is the identity matrix.


- Compute $\mathbf{W}^\text{in}$ and $\mathbf{W}$ using backpropagation through : $\frac{\partial E}{\partial \mathbf{W}^\text{in}} = \frac{\partial E}{\partial \mathbf{Y}} \frac{\partial \mathbf{Y}}{\partial \mathbf{W}^\text{in}}$, and $\frac{\partial E}{\partial \mathbf{W}} = \frac{\partial E}{\partial \mathbf{Y}} \frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$, where $E$ is the error function, $\mathbf{Y}$ is the target output matrix, $\mathbf{W}^\text{in}$ is the input weight matrix, and $\mathbf{W}$ is the recurrent weight matrix.

The error function $E$ is computed as follows:

$$
E = \frac{1}{2} \sum_{t=1}^T \|\mathbf{y}(t) - \mathbf{Y}(t)\|^2
$$

where $T$ is the number of  steps, $\mathbf{y}(t)$ is the output vector at time $t$, and $\mathbf{Y}(t)$ is the target output vector at time $t$.

- Repeat the above steps until convergence.
- Use $\mathbf{W}^\text{in}$, $\mathbf{W}$, and $\mathbf{W}^\text{out}$ to predict the output.


> What is the ridge regression?
>
> Ridge regression is a regularized version of linear regression. It is used to prevent overfitting in linear regression.

### Systemetic Prediction based on Periodic Orbits

<!-- to be written -->

## Monte Carlo Methods

> Jiao Wang, XMU, 2023-07-24 13:01:59


## Quantum Annealing

Formulation of quantum annealing

> Quantum annealing is a metaheuristic for finding the global minimum of a given objective function over a given set of candidate solutions (candidate states), by a process using quantum fluctuations. Quantum annealing is used mainly for problems where the search space is discrete (combinatorial optimization problems) with many local minima; such as finding the ground state of a spin glass.

Transverse field Ising model:

$$
\mathcal{H} = - \frac{A(s)}{2} \sum_{i=1}^n \sigma_i^x - \frac{B(s)}{2} \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z
$$

where $A(s)$ is the transverse field, $B(s)$ is the longitudinal field, $\sigma_i^x$ is the Pauli-X operator on the $i$-th qubit, $\sigma_i^z$ is the Pauli-Z operator on the $i$-th qubit, $J_{ij}$ is the interaction strength between the $i$-th and $j$-th qubits, and $n$ is the number of qubits.

 evolution of the system is given by:

$$
\mathcal{H}(0) = - \frac{A(0)}{2} \sum_{i=1}^n \sigma_i^x \to \mathcal{H}(1) = - \frac{B(1)}{2} \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z
$$

where $\mathcal{H}(0)$ is the Hamiltonian at  $0$, and $\mathcal{H}(1)$ is the Hamiltonian at time $1$.

### An problem: Phase transition of Ising model

density of defect $\rho$ vs the annealling $t_s$.

> how to obtain this relationship $\rho(t_s)$?

Kibble-Zurek mechanism:

> The Kibble-Zurek mechanism (KZM) describes the non-equilibrium dynamics of a system undergoing a continuous phase transition. It was proposed by T. W. B. Kibble in 1976 and independently by Wojciech Zurek in 1985. The KZM is based on the idea that when a system is driven through a continuous phase transition at a finite rate, the system remains in thermal equilibrium during the transition. This assumption allows one to estimate the density of topological defects in the system as a function of the rate of transition.

$$
\rho(t_s) \sim \Big(\frac{t_s}{t_Q}\Big)^{\frac{1}{2}}
$$

where $t_Q$ is the quantum critical .

the theory is

$$
\log \rho = \log(\frac{1}{2\pi}\sqrt{\frac{h}{2J}}) - \frac{1}{2} \log t_s
$$

where $h$ is the transverse field, and $J$ is the interaction strength.

> This theory have no experiment parameter, but reaches good alignment with experiment data.

> This is the first case to show that quantum annealing can be used to solve a problem that is not efficiently solvable by classical computers using 5300 qubits.

Q: Include the long range interaction, will it be better?

A: Maybe, no verificaton yet.

Q: They reached $L=2000$ noise-free qubits, but under only $50\text{ns}$ annealing . (environment noise is the main problem)

![Image](https://pic4.zhimg.com/80/v2-b598aaba3dbe38eda7daec6268706267.png)

> Wikipedia is doing a great job in democrotizing knowledge.

## Dynamics of cell state transition

> Jianhua Xing, University of Pittsburgh

Mathematical considerations:

$$
\frac{d\vec{z}}{dt} = \bold{A(\vec{z})} + \xi(\vec{z},t)
$$

> why looks like this? 
>
> $\bold{A(\vec{z})}$ is the deterministic part, $\xi(\vec{z},t)$ is the stochastic part.

where $\vec{z}$ is a state vector specifying the complete internal state of the cell at a given .

The problem became how to find $\bold{A(\vec{z})}$ and $\xi(\vec{z},t)$. This can be defined as a variational problem:

$$
\min_{\bold{A(\vec{z})}, \xi(\vec{z},t)} \Big(\frac{1}{2} \int_0^T \|\frac{d\vec{z}}{dt} - \bold{A(\vec{z})} - \xi(\vec{z},t)\|^2 dt + \frac{\lambda}{2} \int_0^T \|\bold{A(\vec{z})}\|^2 dt\Big)
$$

where $\lambda$ is the regularization parameter.

We parameterize $\bold{A(\vec{z})}$ as:

$$
\bold{A(\vec{z})} = \sum_{i=1}^n \bold{A_i} \phi_i(\vec{z})
$$

where $\phi_i(\vec{z})$ is the $i$-th basis function, and $\bold{A_i}$ is the $i$-th coefficient.

$$
\phi_i(\vec{z}) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} (\vec{z} - \vec{z}_i)^T \Sigma^{-1} (\vec{z} - \vec{z}_i)}
$$

> is this form necessary as a prior knowledge?
>
> In practice, you can use other basis too.

> How is the noise handled?
>
> Here we use white noise, given enough data, we can learn it too.

where $\vec{z}_i$ is the $i$-th center, and $\Sigma$ is the covariance matrix.



We give the change of $n(\vec{z},t)$ (number density) w.r.t. the diffusion constant $D$ as:

$$
\frac{\partial n(\vec{z},t)}{\partial t} = \nabla [- \bold{J}(\vec{z}) n(\vec{z},t) + \bold{D} \nabla n(\vec{z},t)] + \int d\vec{z}' \bold{K}(\vec{z},\vec{z}') n(\vec{z}',t) 
$$

where $\bold{J}(\vec{z})$ is the drift velocity, $\bold{D}$ is the diffusion constant, and $\bold{K}(\vec{z},\vec{z}')$ is the transition rate.

> This is a Fokker-Planck equation.

A specific ansatz of the dynamical euqation is:

$$
\frac{d\bold{x}}{dt} = f(\bold{x}) + (\bold{\sigma}\bold{x}) \bold{\eta}(t)\\
\frac{\partial n(\bold{x},t)}{\partial t} = -\nabla (n\bold{f}) + \nabla \cdot (\bold{D} \nabla n) + \mu(\bold{x}) n
$$

> Have you considered 各向异性(isotropy) of the cell? 
>
> No, we work in the phase space of cell.

## Human population behavior and propagation dynamics

> Qingyan Hu, 南方科技大学，复杂流动及软物中心，统计与数据科学系

### information spreading dynamics

Information spreading dynamics contain three key elements:

- Information feature $\alpha_i$
- Network structure $\gamma$
- User attribute $f_i$

We model the information spreading dynamics as:

$$
\beta_i = \alpha_i x (1-\gamma)^{f_i(x)}
$$
where $\beta_i$ is the probability of user $i$ to spread the information, $\alpha_i$ is the information feature of user $i$, $x$ is the fraction of users who have received the information, $\gamma$ is the network structure, and $f_i(x)$ is the user attribute of user $i$.

$$
f_i(x) = \frac{1}{1+e^{-\lambda_i(x-\theta_i)}}
$$

where $\lambda_i$ and $\theta_i$ are parameters that depend on the user's attribute, such as age, education, income, etc. The paper says that this function is a logistic function that models the user's cognitive psychology. It means that the user's probability of receiving the information depends on how much the information matches their prior knowledge and beliefs.


Classical theory: social enforcement

$$
\beta_i = \alpha_i x (1-\gamma)^{f_i(x)} + \lambda \sum_{j \in \mathcal{N}_i} \beta_j
$$

where $\mathcal{N}_i$ is the set of neighbors of user $i$, and $\lambda$ is the social enforcement parameter.

> Have your theory condiserted the inhomoegeneity of different users?
>
> No, $\alpha_i$ is the same for all users. But it can be different.

Here we use

$$
f_i(x) = \frac{\ln {[\beta_i(x)/\beta_i(1)]} - \ln x}{\ln{(1-\gamma)}} + 1
$$

where $\beta_i(x)$ is the probability of user $i$ to spread the information when the fraction of users who have received the information is $x$.

We simplify the model as:

$$
f_i(x)=x^{\omega_i}
$$

where $\omega_i$ is the user attribute of user $i$.

### Disease spreading dynamics

The SIR model is a compartmental model that describes the dynamics of disease spreading in a population. It is a system of ordinary differential equations that models the number of susceptible, infected, and recovered individuals in a population over .

$$
\frac{dS}{dt} = -\beta I S\\
\frac{dI}{dt} = \beta I S - \gamma I\\
\frac{dR}{dt} = \gamma I
$$

where $S$ is the number of susceptible individuals, $I$ is the number of infected individuals, $R$ is the number of recovered individuals, $\beta$ is the infection rate, and $\gamma$ is the recovery rate.

In covid, the quarantine  $\tau$ is related to $R$ by

$$
R(\tau) \approx R_0 \frac{\tau - \bar{T}_e}{\bar{Q}\bar{T}_c}
$$

where $R_0$ is the basic reproduction number, $\bar{T}_e$ is the average incubation period, $\bar{Q}$ is the average quarantine , and $\bar{T}_c$ is the average recovery time.

The ciritical quarantine  $\tau_c$ is related to $R_0$ given infinite average degree

$$
\tau_c \approx \bar{T}_e + \frac{\bar{Q}\bar{T}_c}{R_0}
$$

> Have you considerted the spacial factor in disease spreading
>
> No, but can .


## Spiking Neural Networks

Integrate-and-fire model:

$$
C \frac{dV}{dt} = -g_L(V-E_L) + I(t)
$$

where $C$ is the membrane capacitance, $V$ is the membrane potential, $g_L$ is the leak conductance, $E_L$ is the leak reversal potential, and $I(t)$ is the input current.

The membrane potential $V$ is reset to $V_r$ when it reaches the threshold $V_\text{th}$.

$$
V \to V_r \text{ if } V \ge V_\text{th}
$$

> Spiking neural network claims to have low power usage, and are more robust to noise.

> The future of computation: Neuromorphic computing

> Trainning SNN is hard, because it's not differentiable, so backpropagation cannot be used.

> Mimic the brain activity to learn is enough, we don't need to know how the brain works. 

Training methods:

| Method | Pros | Cons |
| --- | --- | --- |
| ANN-SNN conversion | Easy to implement | Low accuracy |
| Approximate gradient | High accuracy | High complexity |
| Neuro synaptic dynamics | High accuracy | Low performance |

> They allpy STDP to train the network.

STDP is a biological process that adjusts the strength of connections between neurons in the brain. The process adjusts the connection strengths based on the relative timing of the input signals to pairs of neurons. The STDP process partially explains the activity-dependent development of nervous systems, especially with regard to long-term potentiation and synaptic plasticity.

$$
\Delta w_{ij} = \begin{cases}
A_+ e^{-\frac{\Delta t}{\tau_+}} & \text{if } \Delta t > 0\\
-A_- e^{-\frac{\Delta t}{\tau_-}} & \text{if } \Delta t < 0
\end{cases}
$$

where $\Delta w_{ij}$ is the change in the synaptic weight between the $i$-th and $j$-th neurons, $A_+$ is the amplitude of the positive change, $A_-$ is the amplitude of the negative change, $\tau_+$ is the  constant of the positive change, $\tau_-$ is the time constant of the negative change, and $\Delta t$ is the time difference between the spikes of the $i$-th and $j$-th neurons.

> They proposed a new method to train SNN called DA-STDP, which is based on the dopamine reward signal.

DA-STDP is based on the following equation:

$$
\Delta w_{ij} = \begin{cases}
A_+ e^{-\frac{\Delta t}{\tau_+}} & \text{if } \Delta t > 0\\
-A_- e^{-\frac{\Delta t}{\tau_-}} & \text{if } \Delta t < 0
\end{cases} + \begin{cases}
\alpha & \text{if } \Delta t > 0\\
-\beta & \text{if } \Delta t < 0
\end{cases}
$$

where $\alpha$ is the reward for the positive change, and $\beta$ is the reward for the negative change.

> the SNN is implemented in hardware

## Probalistic inference reformulated as tensor networks

> Jin-Guo Liu, HKUST

### Reformulate probalistc inference as tensor networks

Probabilistic inference is the task of computing the posterior distribution of the hidden variables given the observed variables.

$$
p(\vec{h}|\vec{v}) = \frac{p(\vec{h},\vec{v})}{p(\vec{v})}
$$

where $\vec{h}$ is the hidden variables, and $\vec{v}$ is the observed variables.

The joint distribution of the hidden variables and the observed variables is given by:

$$
p(\vec{h},\vec{v}) = \frac{1}{Z} \prod_{i=1}^n \phi_i(\vec{h},\vec{v})
$$

where $Z$ is the partition function, and $\phi_i(\vec{h},\vec{v})$ is the potential function.

> why the joint distribution is like this?
>
> This is the definition of the joint distribution.

Graph probabilistic model: the joint distribution is given by: 

$$
p(\vec{h},\vec{v}) = \frac{1}{Z} \prod_{i=1}^n \phi_i(\vec{h}_i,\vec{v}_i) \prod_{j=1}^m \phi_j(\vec{h}_j)
$$

where $\vec{h}_i$ is the hidden variables of the $i$-th node, $\vec{v}_i$ is the observed variables of the $i$-th node, $\phi_i(\vec{h}_i,\vec{v}_i)$ is the potential function of the $i$-th node, $\phi_j(\vec{h}_j)$ is the potential function of the $j$-th edge, and $Z$ is the partition function.

> Where is the graph here?
>
> The graph is the structure of the joint distribution. 

Probabilistic Model:

```
<!-- See TensorInference.jl -->
```

## Variational Autoregressive Network

> Pan Zhang , ITP, 2023-07-28 14:03:12

Variational methods in statistical mechanics

Mean field assumptions:

$$
q(\bold{s})= \prod_{i=1}^n q_i(s_i)
$$

where $q(\bold{s})$ is the variational distribution, $q_i(s_i)$ is the variational distribution of the $i$-th spin.

Bethe ansatz 

$$
q(\bold{s})= \prod_{i=1}^n q_i(s_i) \prod_{i<j}^n \frac{q_{ij}(s_i,s_j)}{q_i(s_i)q_j(s_j)}
$$

where $q(\bold{s})$ is the variational distribution, $q_i(s_i)$ is the variational distribution of the $i$-th spin, and $q_{ij}(s_i,s_j)$ is the variational distribution of the $i$-th and $j$-th spins.

### Chemical reaction simulation with VAN

See lecture note of Yin Tang.


## Machine learning and chaos

Question:

Given

$$
\dot{x}= f(x,r)
$$

where 

$$
f(a+b)\ne f(a)+ f(b)
$$

Predict the final $x$ given the initial $x_0$.

> A simplist example is  logistic map

$$
x_{n+1} = rx_n(1-x_n)
$$

For stability, $r$ must be restricted to $[0,4]$, we found that for some $r$, the period of the system $T\to \infty$.

Simple code to replicate it
```
def logistic(a):
    x = [0.3]
    for i in range(400):
        x.append(a * x[-1] * (1 - x[-1]))
    return x[-100:]

for a in np.linspace(2.0, 4.0, 1000):
    x = logistic(a)
    plt.plot([a]*len(x), x, "c.", markersize=0.1)

plt.xlabel("r")
plt.ylabel("x_f")
plt.show()
```

![Image](https://pic4.zhimg.com/80/v2-444cf9b4023829e83ad7bedb2838c505.png)

Another example is the Lorenz system:

$$
\frac{dx}{dt} = \sigma (y - x)\\
\frac{dy}{dt} = x (\rho - z) - y\\
\frac{dz}{dt} = xy - \beta z
$$

where $x$, $y$, and $z$ are the state variables, $\sigma$, $\rho$, and $\beta$ are the parameters, and $t$ is the .

Fractional ordinary differential equation (FODE): 

$$
\frac{d^\alpha x}{dt^\alpha} = f(x,t)
$$

where $\alpha$ is the order of the fractional derivative, $x$ is the state variable, $t$ is the , and $f(x,t)$ is the function of the state variable and the time.

For example, the fractional derivative of order $\alpha$ of the function $x(t)$ is given by:

$$
\frac{d^\alpha x}{dt^\alpha} = \frac{1}{\Gamma(1-\alpha)} \frac{d}{dt} \int_0^t \frac{x(t')}{(t-t')^\alpha} dt'
$$

where $\Gamma$ is the gamma function.

> What is the physical meaning of the fractional derivative?
>
> It is the memory of the system. The fractional derivative of order $\alpha$ is the memory of the system of length $\alpha$.


### Feature of chaos

- Sensitivity to initial conditions

This is the property that the system is sensitive to initial conditions.

Mathematical definition:

$$
\exists \epsilon > 0, \forall x \in \mathcal{O}, \exists y \in \mathcal{O}, \exists n \in \mathbb{N}, \text{ such that } \|f^n(x) - f^n(y)\| > \epsilon
$$

Lynapunov exponent: 

$$
\lambda = \lim_{t\to\infty} \frac{1}{t} \log \Big|\frac{dx(t)}{dx(0)}\Big|
$$

where $\lambda$ is the Lynapunov exponent, $x(t)$ is the state variable at  $t$, and $x(0)$ is the state variable at time $0$.

Example: logistic map with $r=4$ and $x_0=0.2$ has $\lambda = 0.69$.

If $\lambda > 0$, the system is chaotic.

- Topological mixing

This is the property that the system will eventually reach any state in the phase space. 

Matheatical definition:

$$
\forall U, V \in \mathcal{O}, \exists N \in \mathbb{N}, \forall n \ge N, f^n(U) \cap V \ne \emptyset
$$

where $\mathcal{O}$ is the phase space, $U$ and $V$ are two open sets in the phase space, $N$ is a natural number, $n$ is a natural number, $f^n(U)$ is the $n$-th iteration of the set $U$, and $\emptyset$ is the empty set.

- Dense periodic orbits

This is the property that the system has infinite periodic orbits.

Mathematical definition:

$$
\forall x \in \mathcal{O}, \forall \epsilon > 0, \exists y \in \mathcal{O}, \exists n \in \mathbb{N}, \text{ such that } \|f^n(x) - y\| < \epsilon
$$

where $\mathcal{O}$ is the phase space, $x$ is a point in the phase space, $\epsilon$ is a positive real number, $y$ is a point in the phase space, $n$ is a natural number, $f^n(x)$ is the $n$-th iteration of the point $x$, and $\|f^n(x) - y\|$ is the distance between the $n$-th iteration of the point $x$ and the point $y$.

- Sensitive to initial conditions


### How ML is doing better than traditional methods?

Two questions in chaos study, under the condition that the system dynamics is unknown(model-free):

- predict chaos evolution

- infer bifurcation diagram

> Essence of statistical learning: i.i.d. assumption


### Chaos synchronization

Chaos synchronization: given a coupled chaotic oscillator, we want to synchronize the two oscillators.

Oscillator 1:

$$
\frac{dx_1}{dt} = f(x_1) + \epsilon (x_2 - x_1)
$$

Oscillator 2:

$$
\frac{dx_2}{dt} = f(x_2) + \epsilon (x_1 - x_2)
$$

where $x_1$ is the state variable of oscillator 1, $x_2$ is the state variable of oscillator 2, $f(x_1)$ is the dynamics of oscillator 1, $f(x_2)$ is the dynamics of oscillator 2, and $\epsilon$ is the coupling strength. This coupling is called linear coupling, or diffusive coupling, because the coupling term is proportional to the difference between the two oscillators.

> How is this related to diffusion?
>
> The coupling term is proportional to the difference between the two oscillators, which is similar to the diffusion term in the diffusion equation: $\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}$.

- complete synchronization

$$
(x_1,y_1,z_1) = (x_2,y_2,z_2), \epsilon > \epsilon_c
$$

which means that the two oscillators have the same state variables when the coupling strength is greater than the critical coupling strength.

- phase synchronization

$$
\theta_1 = \theta_2 +c, \epsilon > \epsilon_p
$$

where $\theta_1$ is the phase of oscillator 1, $\theta_2$ is the phase of oscillator 2, $c$ is a constant, and $\epsilon_p$ is the critical coupling strength. This means that the two oscillators have the same phase when the coupling strength is greater than the critical coupling strength.

> How is the pahse $\theta$ defined ?
>
> $\theta$ is the angle of the oscillator in the phase space. For example, for the logistic map, $\theta$ is the angle of the point $(x_n, x_{n+1})$ in the phase space.

- Generalized synchronization

$$
x_1 = G(x_2), \epsilon > \epsilon_g
$$

where $x_1$ is the state variable of oscillator 1, $x_2$ is the state variable of oscillator 2, $G$ is a function, and $\epsilon_g$ is the critical coupling strength. This means that the two oscillators have the same state variables when the coupling strength is greater than the critical coupling strength.

> Science is good, but engineering will make it great.

> Why is chaos synchronization important?
>
> Real systems are coupled, and chaos synchronization is a way to model the coupling.

Can we predict the synchronization point $\epsilon_c$ given some $x_{1,2}(\epsilon_i,t), i\in\{1,2,3\}$?

One method: apply reservior computing.

To produce predictions for an unknow non-linear system is in principle difficult. Here we review some methods to predict chaotic system based on data/model, trying to grasp the essence of chaos.

## Definition and Characterization of Chaos

Question:

Given

$$
\dot{x}= f(x,r)
$$

where 

$$
f(a+b)\ne f(a)+ f(b)
$$

Predict the final $x$ given the initial $x_0$.

> A simplist example is  logistic map

$$
x_{n+1} = rx_n(1-x_n)
$$

For stability, $r$ must be restricted to $[0,4]$, we found that for some $r$, the period of the system $T\to \infty$.

Simple code to replicate it
```
def logistic(a):
    x = [0.3]
    for i in range(400):
        x.append(a * x[-1] * (1 - x[-1]))
    return x[-100:]

for a in np.linspace(2.0, 4.0, 1000):
    x = logistic(a)
    plt.plot([a]*len(x), x, "c.", markersize=0.1)

plt.xlabel("r")
plt.ylabel("x_f")
plt.show()
```

![Image](https://pic4.zhimg.com/80/v2-444cf9b4023829e83ad7bedb2838c505.png)

Another example is the Lorenz system:

$$
\frac{dx}{dt} = \sigma (y - x)\\
\frac{dy}{dt} = x (\rho - z) - y\\
\frac{dz}{dt} = xy - \beta z
$$

where $x$, $y$, and $z$ are the state variables, $\sigma$, $\rho$, and $\beta$ are the parameters, and $t$ is the time. The Lorenz system has chaotic solutions for some parameter values and initial conditions. In particular, for $\sigma = 10$, $\beta = 8/3$, and $\rho = 28$, the Lorenz system exhibits chaotic solutions for many initial conditions.


![Image](https://pic4.zhimg.com/80/v2-e523749ccb4c2f6da0468772c0159b18.png)
### Feature of chaos

- Sensitivity to initial conditions

This is the property that the system is sensitive to initial conditions.

Mathematical definition:

$$
\exists \epsilon > 0, \forall x \in \mathcal{O}, \exists y \in \mathcal{O}, \exists n \in \mathbb{N}, \text{ such that } \|f^n(x) - f^n(y)\| > \epsilon
$$

Lynapunov exponent:

$$
\lambda = \lim_{t\to\infty} \frac{1}{t} \log \Big|\frac{dx(t)}{dx(0)}\Big|
$$

where $\lambda$ is the Lynapunov exponent, $x(t)$ is the state variable at  $t$, and $x(0)$ is the state variable at time $0$.

Example: logistic map with $r=4$ and $x_0=0.2$ has $\lambda = 0.69$.

If $\lambda > 0$, the system is chaotic.

- Topological mixing

This is the property that the system will eventually reach any state in the phase space.

Matheatical definition:

$$
\forall U, V \in \mathcal{O}, \exists N \in \mathbb{N}, \forall n \ge N, f^n(U) \cap V \ne \emptyset
$$

where $\mathcal{O}$ is the phase space, $U$ and $V$ are two open sets in the phase space, $N$ is a natural number, $n$ is a natural number, $f^n(U)$ is the $n$-th iteration of the set $U$, and $\emptyset$ is the empty set.

- Dense periodic orbits

This is the property that the system has infinite periodic orbits.

Mathematical definition:

$$
\forall x \in \mathcal{O}, \forall \epsilon > 0, \exists y \in \mathcal{O}, \exists n \in \mathbb{N}, \text{ such that } \|f^n(x) - y\| < \epsilon
$$

where $\mathcal{O}$ is the phase space, $x$ is a point in the phase space, $\epsilon$ is a positive real number, $y$ is a point in the phase space, $n$ is a natural number, $f^n(x)$ is the $n$-th iteration of the point $x$, and $\|f^n(x) - y\|$ is the distance between the $n$-th iteration of the point $x$ and the point $y$.

- Sensitive to initial conditions

## Prediction of Chaos

Two questions in chaos study, under the condition that the system dynamics is unknown(model-free):

- predict chaos evolution

![Image](https://pic4.zhimg.com/80/v2-ccf772c2634be76d74842dc31af4ab2e.png)
![Image](https://pic4.zhimg.com/80/v2-e523749ccb4c2f6da0468772c0159b18.png)

- infer bifurcation diagram

![Image](https://pic4.zhimg.com/80/v2-444cf9b4023829e83ad7bedb2838c505.png)

![Image](https://pic4.zhimg.com/80/v2-3390df12eec8a4b6ee63ce20e75ac67f.png)

![Image](https://pic4.zhimg.com/80/v2-92a234cb0f14d15bce1a57392a14b687.png)


A paper in 2001 use Reservoir Computing to predict chaos evolution, which is a model-free method. Here we reproduce it.

### Problem Formulation

Chaos synchronization: given a coupled chaotic oscillator, we want to synchronize the two oscillators.

Oscillator 1:

$$
\frac{dx_1}{dt} = f(x_1) + \epsilon (x_2 - x_1)
$$

Oscillator 2:

$$
\frac{dx_2}{dt} = f(x_2) + \epsilon (x_1 - x_2)
$$

where $x_1$ is the state variable of oscillator 1, $x_2$ is the state variable of oscillator 2, $f(x_1)$ is the dynamics of oscillator 1, $f(x_2)$ is the dynamics of oscillator 2, and $\epsilon$ is the coupling strength. This coupling is called linear coupling, or diffusive coupling, because the coupling term is proportional to the difference between the two oscillators.

> How is this related to diffusion?
>
> The coupling term is proportional to the difference between the two oscillators, which is similar to the diffusion term in the diffusion equation: $\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}$.

- complete synchronization

$$
(x_1,y_1,z_1) = (x_2,y_2,z_2), \epsilon > \epsilon_c
$$

which means that the two oscillators have the same state variables when the coupling strength is greater than the critical coupling strength.

- phase synchronization

$$
\theta_1 = \theta_2 +c, \epsilon > \epsilon_p
$$

where $\theta_1$ is the phase of oscillator 1, $\theta_2$ is the phase of oscillator 2, $c$ is a constant, and $\epsilon_p$ is the critical coupling strength. This means that the two oscillators have the same phase when the coupling strength is greater than the critical coupling strength.

> How is the pahse $\theta$ defined ?
>
> $\theta$ is the angle of the oscillator in the phase space. For example, for the logistic map, $\theta$ is the angle of the point $(x_n, x_{n+1})$ in the phase space.

- Generalized synchronization

$$
x_1 = G(x_2), \epsilon > \epsilon_g
$$

where $x_1$ is the state variable of oscillator 1, $x_2$ is the state variable of oscillator 2, $G$ is a function, and $\epsilon_g$ is the critical coupling strength. This means that the two oscillators have the same state variables when the coupling strength is greater than the critical coupling strength.

> Science is good, but engineering will make it great.

> Why is chaos synchronization important?
>
> Real systems are coupled, and chaos synchronization is a way to model the coupling.

Can we predict the synchronization point $\epsilon_c$ given some $x_{1,2}(\epsilon_i,t), i\in\{1,2,3\}$?

#### Generate Data

<!-- ```
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the Lorenz system
def lorenz(x, t, sigma, rho, beta):
    x1, x2, x3 = x # Unpack the state vector
    dxdt = [sigma * (x2 - x1), x1 * (rho - x3) - x2, x1 * x2 - beta * x3] # Compute the derivatives
    return dxdt

# Define the coupling function
def coupling(x1, x2, epsilon):
    return epsilon * (x1 - x2)

# Define the coupled Lorenz system
def coupled_lorenz(x, t, sigma, rho, beta, epsilon):
    x1 = x[:3] # Unpack the state vector for the first system
    x2 = x[3:] # Unpack the state vector for the second system
    dxdt = np.zeros(6) # Initialize the derivative vector
    dxdt[:3] = lorenz(x1, t, sigma, rho, beta) + coupling(x1, x2, epsilon) # Compute the derivatives for the first system with coupling
    dxdt[3:] = lorenz(x2, t, sigma, rho, beta) - coupling(x1, x2, epsilon) # Compute the derivatives for the second system with coupling
    return dxdt

# Define the parameters
sigma = 10 # Prandtl number
rho = 28 # Rayleigh number
beta = 8 / 3 # Physical dimension
epsilon_c = 0.906 # Critical coupling strength for synchronization
T = 50 # Final time
dt = 0.01 # Time step
N = int(T / dt) + 1 # Number of time points

# Define the initial conditions
x10 = 1 # Initial value for x1 in the first system
x20 = 1 # Initial value for x2 in the first system
x30 = 1 # Initial value for x3 in the first system
x40 = -1 # Initial value for x1 in the second system
x50 = -1 # Initial value for x2 in the second system
x60 = -1 # Initial value for x3 in the second system

# Define the time array
t = np.linspace(0, T, N)

# Generate 10 epsilon values randomly under epsilon_c
np.random.seed(42) # Set a random seed for reproducibility
epsilons = np.random.uniform(0, epsilon_c, 10) # Generate 10 random values between 0 and epsilon_c

# Initialize an empty list to store the data of the (x_1, x_2, epsilon) pair
data_list = []

# Loop over each epsilon value
for epsilon in epsilons:
    # Solve the coupled Lorenz system using odeint with the given epsilon value
    x0 = [x10, x20, x30, x40, x50, x60] # Initial state vector
    x = odeint(coupled_lorenz, x0, t, args=(sigma, rho, beta, epsilon)) # Solve the ODEs

    # Extract the observations from the solution array
    x1 = x[:, 0] # Observation sequence for x1 in the first system
    x2 = x[:, 3] # Observation sequence for x1 in the second system

    # Append the data of the (x_1, x_2, epsilon) pair to the data list 
    data_list.append((x1[:], x2[:], epsilon)) # Use only the last values of each sequence as data points

# Convert the data list into a numpy array of shape (10, 3)
data_array = np.array(data_list)

# save the data array as a .npy file
np.save("data_array.npy", data_array)
``` -->

One data pair looks like this:

![Image](https://pic4.zhimg.com/80/v2-fb379794ab26c89d1371348e9f70ab49.png)

![Image](https://pic4.zhimg.com/80/v2-2d7f58d150992ade5b62124c505af837.png)

We can visualize the data by plotting the observations of the first system against the observations of the second system for each epsilon value.

<!-- ```
# Plot the observations of the first system against the observations of the second system for each epsilon value
for i in range(len(data_array)):
    plt.plot(data_array[i, 0], data_array[i, 1], ".", label=f"$\epsilon = {data_array[i, 2]:.3f}$", markersize=0.1)
plt.title("$x_1$ vs $x_2$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# set parameters to make the plot with high resolution
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

``` -->
<!-- 
![Image](https://pic4.zhimg.com/80/v2-7c5b27f9c914b1313fa53ab6da667569.png) -->

<!-- ![Image](https://pic4.zhimg.com/80/v2-50d38c71a1ea5ddf2b2c2d1f9b11e544.png) -->

![Image](https://pic4.zhimg.com/80/v2-b8a934d1383e0a76ca0fbbce1578eee9.png)
This displays the dynamic coupling between the two systems. This can be seen as a visualization of the phase space of the coupled system.

Another visualization look at the final state difference ($|x_1-x_2|$) of the system for each epsilon value.

<!-- ```
# Plot the time averaged state difference (|x_1-x_2|) of the system for each epsilon value
for i in range(len(data_array)):
    plt.plot(data_array[i, 2], np.mean(np.abs(data_array[i, 0] - data_array[i, 1])), ".", label=f"$\epsilon = {data_array[i, 2]:.3f}$", markersize=10)
    # plot the correlation of x_1 and x_2
    # plt.plot(data_array[i, 2], np.corrcoef(data_array[i, 0], data_array[i, 1])[0, 1], ".", label=f"$\epsilon = {data_array[i, 2]:.3f}$", markersize=10)
plt.title("Time averaged state difference")
plt.xlabel("$\epsilon$")
plt.ylabel("$|x_1-x_2|$")
plt.show()
``` -->

<!-- ![Image](https://pic4.zhimg.com/80/v2-46f4aeed2e0653e67fff8c37fd413728.png) -->

![Image](https://pic4.zhimg.com/80/v2-556c76ac745dd828f6e5bbb16feed4ba.png)

Which indicate a larger bias when $\epsilon$ is large.

> Weired right? We expect the system to synchronize!

To better understand the synchronization, we plot the coefficient between $x_1$ and $x_2$.

![Image](https://pic4.zhimg.com/80/v2-bdd9eadefc450c600b455703b075fe2e.png)

This shows that the synchronization is not perfect, but the correlation is approaching -1 when $\epsilon$ is large.

Anyway, we can use the data to train a model to predict the synchronization point $\epsilon_c$.

#### Create a Reservoir

A reservoir can be abstractly thought of as a dynamical system $x_{n+1} = f(x_n)$, where $x_n$ is the state vector at time $n$ and $f$ is the reservoir function. By supervised training, we can fit the reservoir to a given data set.

The [community](https://github.com/reservoirpy/reservoirpy) created a package to create reservoirs and apply it to dynamical system prediction.

![Image](https://pic4.zhimg.com/80/v2-d07221209dd97051c19ef2af8dc83f44.png)

Here we review one of it's [notebook](https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/3-General_Introduction_to_Reservoir_Computing.ipynb) to see how it works.

### A (very) short tutorial on Reservoir Computing

- Data format must be (time, features)
  >e.g. (1000, 3) for 3 features and 1000 time steps.
- If multiple sequences are provided, the data format must be (sequences, time, features)
   >e.g. (10, 1000, 3) for 10 sequences of 3 features and 1000 time steps.
- **Reservoir**: neurons randomly connected to their inputs and to themselves, not trainnable, randomly initialized under some constraints.
   >e.g. 100 neurons, 0.3 spectral radius, 0.5 connectivity, 0.1 input scaling, 0.1 leak rate.
- **Readout**: a decoder with a single layer of neurons, trainnable with a linear regression.
   > No backpropagation needed! See ridge regression.
- feedback: Readout neuron can be connected to the reservoir, to tame the reservoir dynamics. (An optional)
  > tame: make the reservoir dynamics more predictable.
- State vector: the reservoir state at time $t$ is the concatenation of the reservoir neurons states at time $t$.
  > For every time step, the reservoir state is a vector of length $N$, where $N$ is the number of neurons in the reservoir.
  >
  > $x_t = [x_{t,1}, x_{t,2}, ..., x_{t,N}]$, where $x_{t,i} = f(W_{in}u_t + Wx_{t-1})_i$.
  >
  > This $x_t$ will be stored for later use. It has the shape of (time, neurons). For example, if the reservoir has 30 neurons and the data has 100 time steps, then the state vector has the shape of (100, 30).

![Image](https://pic4.zhimg.com/80/v2-d630050707345d264e18f15219d3235e.png)

ESN: Echo State Network, a type of reservoir computing.

![Image](https://pic4.zhimg.com/80/v2-0f59572d40a307bb923cbf409f21cf9e.png)

Now we practice building a reservoir.

```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge
```

> This cute `>>` is a pipeline operator, which is a shorthand for `esn_model = ridge.fit(reservoir.fit(data))`. It's a syntactic sugar.

Let's first train a lorenz system.

<!-- ```python
# Import the modules we need
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import lorenz

# Generate the data of the Lorenz system
data = lorenz(1000, dt=0.01, rho=28, sigma=10, beta=8/3)

# # Plot the data in 3D
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(data[:, 0], data[:, 1], data[:, 2], color='blue')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

# Split the data into train and test sets
train_len = 800
test_len = 200
train_data = data[:train_len]
test_data = data[train_len:train_len+test_len]

# Create the reservoir
reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

# Create the model
model = reservoir >> ridge

# Train the model on the train data
model.fit(train_data[:-1], train_data[1:])
``` -->
![Image](https://pic4.zhimg.com/80/v2-37760a2ca3c7f448401878a1e8444160.png)

Visualize the prediction seperately.

<!-- ```python
# Predict the test data
prediction = model.predict(test_data[:-1])

# Plot the prediction vs the test data for 3 features using subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
for i in range(3):
    axes[i].plot(test_data[1:, i], label="Test data")
    axes[i].plot(prediction[:, i], label="Prediction")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel(f"Feature {i+1}")
    axes[i].legend()

plt.show()
``` -->
![Image](https://pic4.zhimg.com/80/v2-8ecfc3f987a2a4106849791f0cd1ee22.png)

This is the prediction of $x_t$ given $x_{t-1}$.

To see if we can predict long term evolution, we can predict $x_{t+N}$ given $x_{t}$.

<!-- Under construction -->

RC-ESN is [SOTA](https://arxiv.org/pdf/1906.08829.pdf) in short-time weather prediction compared to ANN, RNN-LSTM, that's why it's actively researched.

Yet, recent weather-forcasting method is based on 3D-CNN, RNN-LSTM, and other models that are not based on RC-ESN.


## AI for Physics

> Jie Ren, TJU, 2023-07-31 09:11:25

> Why physics in AI?
>
> Symmetry, geometry, and conservation laws are the key to understand the world.

> Physics-inspired: Inspired from bird is OK, but understand hydrodynamics is the key.

> 神经网络已经很好，但我们应该要超过模仿人脑的水平。

> Hopfield is still alive and well.

> AI done on physical system is the future of computing

> AI algorithm is possiblly inspired by physics: Statistical machine learning inspired by wave and diffusion system.
### Wave System

> Protein folding is a NP-hard problem.
>
> Yet nature is the most performant commputer.
>
> Well, that's why analog computing is proposed and delayed due to it's inaccuracy.
>
> Is today the best time to bring back analog computing?

Matrix Optics:

Wave can be transformed by an optical set, defined by Fresnel-Kirchhoff diffraction integral.

$$
U(x,y) = \frac{1}{i\lambda} \int_{-\infty}^\infty \int_{-\infty}^\infty U(x',y') \frac{e^{ikr}}{r} dx' dy'
$$

where $U(x,y)$ is the wave function, $\lambda$ is the wavelength, $k$ is the wave number, $r$ is the distance between the point $(x,y)$ and the point $(x',y')$.

Not only the amplititue , but also the phase of the wave can be transformed by an optical set.

> Will neural network be better when expanded on complex plain?

Optics is suitable for spiking neural netowork using phase transition and circular-harmoic oscillators.

#### How recurrent NN can be mapped to wave function

Given a wave distribuiton $u(x,y)$, we have the equation:

$$
\frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u = f
$$

where $u$ is the wave distribution, $t$ is the time, $c$ is the wave speed, and $\nabla^2$ is the Laplace operator, and $f$ is the source term.

Do a simple discretization:

$$
\frac{u_{t+1} - 2u_t + u_{t-1}}{\Delta t^2} - c^2 \nabla^2 u_t = f_t
$$

We can rewrite it a s a matrix form:

$$
\begin{bmatrix}
u_{t+1}\\
u_t
\end{bmatrix} = \begin{bmatrix}
2I - c^2 \Delta t^2 \nabla^2 & -I\\
I & 0
\end{bmatrix} \begin{bmatrix}
u_t\\
u_{t-1}
\end{bmatrix}
+
\Delta t^2\begin{bmatrix}
f_t\\
0
\end{bmatrix}
$$

where $u_{t+1}$ is the wave distribution at time $t+1$, $u_t$ is the wave distribution at time $t$, $u_{t-1}$ is the wave distribution at time $t-1$, $I$ is the identity matrix, $\Delta t$ is the time step, and $\nabla^2$ is the Laplace operator.

We can define the hidden state $h_t = \begin{bmatrix} u_t\\ u_{t-1}\end{bmatrix}$, the weight matrix $W = \begin{bmatrix} 2I - c^2 \Delta t^2 \nabla^2 & -I\\ I & 0\end{bmatrix}$, and the input $x_t = \begin{bmatrix} f_t\\ 0\end{bmatrix}$, then we have

$$
h_{t+1} = W h_t + \Delta t^2 x_t\\
y_{t+1} = [P^{(0)} h_{t+1}]^2
$$

where $h_{t+1}$ is the hidden state at time $t+1$, $W$ is the weight matrix, $h_t$ is the hidden state at time $t$, $x_t$ is the input at time $t$, $y_{t+1}$ is the output at time $t+1$, and $P^{(0)}$ is the projection operator.

> How is the training done?

No backpropagation needed. Pseudo-inverse is used to train the network.

Pseudo-inverse is defined as:

$$
A^+ = (A^T A)^{-1} A^T
$$

where $A^+$ is the pseudo-inverse of $A$, $A^T$ is the transpose of $A$, and $A^{-1}$ is the inverse of $A$.

It's called psuedo-inverse because it is not defined when $A^T A$ is not invertible.

To train an optical neural network, we need to solve the following equation:

$$
Y=XW
$$

The solution using pseudo-inverse is:

$$
W = X^+ Y = (X^T X)^{-1} X^T Y
$$

> This is only linear , due to the difficulty in replicating non-linear activations.

#### How quantum grover algorithm can be mapped to ML

Grover algorithm is a quantum algorithm that finds a specific element in an unsorted list with high probability.

To find the element $x$ in the list $L$, we need to find the index $i$ such that $L[i] = x$.

Steps of Grover algorithm:

- Initialize the state $\ket{\psi_0} = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} \ket{i}$, where $N$ is the number of elements in the list.
-  Apply the Hadamard gate to the state $\ket{\psi_0}$ to get the state $\ket{\psi_1} = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} \ket{i}$.
-  Apply the oracle operator $O$ to the state $\ket{\psi_1}$ to get the state $\ket{\psi_2} = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} (-1)^{f(i)} \ket{i}$, where $f(i) = 1$ if $L[i] = x$, and $f(i) = 0$ if $L[i] \ne x$.
-  Apply the diffusion operator $D$ to the state $\ket{\psi_2}$ to get the state $\ket{\psi_3} = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} (-1)^{f(i)} \ket{i}$.
-  Repeat steps 3 and 4 for $k$ times.
-  Measure the state $\ket{\psi_3}$ to get the index $i$.

> Why this works, an intuitive explanation?
>
> The oracle operator $O$ flips the sign of the state $\ket{i}$ if $L[i] = x$, and does nothing if $L[i] \ne x$. The diffusion operator $D$ flips the sign of the state $\ket{i}$ if $L[i] \ne x$, and does nothing if $L[i] = x$. 
> 
> So the oracle operator $O$ and the diffusion operator $D$ together can amplify the amplitude of the state $\ket{i}$ if $L[i] = x$, and reduce the amplitude of the state $\ket{i}$ if $L[i] \ne x$. So after $k$ iterations, the state $\ket{i}$ with $L[i] = x$ will have a much higher amplitude than the state $\ket{i}$ with $L[i] \ne x$, so we can measure the state $\ket{i}$ to get the index $i$.

> Real intelligence is creation.
>
> 看视频的智能<<创造视频的智能

Implementation of Grover algorithm using optical neural network:

> Using surface wave materials to implement the oracle operator $O$ and the diffusion operator $D$?

### Diffusion System

> Dimension reduction is the key to understand the world. It's the essence of understanding.

Two ingredient of Learning:

- Manifold learning
- Diffusion mapping

#### Manifold learning

```
<!-- To be Continued -->
```

#### Diffusion mapping

Given data $\{x_n\}^N_{n=1}, x_n\in \mathbb{R}^p$, we define the distance between $x_i$ and $x_j$ as:

$$
A_{i,j} = \exp \Big(-\frac{\|x_i - x_j\|^2}{\epsilon}\Big)
$$

where $A_{i,j}$ is the distance between $x_i$ and $x_j$, $x_i$ is the $i$-th data point, $x_j$ is the $j$-th data point, $\epsilon$ is the parameter.

> Gaussian kernel is used here, because it is a smooth function.

We define the diagonal matrix $D$ as:

$$
D_{i,i} = \sum_{j=1}^N A_{i,j}
$$

where $D_{i,i}$ is the $i$-th diagonal element of $D$, and $A_{i,j}$ is the distance between $x_i$ and $x_j$.

We define the probability matrix $P$ as:

$$
P = D^{-1} A
$$

where $P$ is the probability matrix, $D$ is the diagonal matrix, and $A$ is the matrix.

Then the Laplacian matrix $L$ is defined as:

<!-- what  ds the difference between row and column normalization?-->

$$
L = I - P  
$$

where $L$ is the Laplacian matrix, $I$ is the identity matrix, and $P$ is the probability matrix.

The equation of motion can be written as:

$$
\frac{\partial }{\partial t} \bold{p}(t) = -L \bold{p}(t)
$$

which is the diffusion equation. Where $\bold{p}(t)$ is the probability distribution at time $t$, and $L$ is the Laplacian matrix.

> what  ds the difference between row and column normalization?
>
> They represent different physics systems. Row normalization is diffusion, and column normalization is heat conduction. $\red{\text{What makes them different?}}$

> Google PageRank is essentially this idea, but it developed a faster algorithm. (well, is left and wright different?)

### 1.1 网络拓扑结构

设W为所有网页的集合，其中的每个网页都可以从根网页开始经过有限步的跳转被访问。

$n=Card(W)$是网页的数量，对Google来说，这个数值随着时间变化，到2004年6月，n已经超过4千万。

定义G为$n\times n$的连接矩阵，满足$g_{ij}=1$如果网页i有指向网页j的链接。否则设为0.

它描述了互联网的拓扑结构，是一个非常巨大，但是很稀疏`sparse` 的矩阵。

实际上，矩阵中所有非零元的数量就是万维网中的超链接数量。

### 1.2 马尔可夫转移矩阵

用户在一个网页可能跟随下一个链接（概率p），也可能随机跳转到任何一个网页，因此可以得到转移概率矩阵

首先定义$r_i=\sum_j h_{ij}$，它是从i网页可以连出去的j网页的数量

那么可以得到转移概率矩阵W

$w_{ij}=ph_{ij}/r_{i}+(1-p)/n$,如果$r_{i}\ne0$，即i网页中存在超链接指向其他网页。

$w_{ij}=1/n$,如果$r_{i}=0$

注意到矩阵W是对连接矩阵行的求和的`scaling` ，此时得到的就是从一个网页跳转到另外一个网页的概率。实际上通常让$p=0.85$，考虑$n=4\times 10^9$，那么这里的每个值都是很小的。

由此我们得到了马尔可夫转移概率矩阵，其元素严格介于0和1，并且每行之和严格等于1（即必然会跳出这个网页）。

### 1.3 网站排序

计算网页的顺序，简单来说就是计算稳定的分布，并按照概率进行评分，即计算方程

$$Ax=x$$

的解，其中$x$ 满足$\sum x_i=1$

实际运算时，一种可行的方案是从一个状态开始，不断的计算$x=Ax$，直到迭代的两个矢量的差的模长小于一个特定的值。

## 2 稀疏矩阵的幂运算

### 2.1 运算的化简

Goggle的实际运行方式其实根本不涉及幂运算，因为$A^k$的计算可以以通过更新超链接的权重进行

在Matlab中计算PageRank利用了马尔可夫矩阵的特定结构，以下的方法可以保持矩阵的稀疏性，将矩阵写为

$$A=pGD+ez^T$$

其中矩阵$D$定义为：如果$r_i\ne0~,d_{ii}=1/r_i$，否则$d_{ii}=0$

而$e$为n维矢量，分量均为1.

$z$为n维矢量，分量满足$z_i=(1-p)/n$，如果$r_i\ne 0$,否则$z_i=1/n$

从而方程可写为

$$(I-pGD)x=\gamma e$$

其中$γ = z^T x.$ 

虽然我们不知道$\gamma$的值，但是只要我们求解了$(1-pGD)x=e$的$x$，就可以通过$\sum x_i=1$直接确定满足的解。

因此问题转化为求解

$$(I-pGD)x=e$$

> Google implement a $\alpha\approx 0.85$ that stops the chain from trapping. But the problem we are dealling here require $n$ eigen value and vector, and is working on a much smaller system.

> Any way, what left is to compute the final distribuiton of the chain.

Find solution for

$$
\bold{P}x= x
$$

> Ergodicity? It dependes on the connectivity of the graph

> This method claims to reached a new representation of the data, and can define a new distance $\rho$ between the data points.

```
<!-- Under construction -->
```

#### Langevin diffusion

Focker-Planck equation:

$$
\frac{\partial p}{\partial t} = -\nabla \cdot (p \bold{v}) + \nabla \cdot (\bold{D} \nabla p)
$$

where $p$ is the probability distribution, $t$ is the time, $\bold{v}$ is the velocity, and $\bold{D}$ is the diffusion tensor.

Langevin equation:

$$
\dot{\bold{x}} = - \nabla U(\bold{x}) + \bold{f} + \sqrt{2D} \bold{\eta}
$$

where $\bold{x}$ is the position, $U(\bold{x})$ is the potential, $\bold{f}$ is the force, $D$ is the diffusion coefficient, and $\bold{\eta}$ is the Gaussian white noise.

> what if this is not gaussian noise.
The equivalence between the Focker-Planck equation and the Langevin equation:

Given 

$$
U(\bold{x}) = - \log p(\bold{x}) \\
\bold{\dot{x}} = - \nabla U(\bold{x}) + \bold{f} + \sqrt{2D} \bold{\eta}\\
\braket{\eta} = 0\\
\braket{\eta_i(t) \eta_j(t')} = \delta_{ij} \delta(t-t')
$$

we have

$$
\partial_tp(x,t) = - \partial_x \cdot (p \dot{\bold{x}}) = - \partial_x \cdot (p \nabla U(\bold{x}) - p \bold{f} - p \sqrt{2D} \bold{\eta}) = \hat{\Omega} p
$$

Then 

$$
p(x, t+ \delta t) = T \exp{(\int_{t}^{t + \Delta t}dt \hat{\Omega})} p(x,t)\\
= [1+ \int_{t}^{t+\Delta t} dt \hat{\Omega} + \frac{1}{2} \int_{t}^{t+\Delta t} dt \int_{t}^{t+\Delta t} dt' \hat{\Omega}(t) \hat{\Omega}(t') ] p(x,t) + O(\Delta t^3)\\
= p(x,t) + \delta t \nabla \cdot [(\delta U)p] + 2\int_{t}^{t+\Delta t} dt \int_{t}^{t+\Delta t} dt' \nabla \cdot [(\delta U) p \sqrt{D} \bold{\eta}] + O(\Delta t^3)\\=
p(x,t) + \Delta t \nabla \cdot[(-\nabla U)p] + (\partial_x^2p)\Delta t + O(\Delta t^3)
$$

As $\Delta t \to 0$, we haves

$$
\partial_t p = - \nabla \cdot [(-\nabla U)p] + \nabla \cdot (\partial_x^2p)
$$

reducing to the Focker-Planck equation.


### Topological Phonon Hall Effect

Topological phonon Hall effect is the phenomenon that the phonon Hall conductance is quantized in a topological insulator, so the heat can only be conducted in one direction.

Theory:

$$
H = \frac{1}{2}p^Tp + \frac{1}{2} u^T(K-A^2)u + u^TAp
$$

where $H$ is the Hamiltonian, $p$ is the momentum, $u$ is the displacement, $K$ is the stiffness matrix, and $A$ is the matrix of spin-latice coupling.

> What is the physical meaning of the stiffness matrix?
>
> It is the matrix of the spring constants.

We have

$$
J = \frac{\hbar}{4V}\sum_{\sigma, \sigma' ,\bold{k}} \frac{\omega_\sigma, \bold{k}}{\omega_\sigma, \bold{k}} \bold{a}_{\sigma, \bold{k}} (\frac{\partial \Omega^2}{\partial \bold{k}} + [A,\Omega^2])_{\sigma, \sigma'} a_{\sigma', \bold{k}}
$$

where $J$ is the phonon Hall conductance, $\hbar$ is the reduced Planck constant, $V$ is the volume, $\sigma$ is the spin, $\bold{k}$ is the wave vector, $\omega_\sigma, \bold{k}$ is the phonon frequency, $\bold{a}_{\sigma, \bold{k}}$ is the phonon annihilation operator, $\Omega$ is the Berry connection, and $A$ is the matrix of spin-latice coupling.

The normal velocity $\frac{\partial \Omega^2}{\partial \bold{k}}$ is responsible for the longitudinal phonon transport. The Berry curvature $[A,\Omega^2]$ is responsible for the transverse phonon transport.

> How theory classify different phases?
>
> By the topological invariant, e.g. Chern number.

Problems arise when we are dealing with amorphous topological phonon strucutures, because the topological invariant is not well defined.

> Why classification of different phase important?

### Recommendation system

Previous diffusion model, we define

$$
P=AD^{-1}
$$

but for heat conduction, we define

$$
P=D^{-1}A
$$

> What is the difference between the two?


Recommedation system is a system that recommends items to users based on their preferences, if we denote the preference of user $i$ for item $j$ as $r_{ij}$, then we have

$$
r_{ij} = \sum_{k=1}^K u_{ik} v_{kj}
$$

where $r_{ij}$ is the preference of user $i$ for item $j$, $u_{ik}$ is the preference of user $i$ for feature $k$, $v_{kj}$ is the preference of item $j$ for feature $k$, and $K$ is the number of features.

Consider a book selling situation, here item is book, user is reader, and feature is genre.

The central challenge in recommendation system:

After some tedious derivaion, our gioal is still find the eigen values and eigen vectors. It apply finite iterations to do this. (with truncation)

### Machine learning stochastic dynamics

> yin Tang, BNU, 2023-07-31 13:04:59

> Dynamic system can be roughly divided into deterministic and stohastic

Stochastic dynamics (analytical on blackbooard)

- Markvian, discrete/continuous state

Machine learning discrete-state stochasitc dynamics (coding)

 - Chemical master eq (Gillespie algorithm, VAN)
 - Nonequilibrium statitical physics (Langevin equation, Fokker-Planck equation)

Machine learning continuous-state stochasitic dynmaics;
-  Flow model to Langevin dynamics


#### Marvovian stochastic dynamics

|State|Discrete|Continuous
|---|---|---|
|Trajectory view|Recursion equation(e.g. Ranodm walk)|Stochastic differential (e.g. Ornstein-Uhlenbeck process)
|Distribution view|Master equation(e.g. random walk, birth-death, chemical master equation)| Fokker-Planck equation(e.g. diffusion equation)|

Brownian motion:

$$
\dot{x} = \xi(t)
$$

where $\dot{x}$ is the velocity, and $\xi(t)$ is the Gaussian white noise.

It's statistical property is:

$$
\braket{\xi(t)} = 0\\
\braket{\xi(t) \xi(t')} = \delta(t-t')
$$

where $\braket{\xi(t)}$ is the average of $\xi(t)$, and $\braket{\xi(t) \xi(t')}$ is the average of $\xi(t) \xi(t')$.

Now  we have

$$
\braket{x(t)} = \braket{\int_0^t dt' \xi(t')} = 0\\
\braket{x(t) x(t')} = \braket{\int_0^t dt' \xi(t) \int_0^{t'} dt'' \xi(t'')} = \int_0^t dt' \int_0^{t'} dt'' \braket{\xi(t) \xi(t'')} = t\\
$$

Given a random walk on a line with movement 1, with equal probability on left or right


$$
\braket{x_t^2} = \braket{(x_t{t-1}+\delta x)^2} = 2Dt
$$

The final distribution of the data is 

$$
P(x,t)=\frac{1}{\sqrt{2Dt}}e^{-\frac{x^2}{2Dt}}
$$

where $P(x,t)$ is the probability distribution, $x$ is the position, $t$ is the time, and $D$ is the diffusion coefficient.

Distribuiton view: random walk

$$
P(n, t+\Delta ) = P(n-1,  t)\cdot \frac{1}{2} + P(n+1,  t)\cdot \frac{1}{2}
$$

The change in probability overthe time interval \$\Delta t$ is 

$$
\Delta P(n, t) = P(n, t+\Delta t) - P(n, t) = P(n-1,  t)\cdot \frac{1}{2} + P(n+1,  t)\cdot \frac{1}{2} - P(n, t)
$$

Take continuous-time space limit and use diffusoion constant:

$$
\frac{\partial P}{\partial t} =
 D \frac{\partial^2 P}{\partial x^2} 
$$

where $P$ is the probability distribution, $t$ is the time, $D$ is the diffusion coefficient, and $x$ is the position.

> can we dervie the distribution from this equation?

Solve the equation:

- Fourier transform: $P(x,t) = \int_{-\infty}^\infty \frac{dk}{2\pi} \tilde{P}(k,t) e^{ikx}$
- Then we have: $\frac{\partial \tilde{P}}{\partial t} = -Dk^2 \tilde{P}$
- The solution is: $\tilde{P}(k,t) = \tilde{P}(k,0) e^{-Dk^2 t}$
- Inverse Fourier transform: $P(x,t) = \int_{-\infty}^\infty \frac{dk}{2\pi} \tilde{P}(k,0) e^{-Dk^2 t} e^{ikx}$
- The solution is: $P(x,t) = \frac{1}{\sqrt{4\pi Dt}} e^{-\frac{x^2}{4Dt}}$

> Here we have $4D$, but what we commonly see is $2D$, this is because we have two directions, and the diffusion coefficient is the average of the two directions.
>
> $$ D = \frac{1}{2} (D_x + D_y)$$
>
> So 
>
> $$ P(x,t) = \frac{1}{\sqrt{2\pi Dt}} e^{-\frac{x^2}{2Dt}}$$
>
> is for 1- Ddiffusion, and $4D$ is for 2-D diffusion.
>
> For 3-D diffusion, it is $6D$.

> If you got 7 solutions to a problem, you finally understand it. -- Feynman

> A Random Walk Down Wall Street: The Time-Tested Strategy for Successful Investing. -- Burton G. Malkiel

An experiemnt gives $D=51.1\mu\text{ m}^2/s$ of 大肠杆菌(E. coli), we can derive

$$
\braket{x^2} = 2Dt
$$

To walk for around $x=1\text{ cm}$, we need $t=10^6\text{ s}$, which is around 11 days.

This is too slow.

If we put food to the side, it will have a drift velocity $v$, and the equation becomes

$$
\frac{\partial P}{\partial t} = D \frac{\partial^2 P}{\partial x^2} - v \frac{\partial P}{\partial x}
$$

where $P$ is the probability distribution, $t$ is the time, $D$ is the diffusion coefficient, $x$ is the position, and $v$ is the drift velocity.

> Random walk is everywhere, e.g. complex network, search problem, complex system, and anomalous diffusion.

> Every stochastic process can be mapped to a random walk.
>
> seriously ?

### Chemical master equation

Chemical master equation is a stochastic model for chemical reactions. It is stochastic with fluctuations, many chemical species beyond a single "walker".

One specis., one reactin: $x\to^k \emptyset$, which means that the species $x$ decays to nothing with rate $k$.

Continuous: Reaction Rate Equation

$$
\frac{dx}{dt} = -kx + \text{noise}
$$

which is the rate of change of the number of molecules.

Discrete: Chemical Master Equation

$$
\partial_t P_t(n) = -k(n+1)P_t(n+1) - knP_t(n)
$$

where $P_t(n)$ is the probability distribution, $t$ is the time, $n$ is the number of molecules, and $k$ is the reaction rate. This function is intuitively the probability of having $n$ molecules at time $t$.

The discre description is more accurate, but it is hard to solve for large $n$.

#### A solvable example: birth-death process

Define

$$
X\in [0,N], \emptyset \to^{k_2} X, X \to^{k_1} \emptyset
$$

We have the equation

$$
\frac{dP_t(n)}{dt} = B(n+1)P_t(n+1) + F(n-1)P_t(n-1) - (B(n) + F(n))P_t(n)\\
F(n) = k_2\\
B(n) = k_1 n
$$

where $P_t(n)$ is the probability distribution, $t$ is the time, $n$ is the number of molecules, $B(n)$ is the birth rate, $F(n)$ is the death rate, $k_1$ is the birth rate constant, and $k_2$ is the death rate constant.

> How to solve it?

Steady state: $\frac{dP_t(n)}{dt} = 0$, or

$$
B(n+1)P_t(n+1) + F(n-1)P_t(n-1) = (B(n) + F(n))P_t(n)
$$

We can use a generating function to solve it:

- Define $G(z) = \sum_{n=0}^N z^n P_t(n)$
- Then we have $G'(z) = \sum_{n=0}^N n z^{n-1} P_t(n)$
- And $G''(z) = \sum_{n=0}^N n(n-1) z^{n-2} P_t(n)$
- So we have $G''(z) = B(z) G(z) + F(z) G(z)$
- Then we have $G''(z) = (B(z) + F(z)) G(z)$
- Then we have $G''(z) = (k_1 z + k_2) G(z)$
- Then we have $G''(z) - k_1 z G(z) - k_2 G(z) = 0$
- Then we have $G(z) = A e^{r_1 z} + B e^{r_2 z}$

Solve the equation ,we have

$$
A=\frac{1}{r_2 - r_1} \frac{r_2}{r_2 - k_1} \\
B=\frac{1}{r_2 - r_1} \frac{-r_1}{r_2 - k_1}\\
r_1 = \frac{k_1 - \sqrt{k_1^2 + 4k_2}}{2}\\
r_2 = \frac{k_1 + \sqrt{k_1^2 + 4k_2}}{2}
$$


> This is only for stationary state, how about the time evolution?
>
> Omitted.

Catalan number is the number of ways to put $n$ pairs of parentheses:

$$
C_n = \frac{1}{n+1} \binom{2n}{n}
$$

We can derive it's recurrence relation using generating function:

- Define $C(z) = \sum_{n=0}^\infty C_n z^n$
- Then we have $C(z) = 1 + z C(z)^2$
-  Then we have $C(z)^2 - z C(z) + 1 = 0$
-  Then we have $C(z) = \frac{1 \pm \sqrt{1 - 4z}}{2z}$
-  Then we have $C(z) = \frac{1}{2z} (1 + \sqrt{1 - 4z})$
-  Then we have $C(z) = \frac{1}{2z} \sum_{n=0}^\infty \binom{1/2}{n} (1 - 4z)^n$
-  Then we have $C_n = \frac{1}{2} \binom{2n}{n}$


化学主方程是一种描述随机化学反应系统中分子数目的概率分布随时间变化的数学模型。它可以用来研究那些分子数目很少或者受到随机波动影响的反应系统，例如生物细胞内的反应。

连续的情况下，反应速率方程是一种常微分方程，它假设分子数目是一个连续变量，可以用浓度或者摩尔数来表示。反应速率方程描述了分子数目随时间的变化率，它与反应速率常数和分子数目本身有关。

$$
\frac{dx}{dt} = -kx
$$

表示一个一阶反应$A \xrightarrow{k} B$，其中$x$是$A$分子的数目，$k$是反应速率常数。这个方程的意义是，$A$分子的消耗速率与$A$分子的数目成正比，比例系数为$k$。这个方程的解是

$$
x(t) = x(0) e^{-kt}
$$

其中$x(0)$是初始时刻$A$分子的数目。这个解表明，随着时间的增加，$A$分子的数目呈指数衰减。

离散的情况下，化学主方程是一种偏微分方程，它假设分子数目是一个离散变量，只能取整数值。化学主方程描述了分子数目的概率分布随时间的变化率，它与转移概率有关。转移概率是指在一个很小的时间间隔内，系统从一个状态跳到另一个状态的概率。

$$
\partial_t P_t(n) = -k(n+1)P_t(n+1) - knP_t(n)
$$

也表示一个一阶反应$A \xrightarrow{k} B$，其中$n$是$A$分子的数目，$P_t(n)$是在时刻$t$系统处于状态$n$的概率。这个方程的意义是，状态$n$的概率随时间的变化率等于从状态$n+1$跳到状态$n$和从状态$n$跳到状态$n-1$的概率之差。从状态$n+1$跳到状态$n$的概率为$k(n+1)P_t(n+1)$，因为每个$A$分子都有$k\Delta t$的概率在$\Delta t$内发生反应，并且有$n+1$个这样的分子。从状态$n$跳到状态$n-1$的概率为$k n P_t(n)$，原因同上。这个方程可以用递推公式或者生成函数来求解.

> 高情商：时代变了，大家解析能力不一样了

Solve function

$$
\partial_t \eta(y,t) = -k_1\partial_y \eta
$$

We use characteristic line method, which is based on the idea that the solution of the PDE is constant along the characteristic lines, which are the integral curves of the vector field $(-1, -k_1)$.

An intuitive explanation: the solution of the PDE is constant along the characteristic lines, because the PDE is the rate of change of the solution, and the characteristic lines are the lines that the solution is constant.


```mathematica
# Define the PDE of the form a(x,y)u_x + b(x,y)u_y = c(x,y)u + d(x,y)
# Define the boundary conditions for u
# Define the domain and the grid size
# Initialize the grid and the solution matrix
# Loop over the grid points
  # Check if the point is on the boundary
  # If yes, use the boundary condition
  # If no, use the characteristic line method
    # Find the slope of the characteristic curve using dy/dx = b(x,y)/a(x,y)
    # Find the ODE along the characteristic curve using du/dx = (d(x,y) - c(x,y)u)/a(x,y)
    # Find the previous point on the characteristic curve using backward difference
    # Interpolate the value of u at the previous point using linear interpolation
    # Solve the ODE along the characteristic curve using a numerical method
  # Store the value of u at the current point in the solution matrix
# End loop
# Plot or output the solution
```

 ```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the functions a, b, c, d
def a(x,y):
    return 1

def b(x,y):
    return 1

def c(x,y):
    return 0

def d(x,y):
    return 0

# Define the boundary conditions for u
def u_left(y):
    return np.sin(y)

def u_right(y):
    return np.cos(y)

# Define the domain and the grid size
x_min = 0
x_max = 1
y_min = 0
y_max = np.pi
n_x = 11 # number of grid points in x direction
n_y = 11 # number of grid points in y direction
h_x = (x_max - x_min) / (n_x - 1) # grid spacing in x direction
h_y = (y_max - y_min) / (n_y - 1) # grid spacing in y direction

# Initialize the grid and the solution matrix
x = np.linspace(x_min, x_max, n_x)
y = np.linspace(y_min, y_max, n_y)
u = np.zeros((n_x, n_y))

# Define a function that computes the slope of the characteristic curve
def slope(x,y):
    return b(x,y) / a(x,y)

# Define a function that computes the ODE along the characteristic curve
def ode(x,y,u):
    return (d(x,y) - c(x,y) * u) / a(x,y)

# Define a numerical method to solve the ODE (Euler's method)
def euler(x,y,u,h):
    return u + h * ode(x,y,u)

# Loop over the grid points
for i in range(n_x):
    for j in range(n_y):
        # Check if the point is on the left or right boundary
        if i == 0:
            # Use the left boundary condition
            u[i,j] = u_left(y[j])
        elif i == n_x - 1:
            # Use the right boundary condition
            u[i,j] = u_right(y[j])
        else:
            # Use the characteristic line method
            # Find the previous point on the characteristic curve
            x_prev = x[i-1]
            y_prev = y[j] - h_x * slope(x_prev, y[j])
            # Interpolate the value of u at the previous point
            u_prev = np.interp(y_prev, y, u[i-1,:])
            # Solve the ODE along the characteristic curve
            u[i,j] = euler(x_prev, y_prev, u_prev, h_x)

# Plot or output the solution
plt.contourf(x, y, u.T)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of PDE by characteristic line method')
plt.colorbar()
plt.show()
``` 

![Image](https://pic4.zhimg.com/80/v2-fbc3b28dd391321d01178bcc692e1f52.png)
This image is a solution for the PDE 
$$\partial_t \eta(y,t) + \partial_y \eta = 0$$ 
with the boundary conditions $\eta(0,t) = \sin(t)$ and $\eta(\pi,t) = \cos(t)$.

#### Gillespie algorithm

Gillespie algorithm is a stochastic simulation algorithm for chemical master equation. It is a Monte Carlo method that simulates the time evolution of a chemical system.

A pseudo-code of the algorithm:

```
# Define the system with the initial number of molecules and the reaction rate constants
# Initialize the time to zero
# Loop until the end time or condition is reached
  # Calculate the total propensity function by summing all the reaction propensities
  # Generate two random numbers from a uniform distribution between 0 and 1
  # Use one random number to determine the time interval until the next reaction event
  # Use another random number to determine which reaction will occur next
  # Update the system by increasing the time by the time interval and changing the number of molecules according to the chosen reaction
# End loop
# Plot or output the results
```

This algorithm is based on the following theory:

- The time interval until the next reaction event is an exponential random variable with the rate parameter equal to the total propensity function: $t \sim \text{Exp}(\alpha_0)$
- The probability of each reaction event is proportional to its propensity function: $\text{Pr}(R_i) = \frac{\alpha_i}{\alpha_0}$
- The number of molecules after each reaction event is a multinomial random variable with the number of trials equal to the number of molecules before the reaction event and the probability of each outcome equal to the probability of each reaction event: $X \sim \text{Multinomial}(n, \text{Pr}(R_i))$
- The number of molecules after each reaction event is a Poisson random variable with the rate parameter equal to the propensity function: $X \sim \text{Poisson}(\alpha_i)$

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the system with two reactions: A -> B and B -> A
# The initial number of molecules for A and B are 10 and 0, respectively
# The reaction rate constants for A -> B and B -> A are 1.0 and 0.5, respectively
x_A = 10 # number of A molecules
x_B = 0 # number of B molecules
k_1 = 1.0 # rate constant for A -> B
k_2 = 0.5 # rate constant for B -> A

# Initialize the time to zero
t = 0

# Create empty lists to store the time and molecule values
t_list = []
x_A_list = []
x_B_list = []

# Loop until the end time of 10 is reached
while t < 10:
    # Calculate the total propensity function
    a_total = k_1 * x_A + k_2 * x_B
    
    # Generate two random numbers from a uniform distribution between 0 and 1
    r_1 = np.random.uniform(0,1)
    r_2 = np.random.uniform(0,1)
    
    # Use one random number to determine the time interval until the next reaction event
    tau = (1 / a_total) * np.log(1 / r_1)
    
    # Use another random number to determine which reaction will occur next
    if r_2 < (k_1 * x_A) / a_total:
        # A -> B occurs
        x_A -= 1 # decrease A by 1
        x_B += 1 # increase B by 1
    else:
        # B -> A occurs
        x_A += 1 # increase A by 1
        x_B -= 1 # decrease B by 1
    
    # Update the time by adding the time interval
    t += tau
    
    # Append the time and molecule values to the lists
    t_list.append(t)
    x_A_list.append(x_A)
    x_B_list.append(x_B)

# Plot or output the results
plt.plot(t_list, x_A_list, label="A")
plt.plot(t_list, x_B_list, label="B")
plt.xlabel("Time")
plt.ylabel("Number of molecules")
plt.title("Gillespie algorithm for a simple chemical system")
plt.legend()
plt.show()
```

![Image](https://pic4.zhimg.com/80/v2-c20b480fdd203f31ddeb0be7a88ceb93.png)

Gaurateen of this algorihtm:

- Theorem: Two random variable can simulate a random walk process. (?)

Take-home message:

- Diffusion coefficient is $2dD$, where $d=1,2,3$ is the dimension, and $D$ is the diffusion coefficient.

### Stochastic reaction networks with $M$ species (each with count $N$)

$$
\sum_{j=1}^M r_{ki}X_i \xrightarrow{c_k} \sum_{j=1}^M p_{ki}X_i
$$

We have equation

$$
\partial_t P_t(\bold{n}) = \sum_{k=1}^K [a_k(\bold{n} - s_k)P_t(\bold{n} - s_k) - a_k(\bold{n})P_t(\bold{n})]
$$

Then we can derive the final state of the system with the matrix $r,p,c$. (The matrix $r$ is the stoichiometry matrix, $p$ is the propensity matrix, and $c$ is the reaction rate matrix.)

$$
c = p - r
$$

The chemical transformation matrx $\mathbb{T}$ is defined as

A numeric example:

$$
\emptyset \xrightarrow{1} A\\
A \xrightarrow{1} \emptyset
$$

We have

$$
r = \begin{bmatrix}
-1\\
1
\end{bmatrix}\\
p = \begin{bmatrix}
1\\
1
\end{bmatrix}\\
c = \begin{bmatrix}
2\\
0
\end{bmatrix}
$$

The joint probability $P_t(\bold{n})$ can be difficult to handle, since $\bold{n}$ is a vector of length $N^M$.

We can propose a parameterization of the joint probability $P_t^{\theta_{t}}(\bold{n})$, and then minimize the loss defined as

$$
\mathcal{L}=D_\text{KL}(P_{t^+\delta t}^{\theta_{t+\delta t}}(\bold{n})||\mathbb{T}P_t^{\theta_t}(\bold{n}))
$$

where $D_\text{KL}$ is the Kullback-Leibler divergence, $P_{t^+\delta t}^{\theta_{t+\delta t}}(\bold{n})$ is the joint probability at time $t+\delta t$ with parameter $\theta_{t+\delta t}$, $P_t^{\theta_t}(\bold{n})$ is the joint probability at time $t$ with parameter $\theta_t$, and $\mathbb{T}$ is the chemical transformation matrix.

The dimension of the matrix $\mathbb{T}$ is $N^M\times N^M$, where $N$ is the number of molecules, and $M$ is the number of species.

The loss function is the Kullback-Leibler divergence between the joint probability at time $t+\delta t$ and the joint probability at time $t$ transformed by the chemical transformation matrix. 

> Why not implement the time $t$ as an embedding?
> 
> Tried , not promising due to the long step negative probability problem.

> Is this size big enough? Given $N$ and $M$, the distribuiton is of dimension $N^M$, where $\bold{n}$ indicates the number state of he system
> 
> e.g. $\ket{1,0,0}$ means there is one molecule of species 1, and no molecule of species 2 and 3.

## Learning nonequilibrium statistical mechanics and dynamical phase transitions

> Ying Tang, International Academic Center of Complex Systems, Beijing Normal University

Non-equilibrium statistical mechanics is the study of the behavior of systems that are not in thermodynamic equilibrium. Most systems found in nature are not in thermodynamic equilibrium because they are not in stationary states, and are continuously and discontinuously subject to flux of matter and energy to and from other systems and to and from their environment.

$$
\frac{d}{dt}\ket{P_t} = \mathcal{W} \ket{P_t}
$$

where $\ket{P_t}$ is the probability distribution at  $t$, and $\mathcal{W}$ is the generator of the Markov process.

$$
\ket{P} = \sum_x P(x) \ket{x}\\
\ket{x} = (x_1, x_2, \dots, x_n)
$$

where $\ket{P}$ is the probability distribution, $P(x)$ is the probability of the state $x$, and $\ket{x}$ is the state.

Kinetically constrained models (KCMs) are lattice models of interacting particles that are subject to constraints on their dynamics. They are used to model the dynamics of glassy systems.

> Why KCMs can be used to model the dynamics of glassy systems?
>
> It provide a const

FA model, South-East model

$$
\mathcal{W}^{\text{KCM}} = \sum_i f_i (c\sigma_i^+ + (1-c)\sigma_i^- - c(1 - n_i) - (1-c)n_i)
$$

where $f_i$ is the rate of the $i$-th site, $c$ is the constraint parameter, $\sigma_i^+$ is the creation operator on the $i$-th site, $\sigma_i^-$ is the annihilation operator on the $i$-th site, $n_i$ is the occupation number on the $i$-th site.

> How to determine the way to flip the spin?
>
> This can be done in the following way: flip the spin with probability $c$ if the site is occupied, and flip the spin with probability $1-c$ if the site is unoccupied.

Difficulty in applying this method to 3D systems:

- The number of states is too large.


ML can be used to estimate the dynamic partition function:

$$
\mathcal{Z}(t) = \sum_x e^{-\beta E(x)} P(x,t)
$$

where $\mathcal{Z}(t)$ is the dynamic partition function, $E(x)$ is the energy of the state $x$, and $P(x,t)$ is the probability of the state $x$ at  $t$.

> How to estimate the dynamic partition function?

We may use autorergressive model to estimate the dynamic partition function:

$$
\mathcal{Z}(t) = \sum_x e^{-\beta E(x)} P(x,t) \approx \sum_x e^{-\beta E(x)} \prod_{i=1}^n P(x_i|x_{<i},t)
$$

where $P(x_i|x_{<i},t)$ is the probability of the $i$-th spin given the previous spins at  $t$.

### Dynamic partition function and observable

Consider trajectory ensemble $\omega_t = \{\bold{x_0 \to x_{t_1} \to \dots \to x_{t_I}}\}$.

The dynamical partition funtion is the moment generating function with the counting field $s$:

$$
Z_t(s) = \sum_{\omega_t} e^{-\sum_{i=1}^I s_i E(x_{t_i})} P(\omega_t)
$$

Here counting field $s$ is ?

> This is the only work (the lecturer done) that use NN to observe things others don't.

### Track the distribuiton oin Ornstein-Uhlenbeck process


Stochastic differential equation:

$$
\dot{x} = -kx  + \sqrt{D} \xi(t)
$$

Fokker-Planck equation:

$$
\frac{\partial P}{\partial t} = \partial_x[kx\rho(x,t)] + \frac{1}{2}D\partial_x^2[\rho(x,t)]
$$

where $P$ is the probability distribution, $t$ is the time, $x$ is the position, $k$ is the drift coefficient, $D$ is the diffusion coefficient, and $\xi(t)$ is the Gaussian white noise.

> How to solve it?
>
> path integral approach

This is the Langevin equation of the Ornstein-Uhlenbeck process, and can be analytically solved:

$$
P(x_N,t_N|x_0,t_0) = \sqrt{\frac{k}{2\pi D(1-e^{-2k(t_N-t_0)})}} e^{-\frac{k(x_N-x_0e^{-k(t_N-t_0)})^2}{2D(1-e^{-2k(t_N-t_0)})}}
$$

where $P(x_N,t_N|x_0,t_0)$ is the probability distribution of the position $x$ at time $t_N$ given the position $x_0$ at time $t_0$, $k$ is the drift coefficient, $D$ is the diffusion coefficient, $t_N$ is the final time, and $t_0$ is the initial time.

> How is this connected to Stochastic gradient discent?
>
> SGD works as follows:
> $$x_{n+1} = x_n - \eta \nabla f(x_n) + \sqrt{2\eta D} \xi_n$$
>
> Under continuum limit, we have
>
> $$ \dot{x} = -\nabla f(x) + \sqrt{2D} \xi(t)$$

> Use flow model to solve Fokker-Planck equation: not introduced in detail. (work in progress)

## Machine Learning, Statistical Physic, and Complex System

> Xiaosong Chen, YNU, 2023-08-01 13:08:31

> Appy it to some toy models, to see if this blackbox model is expalainable.

### Long-range connected 2-D network percolation

This model exhibits a phase transition from a non-percolating phase to a percolating phase.

普适类(i.e. universality class): 在临界点附近，系统的物理性质具有普适性，即物理量的变化服从幂律分布。

普适类的定义：在一个普适类中，所有的模型都有相同的临界指数。

Critical phenomena of percolation:

- Order parameter $P(p)$: the probability that a site belongs to infinite cluster

$$
P(p)\propto \begin{cases}
(p-p_c)^\beta & p\to p_c^+\\
0 & p\le p_c
\end{cases}
$$

- Correlation length $\chi(p)$: the average distance between two sites in the same cluster, also mean cluster size

$$
\chi(p)\propto |p-p_c|^{-\gamma}
$$

标度性(i.e. scale invariance)：在临界点附近，系统的物理性质具有标度性，即物理量的变化服从幂律分布。

> ML supervised learning the phase transition of percolation, input is a 2-D network, output is the probability of percolation.

Different percolation models:

- Bond percolation: each bond is occupied with probability $p$.
- Site percolation: each site is occupied with probability $p$.

> Differnce: bond percolation is more difficult to simulate

Finding:

- Same network can learn different percolation under the same universality class.
- Different universality class must use the same network
- ML of critical behavior should have university class.

> what the hell is this all about, since it provide no theoretical insights or rigorous proof on these claims.

## 厄尔尼诺预测

A completely data driven approach


> Open the blackbox, what can we do?
>
> - visualize the weight matrix
> - statistical analysis of weights and output
> - connect the weights and the patterns/rules/law in the problem.

> The key is to represent, not solve.

> Play around with the dataset.

> Weather is different in different areas. have you considered the difference?
> 
> This is a global data, not limited to a region.

> Have you considered the long term prediction?
>
> Data is challenging, weather data is limited in past decades.

> Well, weather prediction is now working with CNN.

## Variational Bayesian Method

<!-- Some old stories that can be seen in papers. -->

Derivation of ELBO using KL divergence:

$$
\ln p(x_0) = \ln \int p(x_0, z_0) dz_0 = \ln \int q(z_0) \frac{p(x_0, z_0)}{q(z_0)} dz_0 \ge \int q(z_0) \ln \frac{p(x_0, z_0)}{q(z_0)} dz_0 = \mathcal{L}(q)
$$

where $p(x_0)$ is the marginal likelihood, $p(x_0, z_0)$ is the joint likelihood, $q(z_0)$ is the variational distribution, and $\mathcal{L}(q)$ is the ELBO.

### Fluctuation theorem

Fluctuation theorem is a theorem in statistical mechanics that describes the probability distribution of the time-averaged irreversible entropy production of a system that is arbitrarily far from equilibrium.

$$
\frac{P(\sigma)}{P(-\sigma)} = e^{\sigma}
$$

or

$$
\frac{p_0[W_0;\lambda]}{p_0[-W_0;\epsilon_Q\lambda]} = e^{\beta W_0}
$$

where $P(\sigma)$ is the probability distribution of the time-averaged irreversible entropy production $\sigma$, $p_0[W_0;\lambda]$ is the probability distribution of the work $W_0$ done on the system, $\lambda$ is the control parameter, $\epsilon_Q$ is the quantum of energy, and $\beta$ is the inverse temperature.


### Diffusion Model

- Foward  Process: $\green{\bold{x}_0} \to \bold{x}_t \to \bold{x}_{t+1}\to \red{\bold{x}_T}$
    
    The forward diffusion process is defined by a series of Gaussians  with first order Markov property. 
    
    $$
    q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})
    
    \\
    
    q(x_{1:T}|x_0)=\prod_{t=1}^Tq(x_t|x_{t-1})
    $$
    
    where $\beta_t = 0.02 t / T, T=1000$, increases linearly with time $t$
    set $\alpha_t = \sqrt{1-\beta_t}$
    
    - **A Nice Property: $x_t = ᾱ_tx_0 + \bar{β}_tϵ̄_t$**
        
        $$
        \begin{aligned}\mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\&= \dots \\&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})\end{aligned}
        $$
        
         
        🔥 **Merge Gaussians**
        
        - line 1 → line 2:
            
            $$
            \begin{aligned}
            & \sqrt{\alpha_t}{\color{blue}x_{t-1}} + \sqrt{1- \alpha_t} z_{t-1} \\
            
            = & \sqrt{\alpha_t}{\color{blue}(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1- \alpha_{t-1}} z_{t-2})} + \sqrt{1- \alpha_t} z_{t-1} \\
            
            = & \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + 
            {\color{red}\sqrt{{\alpha_t}(1- \alpha_{t-1})} z_{t-2} + \sqrt{1- \alpha_t} z_{t-1}} \\
            
            = & \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + {\color{red}\sqrt{1- \alpha_{t-1}\alpha_t} \bar{z}_{t-2}} \\
            
            \end{aligned}
            $$
            
            where the last two lines use the fact that the sum of two (independent) Gaussians is another Gaussian: 
            
            $$
            \mathcal{N}(\mu_1, \sigma_1^2\mathbf{I}) + \mathcal{N}(\mu_2, \sigma_2^2\mathbf{I}) = \mathcal{N}(\mu_1 + \mu_2, (\sigma^2_1 + \sigma_2^2)\mathbf{I})
            $$
            
            such that 
            
            $$
            \begin{aligned}
            \sigma_{\text{new}} =\sqrt{\sigma_1^2 + \sigma_2^2} &= \sqrt{ \color{red}\left(\sqrt{{\alpha_t}(1- \alpha_{t-1})}\right)^2+ \left(\sqrt{1- \alpha_t}\right)^2} \\
            & = \color{red}\sqrt{{\alpha_t}(1- \alpha_{t-1}) + (1- \alpha_t)} \\
            & ={\color{red}\sqrt{1- \alpha_{t}\alpha_{t-1}} }
            \end{aligned}
            $$
            
        </aside>
        
    
    > **Connection of Langevin dynamics**
    > 
    
     
    🔥 In fact, Lagevin dynamics is also a kind of **Markovian** Process with  **detailed balance** , so it also  satisfies Jarzyskin Equality  .[proved in his 1997 paper]
    **Snippt from that paper**
    
    ![Untitled](Untitled%2018.png)
    
    </aside>
    
    $$
    \bold{x}_t=
    \bold{x}_{t-1}
    +\frac{\delta}{2}\nabla_\bold{x} \log{q(x_{t-1})} 
    + \sqrt{\delta}\bold{\epsilon_t}, 
    
    \text{where }\bold{\epsilon_t}\thicksim \mathcal{N}(\bold{0},\bold{I}),\delta \text{ is the step size}
    $$
    
    > [**苏剑林. (Aug. 03, 2022)**](https://spaces.ac.cn/archives/9209)
    > 
    > 
    > ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2019.png)
    > 
    
     
    🚧 **Exercise**
    Can you find another process that satisfy Jarzyskin equality?
    
    </aside>
    
- Reverse Process: $\red{\bold{x}_T} \to \bold{x}_{t} \to \bold{x}_{t-1}\to \green{\bold{x}_0}$
    
    In the forward process, we model $q(\bold{x}_{t}|\bold{x}_{t+1})$ with markov chain.
    If we can sample from  $q(\bold{x}_{t-1}|\bold{x}_t)$, then we can reverse the process.
    
     
    🚨 **Pause and think, what would you do if you want to reverse the process?** 
    [Hint: with the help of Neural Networks]
    
    </aside>
    
     
    🤮 **DDPM: we define it**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2020.png)
    
    Very boring derivation, trying to learn $p_\theta$ by minimizing the **Negative Log Likelyhood**
    
    - KL divergence $D_{\text{KL}}(q(x) || p(x))$
        - KL divergence:  $D_{\text{KL}}(q(x) || p(x)) = \mathbb{E}_{q(x)} \log [q(x) / p(x)]$
            
            [https://www.assemblyai.com/blog/content/media/2022/05/KL_Divergence.mp4](https://www.assemblyai.com/blog/content/media/2022/05/KL_Divergence.mp4)
            
        - KL-divergence between two gaussians is tractable, having closed-form formula.
            - Let’s consider the case of single variable Gaussians:
            - Gaussian pdf: $\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp \left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$
            
            $$
            \begin{aligned}
            & D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) || \mathcal{N}(\mu_2, \sigma_2^2)) \\
            
            = & \int dx \left[\log \mathcal{N}(\mu_1, \sigma_1^2) - \log \mathcal{N}(\mu_2, \sigma_2^2)\right] \mathcal{N}(\mu_1, \sigma_1^2) \\
            
            = & \int dx  \left[ -\frac{1}{2} \log(2\pi) - \log \sigma_1 - \frac{1}{2} \left(\frac{x - \mu_1}{\sigma_1} \right)^2 \right. \\
            &\left. ~~~~~~~~~~~~+ \frac{1}{2} \log(2\pi) + \log \sigma_2 + \frac{1}{2} \left(\frac{x - \mu_2}{\sigma_2}\right)^2 \right] \\
            &~~~~~~~~~~~\times\frac{1}{\sqrt{2\pi\sigma_1}} \exp \left[ -\frac{1}{2}\left( \frac{x - \mu_1}{\sigma} \right)^2 \right] \\
            
            = & \mathbb{E}_{1}  \left[ \log \frac{\sigma_2}{\sigma_1} + \frac{1}{2} \left[ \left(\frac{x - \mu_2}{\sigma_2} \right)^2 - \left(\frac{x - \mu_1}{\sigma_1}\right)^2 \right] \right ] \\
            
            = & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} \mathbb{E}_1 [({x - \mu_2})^2] - \frac{1}{2\color{green}\sigma_1^2} \color{green}\mathbb{E}_1 [({x - \mu_1})^2] \\
            
            = & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} \mathbb{E}_1 [({x - \mu_2})^2] -  \frac{1}{2} \\
            
            = & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} \mathbb{E}_1 [({x - \mu_1 + \mu_1 - \mu_2})^2] -  \frac{1}{2} \\
            
            = & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} {\color{green}\mathbb{E}_1 [(x - \mu_1)^2} + 2(x-\mu_1)(\mu_1 - \mu_2) + (\mu_1 - \mu_2)^2] -  \frac{1}{2} \\
            
            = & \log\frac{\sigma_2}{\sigma_1} + \frac{{\color{green}\sigma_1^2} + (\mu_1 - \mu_2)^2}{2\sigma_2^2} -  \frac{1}{2} 
            \end{aligned}
            $$
            
            More generally for multivariate Gaussians with dimension $d$:
            
            $$
            \begin{aligned}
            & D_{\text{KL}}(\mathcal{N}(\mu_1, \Sigma_1) || \mathcal{N}(\mu_2, \Sigma_2)) \\
            
            = & \frac{1}{2} \left[\log\frac{|\Sigma_2|}{|\Sigma_1|} -d + \mathrm{tr}\{\Sigma_2^{-1}\Sigma_1\}  + (\mu_2 - \mu_1)\Sigma_2^{-1}(\mu_2 - \mu_1) \right]
            \end{aligned}
            $$
            
    - Negative Log Likelyhood
        
        The loss function is the difference between $p(x)$ and sample distribution $\pi ({x})$, we minimize
        
        $$
        \mathbb{KL}(\pi||p)=\sum_{x\in D}\pi(x)\ln{\Big[\frac{\pi(x)}{p(x)}\Big]} = \lang \ln{\frac{\pi}{p}} \rang_\pi 
        $$
        
        we may simplify it as
        
        $$
        \mathbb{KL}(\pi||p)=\lang \ln{\frac{\pi}{p}} \rang_\pi =\lang \ln{\pi} \rang_\pi - \lang \ln{p} \rang_\pi 
        $$
        
        in most cases, we treat $\pi(x)=\frac{1}{|D|}\sum_{x'\in D}\delta(x-x')$, and all we need is to minimize $- \lang \ln{p} \rang_\pi$, which can be simplified as
        
        $$
        \mathcal{L}=- \lang \ln{p} \rang_\pi=-\frac{1}{|D|}\sum_{x\in D}\ln{p(x)} 
        $$
        
        this is the **Negative Log Likelyhood** (what's this?), to minimize this is to maximize the likelyhood.
        
    - 🤮 Detailed Derivation in case you want to know
        
        Like VAE, we want to minimize the log likely hood, so 
        
        $$
        \begin{aligned}- \log p_\theta(\mathbf{x}_0) &\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) \\&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\\text{Let }L_\text{VLB} &= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)\end{aligned}
        $$
        
        We then try to minimize $L_{\text{VLB}}$, expand it into a sum of several parts
        
        $$
        \begin{aligned}L_\text{VLB} 
        
        &= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] 
        
        \\&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ \red{p_\theta(\mathbf{x}_T) }\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] 
        
        \\&= \mathbb{E}_q \Big[ \red{-\log p_\theta(\mathbf{x}_T)} + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] 
        
        \\&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{\red{t=2}}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] 
        
        \\&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{\red{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \red{\frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)}} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] 
        
        \\&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] 
        
        \\&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \red{\log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)}} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
        
        \\&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] 
        
        \\&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]\end{aligned}
        $$
        
        - Note
            - line 3 → line 4: just separate case of $t=1$ from the summation
            - line 4 → line 5: Markov property of the forward process, and the Bayes’ rule
                
                $$
                \begin{aligned}
                q(x_{t}|x_{t-1}) &= q(x_{t}|x_{t-1}, x_0) = \frac{q(x_t, x_{t-1}|x_0)}{q(x_{t-1}|x_0)} = {\color{red}q(x_{t-1} | x_{t}, x_0)} \cdot  \frac{q(x_t | x_0)}{q(x_{t-1}|x_0)}
                
                \end{aligned}
                $$
                
            - line 6 → line 7: sum of logarithms → logarithm of product. Then cancel those identical numerators and denominators
            - line 7 → line 8: rearrangement.
            - line 8 → line 9 (the last one):
                - recall that $D_{\text{KL}}(q(x) || p(x)) = \mathbb{E}_{q(x)} [\log q(x) / p(x)]$
                - for $L_t$:
                    
                    $$
                    \begin{aligned}
                    &\mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \\
                    
                    =~ &\mathbb{E}_{{\color{red}q(x_{t-1}|x_t, x_0)}q(x_t,x_0)q(x_{1:t-2,t+1:T}|x_{t-1},x_t,x_0)} \left[\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \\
                    
                    =~ & \mathbb{E}_{\bar{q}(x_{t-1})} \left[ \mathbb{E}_{q(x_{t-1}|x_t, x_0)} \left[\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \right]  \\
                    
                    =~ & \mathbb{E}_{\bar{q}(x_{t-1})} \left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0)|| p_\theta(x_{t-1}|x_t)) \right] \\
                    
                    =~ & D_{\text{KL}}(q(x_{t-1}|x_t, x_0)|| p_\theta(x_{t-1}|x_t))
                    \end{aligned}
                    $$
                    
                    where the expectation in line 3 is over a distribution $\bar{q}(x_{t-1})$ that is independent from the variable (namely $x_{t-1}$).
                    
                - Same for the rest.
        
        Let’s take a look at this
        
        $$
        \begin{aligned}L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)\end{aligned}
        $$
        
        Since all of them obey Gaussian Distribution, these can be analytically computed.
        
        $$
        \begin{aligned}L_t &= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \end{aligned}
        $$
        
         
        🔥 **Aha! We are in the end learning the noise! 
        BUT WAIT!**
        The above weighted MSE losses were found unstable in training. [DDPM (Ho et al 2020)](https://arxiv.org/pdf/2006.11239.pdf) instead use a simplified loss without the weighting term
        
        </aside>
        
        $$
        \begin{aligned}L_t^\text{simple}&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]\end{aligned}
        $$
        
        But we still need $\bold{\Sigma_\theta}$, they use $\beta_t$ instead.
        [Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed a new approach to learn $\bold{\Sigma_\theta}$ (And achived SOTA performance, of course), check it out if interested.
        
    </aside>
    
     
    🧐 **Lilian Weng: if $\beta_t$ is small, it will be.**
    Proved by: *Feller, William. "On the theory of stochastic processes, with particular reference to applications." Proceedings of the [First] Berkeley Symposium on Mathematical Statistics and Probability. University of California Press, 1949.* [No Papers Avaliable]
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2021.png)
    
    </aside>
    
     
    😎 **Jianlin Su: Bayes + Denoise, it’s natural!**
    
    [生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪](https://spaces.ac.cn/archives/9164)
    
    To derive $q(\bold{x}_{t-1}|\bold{x}_t)$, we apply Bayesian theorem
    
    $$
    q(\bold{x}_{t-1}|\bold{x}_t)
    
    =q(\bold{x}_{t}|\bold{x}_{t-1})
    
    \frac
    {q(\bold{x}_{t-1})}{q(\bold{x}_t)}
    $$
    
    In order to calculate, we consider the condition when $\bold{x}_0$ is given
    
    [](https://www.notion.so/2d782064ee2e4b5b8b8e1f3018f774bf?pvs=21)
    
    $$
    q(\bold{x}_{t-1}|\bold{x}_t,\red{\bold{x}_0})
    
    =q(\bold{x}_{t}|\bold{x}_{t-1},\red{\bold{x}_0})
    
    \frac
    {q(\bold{x}_{t-1}|\red{\bold{x}_0})}{q(\bold{x}_t|\red{\bold{x}_0})}
    $$
    
    All of them are tractable, we can derive the final result as
    
    $$
    q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}\left(\boldsymbol{x}_{t-1};\frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t}\boldsymbol{x}_t + \frac{\bar{\sqrt{\alpha}_{t-1}}\beta_t}{\bar{\beta}_t}\boldsymbol{x}_0,\frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t} \boldsymbol{I}\right)
    $$
    
    or more neatly
    
    $$
    q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
    $$
    
    - Detailed Derivation
        
        $$
        \begin{aligned}q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) &= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}\end{aligned}
        $$
        
        The second line:
        
        - the 1st term: using the definition the posterior of the reverse process Gaussian
            
            $$
            \begin{aligned}
            q(x_t | x_{t-1}, x_0) = q(x_t|x_{t-1}) &=  \mathcal{N}(x_{t} ;\sqrt{\alpha_t}x_{t-1}, \beta_t\mathbf{I}) \\
            &\propto \exp\left(-\frac{1}{2}\frac{(x_{t} - \sqrt{\alpha_t}x_{t-1})^2}{\beta_t} \right)
            
            \end{aligned}
            $$
            
        - the 2nd term: using the property
            
            $$
            \begin{aligned}
            q(x_{t-1} | x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, (1 - \bar{\alpha}_{t-1})\mathbf{I}) \propto \exp(-\frac{1}{2}\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1 - \bar{\alpha}_{t-1}})
            \end{aligned}
            $$
            
        - the 3rd term: substitute $t-1$ with $t$ in the 2nd term
        
        The third line:
        
        Using the fact that $ax^2 + bx + C = a(x + \frac{b}{2a})^2$, rearrange the second line with regards to $x_{t-1}$
        
        - $x_{t-1}^2$: $\frac{\alpha_t}{\beta_t} x_{t-1}^2 + \frac{1}{1-\sqrt{\bar{\alpha}_{t-1}}} x^2_{t-1}$. so $a = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\sqrt{\bar{\alpha}_{t-1}}}$.
        - $x_{t-1}$: $(- \frac{2\sqrt{\alpha_t} }{\beta_t}x_t)x_{t-1} + (- \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\sqrt{\bar{\alpha}_{t-1}}} x_0)x_{t-1}$, so $b= - (\frac{2\sqrt{\alpha_t} }{\beta_t}x_t  + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\sqrt{\bar{\alpha}_{t-1}}} x_0)$
        
        and we get to the third line up to a constant independent to $x_{t-1}$, and a factor $1/2$.
        
        Hence, we can get the expressions of the Gaussian parameters of
        
        $$
        q(x_{t-1}|x_t, x_0)= \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta_t}\mathbf{I}) \approx \exp \left( -\frac{(x - \tilde{\mu}(x_t, x_0))^2}{2\tilde{\beta}_t} \right)
        $$
        
        as follow: 
        
        - the variance: $\tilde{\beta}_t = \frac{1}{a}$
        
        $$
        \begin{aligned}
        
        \tilde{\beta}_t 
        
        &= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
        
        = 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
        
        = \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
        
        \\\end{aligned}
        $$
        
        - the mean: $\tilde{\mu}(x_t, x_0) = -\frac{b}{2a}$
        
        $$
        \begin{aligned}
        \tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
        
        &= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
        
        &= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0
        
        \end{aligned}
        $$
        
    
     
    🔥 **Here comes the bold idea!**
    We may use a Neural Network to predict $\bold{x}_0$ using $\bold{x}_t$, namely $\bold{x_0}={\boldsymbol{\Phi}}(\boldsymbol{x}_t)$
    we train the network using the loss $\Vert \boldsymbol{x}_0 - {\boldsymbol{\Phi}}(\boldsymbol{x}_t)\Vert^2$
    after that, we can say $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \approx p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0={\boldsymbol{\Phi}}(\boldsymbol{x}_t))$ , or
    
    </aside>
    
    $$
    p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \approx 
    
    \mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\red{\boldsymbol{\Phi}}(\boldsymbol{x}_t),\frac{\bar{\beta}_{t-1}^2\beta_t^2}{\bar{\beta}_t^2} \boldsymbol{I}\right)
    $$
    
     
    🔥 **Reparameterization Trick**
    As we know
    
    $$
    \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)
    $$
    
    a reasonable set-up of $\bold{\Phi}(\bold{x_t})$ would be
    
    $$
    \bold{\Phi}(\bold{x_t}) = \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\bold{x}_t,t)\big)
    $$
    
    so we need to minimize (in the end)
    
    $$
    \Vert \boldsymbol{x}_0 - {\boldsymbol{\Phi}}(\boldsymbol{x}_t)\Vert^2 = 
    
    \frac{\bar{\beta}_t^2}{\bar{\alpha}_t^2}\left\Vert\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, t)\right\Vert^2
    
    $$
    
    - Detailed Derivation
        
        $$
        \begin{aligned}
        & \Vert \boldsymbol{x}_0 - {\boldsymbol{\Phi}}(\boldsymbol{x}_t)\Vert^2 
        
        \\& = \Vert \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) -  \frac{1}{\sqrt{\bar{\alpha}_t}}\big(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\bold{x}_t,t)\big)\Vert ^2
        
        \\& = \Vert -\frac{1}{\sqrt{\bar{\alpha}_t}}\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t + \frac{1}{\sqrt{\bar{\alpha}_t}} \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\bold{x}_t,t)\big)\Vert ^2
        
        \\& = \frac{1}{{\bar{\alpha}_t}}\Vert -\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\bold{x}_t,t)\big)\Vert ^2
        
        \\& = \frac{\bar{\beta}_t^2}{\bar{\alpha}_t^2}\left\Vert\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, t)\right\Vert^2
        
        \end{aligned}
        $$
        
    </aside>
    
    </aside>
    
- Algorithm 😇
    
    ![Untitled](Untitled%2022.png)
    
    **Problem:** Variance learning, Slow sampling
    

### (Jiaming Song et al, ICLR 2021)  Denoising Diffusion Implicit Model

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

 
🍀 *Jianlin Su* (Engineer*,*ZhuiYi AI) provided in-depth derivation on DDIM.
*Yang Song* (PhD, Stanford)   introduced the connection of Diffusion Model to Score-based Models.

</aside>

- Brief **Summary**
    
    The core part of DDPM is using
    
    $$
    q(\bold{x}_{t-1}|\bold{x}_t,\red{\bold{x}_0})
    
    =q(\bold{x}_{t}|\bold{x}_{t-1},\red{\bold{x}_0})
    
    \frac
    {q(\bold{x}_{t-1}|\red{\bold{x}_0})}{q(\bold{x}_t|\red{\bold{x}_0})}
    $$
    
    to approximate 
    
    $$
    q(\bold{x}_{t-1}|\bold{x}_t)
    $$
    
    What if we don’t know how the image is destroyed?  e.g. Don’t know $q(\bold{x}_{t}|\bold{x}_{t-1})$
    
     
    🔥 **Marginal Distribution**
    
    $$
    \int 
    q(\bold{x}_{t-1}|\bold{x}_t,\red{\bold{x}_0})
    
    q(\bold{x}_t|\red{\bold{x}_0})
    
     d\boldsymbol{x}_t 
    
    = q(\bold{x}_{t-1}|\red{\bold{x}_0})
    $$
    
    </aside>
    
    But…… Still some artifacts
    
     
    😒 **We just set it to be Gaussian Distribution**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2023.png)
    
    </aside>
    
    And most of the model remains unchanged……
    
     
    🧐 **No restriction on** $\red{\sigma_t}$
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2024.png)
    
    </aside>
    
    DDIM = Deterministic DDPM ……
    
     
    😇 **Turns out to be useful**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2025.png)
    
    </aside>
    
    - What is implicit?
        
         
        🔥 Confusions
        In DDIM, the author claims that their model is implict, but this has led to criticize during the open view.
        
        [Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP)
        
        The reviewer comment that:
        
        > “**Intractability of the likelihood** is one of the **defining factors of an implicit model**, evidenced by the fact that the terms implicit and likelihood-free are often used interchangeably, and the fact that the above paper exists to deal with learning in implicit generative models *because* the likelihood is intractable. 
        Your model **has a tractable lower bound on the likelihood** which you use for training, and only becomes deterministic at test time in the limit of a scalar hyperparameter. I also do not understand how a normalizing flow could be described as an implicit model. ”
        > 
        </aside>
        
    
    But they do accelerated the sampling process
    
     
    🔥 **Less steps**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2026.png)
    
    > A Chinese quote “**取法乎上，仅得乎中**”——《帝范》
    > 
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2027.png)
    
    </aside>

> How to generate two images interpolation using generative model?
> 
> An intuitive way: make it noisy again, and interpolate noise?
>
> Guided diffusion: ?

> How GAN interpolate ?

> Three color lnk in a non-newton liquid

### AI4Materials

Element table + spatial group = Crystal structure

- Optimization

Find material strucutre with high free energy and thermo-electrical value.

Boids algorithm: Birds as agents with the following rule:

- Separation: avoid crowding neighbors (short range repulsion)
- Alignment: steer towards average heading of neighbors
- Cohesion: steer towards average position of neighbors (long range attraction)

> The problem of new material design lies in the ability to grow that material.
>
> Synthesis is another proess that can be reformulated as AI problem.
>
>**Automation is the key**

> A new Matterial that does not absorb sun radiation, but only the radiation of the universe background, which makes the temperature lower.

> 众包平台: similar to the ones for protein design and protein folding.

> ## Nature is the best teacher

> "Go for the mess".


## Close speech

> Find prosperious direction
>
> Iteration: 
> Iterate is the key in building something great. Instead of start everything from scratch, we would better build things on top of existing attempts.

Some quotes from Sam Altman (Founder/CEO of OpenAI):

- 越是在年轻的时候努力越划算，努力像利息一样，好处会随着时间越滚越多。年轻时你的义务也越少，抓紧时间找到最舒服的努力方式，最好的是跟一群相处愉快的人一起做有意思有价值的事。

- 可以摘低垂的果实，但也要试着做点不容易的事，一旦你在重要的问题上有些许进展，想帮你的人会络绎不绝。

- 勇敢追寻自己所爱，虽然你多半会失败，并饱尝被拒绝的痛苦。但一旦成功，你会发现事情比想象更顺利。不断提醒自己，乐观会最大化你的运气。