---
layout: post
title: "Diffusion Model"
date: 2033-07-25 13:49:29 +0800
tags: [Research]
---

> 

> What I need to do for this commuinity?
>
> What level of detail should thsi be?
>
> Peer view session.

## Introduction

### "Generative" versus "discriminative" learning

What we mean by saying that a model is "generative"? You may be familiar with the common machine-learning task of recognizing images. A dataset of labeled images of, say cats or dogs, are fed into a neural network. By properly defining a loss function, one can optimize the network parameters by comparing its predictions with the ground-truth labels. This is generally known as the **discriminative learning**. Mathematically, this simply amounts to fitting a function $\boldsymbol{y} = f(\boldsymbol{x})$ or a conditional probability distribution $p(\boldsymbol{y} | \boldsymbol{x})$, where $\boldsymbol{x}$ and $\boldsymbol{y}$ stand for the input data and output labels, respectively.   

<!-- ![image](figs/discriminative.jpeg) -->
![Image](https://pic4.zhimg.com/80/v2-e6cfe18be303f9c82ad8423276e8e4dd.png)

**Generative learning**, on the other hand, goes a step further by considering the joint probability distribution $p(\boldsymbol{x}, \boldsymbol{y})$ of both data and labels. In this way, one can not only perform discrimination tasks as described above, but also **generate new data** from the posterior probability $p(\boldsymbol{x} | \boldsymbol{y})$, hence the origin of the topic name. Generally speaking, generative learning aims at the modeling, sampling and training of high-dimensional probability distributions. Even in the scenario of unsupervised learning where the labels are lacking, one can still obtain useful information by investigating the probabilistic structure $p(\boldsymbol{x})$ of the raw data.

### Basic formulation

We demonstrate the basic idea of generative modeling by considering the task of **density estimation**: given a set of (unlabeled) data $\{\boldsymbol{x}^{(n)}\}_{n=1}^N$, we would like to approximate the underlying distribution $q(\boldsymbol{x})$ of the data by a probabilistic model $p(\boldsymbol{x}; \boldsymbol{\theta})$, where $\boldsymbol{\theta}$ denotes the parameters. To facilitate the training of the model, we need a loss function that measures the dissimilarity between the two distributions. One common choice is the [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), defined by

$$
    \begin{equation}
    	\mathrm{KL}(q||p) = \int d\boldsymbol{x} q(\boldsymbol{x}) \ln\frac{q(\boldsymbol{x})}{p(\boldsymbol{x})}.
	\end{equation}
$$

Using the [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality), it is easy to prove that $\mathrm{KL}(q||p) \geq 0$, where the equality holds only when the two probability distributions $q(\boldsymbol{x})$ and $p(\boldsymbol{x})$ coincide. The ground-truth distribution $q(\boldsymbol{x})$ for the data is generally unknown, and one has to replace it by the empirical density based on the given dataset:

$$
    \begin{equation}
    	q_\textrm{data} (\boldsymbol{x}) = \frac{1}{N} \sum_{n=1}^N \delta(\boldsymbol{x} - \boldsymbol{x}^{(n)}).
    \end{equation}
$$

The loss function $\mathcal{L}(\boldsymbol{\theta})$ is then given by

$$
    \begin{align}
    	\mathcal{L} &= \mathrm{KL}(q_\textrm{data}(\boldsymbol{x}) || p(\boldsymbol{x}; \boldsymbol{\theta})) \\
    	&= -\frac{1}{N} \sum_{n=1}^N \ln p(\boldsymbol{x}^{(n)}; \boldsymbol{\theta}) + \textrm{const}.
    \end{align}
$$

Apart from an unimportant constant with no dependence on the model parameters, this is known as the **negative log-likelihood** of the model; its minimization is well-known in statistics as the **maximum likelihood estimation**. In practice, the optimization of the model can be achieved by computing the gradient $\nabla _{\boldsymbol{\theta}} \mathcal{L}$ of loss function w.r.t. the parameters using the technique of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

Density estimation is a data-driven approach, and it is by no means the only application of generative learning. In the context of scientific applications, for example, people are typically interested in target distributions $q(\boldsymbol{x})$ that are not specified by data, but by certain domain-specific laws inherent to the system considered. Many fundamental principles of nature are formulated in terms of probabilities. To name an example, in the discipline of **statistical mechanics**, it is known that the thermodynamic properties of a classical system is specified by the **Boltzmann distribution** of the underlying microscopic degrees of freedom:

$$
    \begin{equation}
    	q(\boldsymbol{x}) = \frac{1}{Z} e^{-E(\boldsymbol{x}) / k_B T},
    \end{equation}
$$

where $E(\boldsymbol{x})$ is the energy function of the system for certain configuration $\boldsymbol{x}$, $T$ the temperature, $k_B$ the Boltzmann constant , and $Z$ is a normalization constant, which is also known as the partition function. Below we adopt the conventional notation $\beta = 1/k_B T$. Given a probabilistic model $p(\boldsymbol{x}; \boldsymbol{\theta})$, we can then write the loss function as

$$
    \begin{align}
    	\mathcal{L} &= \mathrm{KL}(p(\boldsymbol{x}; \boldsymbol{\theta}) || q(\boldsymbol{x})) \\
    	&= \mathop{\mathbb{E}}_{\boldsymbol{x} \sim p(\boldsymbol{x}; \boldsymbol{\theta})} \left[ \ln p(\boldsymbol{x}; \boldsymbol{\theta}) + \beta E(\boldsymbol{x}) \right] + \ln Z \\
    	&= \beta(F(\boldsymbol{\theta}) - F_0).
    \end{align}
$$

Notice that in the last line we have defined the **variational free energy**

$$
    \begin{equation}
    	F = \mathop{\mathbb{E}}_{\boldsymbol{x} \sim p(\boldsymbol{x}; \boldsymbol{\theta})} \left[ \frac{1}{\beta} \ln p(\boldsymbol{x}; \boldsymbol{\theta}) + E(\boldsymbol{x}) \right],
    \end{equation}
$$

which is an upper bound of the **true free energy** $F_0 = -\frac{1}{\beta} \ln Z$ of the system. This is known as the **Gibbs-Bogoliubov-Feynman variational free-energy principle** in statistical mechanics. As an example, in the next section we will use this variational principle to investigate the thermodynamics of some interacting particles trapped in a harmonic potential, which can be used to describe systems like a quantum dot. This will also give us the opportunity to inspect various practical aspects of generative modeling, such as model architecture design, Monte Carlo sampling, evaluating gradients, optimization algorithms, and so on.

```
aaaNote:aaaa
```

---


# Diffusion Models Inspired From Thermodynamics

<!-- add a to-do list -->
<!-- - [ ]  Implement a Working Model -->

<!-- add a toggle list -->
<!-- <details> -->
<!-- <summary>Table of Contents</summary> -->



Label: Deep Learning, Diffusion Model, Generative Model, Statistical Physics, Thermodynamics, survey
Assign: Osgood Ou
Due: September 16, 2022
Status: Paused
<!-- 
 
🔥 TODOs

- [x]  Implement a Working Model
- [x]  Complete reading about Score-based models
- [ ]  Explicitly tell the connection between DM, Jarzyskin Equality and AIS
- [ ]  Think about application to Physics Problem
</aside> -->

Newest Update: [Practical Model](https://www.notion.so/Practical-Model-486aaf33b9fb4384aa87cf249e4393b3?pvs=21) 

# Author

 
💡 Shigang Ou 
Contact me at [[oushigang19@mails.ucas.ac.cn](mailto:oushigang19@mails.ucas.ac.cn)]

</aside>

# Resource

[Untitled Database](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%20Database%2041eb4a07a6e74d24aa27745b1965d461.csv)

# Introduction 介绍

## SOTA Generative Models

[DALL·E 2](https://openai.com/dall-e-2/)

> DALL·E 2 is a new AI system that can create realistic images and art from a description in natural language.
从自然语言产生逼真的图片和艺术作品
> 

![DALL-E 2: **An astronaut riding a horse in a photorealistic style 骑马的宇航员，逼真的照片**](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled.png)

DALL-E 2: **An astronaut riding a horse in a photorealistic style 骑马的宇航员，逼真的照片**

![DALL-E 2: **Teddy bears mixing sparkling chemicals as mad scientists as a 1990s Saturday morning cartoon 混合化学物质的泰迪熊，上世纪90年代卡通风格**](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%201.png)

DALL-E 2: **Teddy bears mixing sparkling chemicals as mad scientists as a 1990s Saturday morning cartoon 混合化学物质的泰迪熊，上世纪90年代卡通风格**

![DALLE-2: **A bowl of soup that looks like a monster knitted out of wool 毛线织出来的一碗汤，看起来像怪兽**](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%202.png)

DALLE-2: **A bowl of soup that looks like a monster knitted out of wool 毛线织出来的一碗汤，看起来像怪兽**

[Imagen](https://imagen.research.google/)

Google Research, Brain Team: ****unprecedented photorealism × deep level of language understanding
对语言和逼真图片的深度理解和生成****

## Samples

![A strawberry that looks like frog
长得像青蛙的草莓](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%203.png)

A strawberry that looks like frog
长得像青蛙的草莓

![A frog that looks like strawberry
长得像草莓的青蛙](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%204.png)

A frog that looks like strawberry
长得像草莓的青蛙

![A snake that looks like banana
长得像香蕉的蛇](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%205.png)

A snake that looks like banana
长得像香蕉的蛇

![A bald eagle made of chocolate powder, mango, and whipped cream
由巧克力粉、芒果和鲜奶油制成的秃鹰](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%206.png)

A bald eagle made of chocolate powder, mango, and whipped cream
由巧克力粉、芒果和鲜奶油制成的秃鹰

## Any Physics?

![The inner structure of an electron
电子的内部结构](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%207.png)

The inner structure of an electron
电子的内部结构

![Three spheres made of glass falling into ocean. Water is splashing. Sun is setting
三个玻璃球正在坠入海洋,水花四溅,太阳落下](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%208.png)

Three spheres made of glass falling into ocean. Water is splashing. Sun is setting
三个玻璃球正在坠入海洋,水花四溅,太阳落下

![Original Image](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%209.png)

Original Image

Add A Flamingo
添加一只火烈鸟🐦 

✅High Quality Image 

❓High Precision Physics

![Edited Image](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2010.png)

Edited Image

## Physics Informed Neural Network

[Physics-informed neural networks - Wikipedia](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)

 
🔖 The prior knowledge of general physical laws acts in the training of [neural networks](https://en.wikipedia.org/wiki/Neural_network)(NNs) as a regularization agent that limits the space of admissible solutions, increasing the correctness of the function approximation.
物理定律可帮助神经网络减少搜索空间，提高神经网络近似的正确性

</aside>

- **Comment**
    
    Recently, a lot of people working on PINN tries to solve 2-dimensional differential equations with NN. But some people also criticize this trend as such work makes no contribution to theory or practice.
    

 
🔥 Unlike PINN, Diffusion Model get its original idea **directly from Statistical Physics**, making it more attractive.
**Let’s dive into it!**

</aside>

 
👏 **Hands on Session!**
 Interactive example using **Julia**
[Warning] The julia script is excuted in a local machine, contact me for the file if needed [[oushigang19@mails.ucas.ac.cn](mailto:oushigang19@mails.ucas.ac.cn)]

**recommended python notebooks:**

A Diffusion Model from Scratch in Pytorch

[Google Colaboratory](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing)

</aside>

[由浅入深了解Diffusion Model](https://zhuanlan.zhihu.com/p/525106459)

![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2011.png)

# Origins of Idea  想法的来源

## Central Problem in ML 核心问题

 
⚙ A central problem in machine learning involves modeling complex data-sets using highly **flexible** families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally **tractable**.
用足够**灵活(通用)**的模型来建模复杂分布，并确保学习、采样、推理、评估等环节在解析/计算上**可行**。

</aside>

- **Flexibility 🆚 Tractablility 灵活度和可行度**
    - Flexiblility: Can model **arbitrary data**, but can be **analytically intractable.**
    for exmaple, the following Boltzman Machine is flexible, but has an intractable normalization factor $**Z$
    Use Expensive Monte Carol**
    
    $$
    p(x)=\frac{e^{-E(x)}}{Z}
    $$
    
    - Tractablility: Can be **analytically evaluated** and **easily fit** to data
    e.g. Gaussian is tractable, things like KL divergence, log likelyhood can be easily computed, but has low flexibility.
    
    $$
    p_{\text{Gaussian}}(x)=\frac{1}{\sqrt{2\pi}\sigma} \exp \left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
    $$
    

 
🔥 **Diffusion Model is NICE！**
Here, we develop an approach that simultaneously achieves both flexibility and tractability.

</aside>

## First Diffusion Model 扩散模型的首次提出

### (Jascha Sohl-Dickstein et al, 2015)  Nonequilibrium Thermodynamics

[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

> The essential idea, inspired by **non-equilibrium statistical physics**, is to systematically and slowly **destroy structure** in a data distribution through an iterative forward diffusion process.We then learn a reverse diffusion process that restores structure in data, yielding a **highly flexible and tractable generative model** of the data.
> 

> Our method uses a Markov chain to gradually **convert one distribution into another**, an idea used in non-equilibrium statistical physics (Jarzynski, 1997) and sequential Monte Carlo (Neal, 2001).
Related ideas from physics include the Jarzynski equality (Jarzynski, 1997), known in machine learning as Annealed Importance Sampling (AIS) (Neal, 2001), which uses a Markov chain which slowly **converts one distribution into another** to **compute a ratio of normalizing constants**.
> 

## **Physics Origin 物理背景**

### (Neal, 2001) Annealed Importance Sampling

[Annealed Importance Sampling](https://arxiv.org/abs/physics/9803008v2)

> In independent work, **Jarzynski (1997a,b)** has described a method primarily aimed at free energy estimation that is **essentially the same** as the annealed importance sampling method described here.
I will focus instead on statistical applications, and will discuss use of the method for **estimating expectations of functions of state**, as well as the **normalizing constant** $Z$
> 
- **Annealed Importance Sampling**
    
    > This method is inspired by the idea of “**annealing**” as a way of coping with isolated modes, which leads me to call it **annealed importance sampling**.
    > 
    - **Importance Sampling**
        
        ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2012.png)
        
        ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2013.png)
        
        Given a quantity $x$ with distribution probability $\propto f(x)$, we want to calculate the expectation of  a function $a(x)$, this can be done by
        
        $$
        \bar{a}= \frac{1}{\blue{Z_f}}\int dx f(x) a(x)
        $$
        
        where $\blue{Z_f}=\int dx f(x)$ is the normalization factor.  The problem is that we cannot sample from $f(x)$ directly.
        
         In the case of statistical physics, $f(x)=\exp{(-\beta H(x))}$ and $Z_f$ are often intractable.
        
        A general proposal of Importance Samping is using a tractable distribution $g(x)$, with $Z_g=\int dx g(x)$ easy to calculate.
        
        But this will lead to large error, to facilitate with this, we introduce **Importance Weight  $\omega(x):={f(x)}/{g(x)}$**, so
        
        $$
        f(x)=g(x)\omega(x)
        $$
        
        Plug this into the aobve equation, we get
        
        $$
        \begin{aligned}
        \bar{{{a}}}
        %= \frac{Z_f}{{\red{Z_g}}}\int dx g(x) \green{\omega(x)}a(x)
        
        & = \frac{1}{\blue{Z_f}}\int dx f(x) a(x)
        
        \\& = \frac{1}{\int dx f(x)}\int dx f(x) a(x)
        
        \\ & =\frac{1}{\int dx g(x) 
        \green{\omega(x)}}\int dx g(x) \green{\omega(x)}a(x)
        
        \\ & \approx \frac{\sum_i \green{\omega(x_i)}a(x_i)}{\sum_i \green{\omega(x_i)}}
        \end{aligned}
        $$
        
        where $x_i$ is sampled from $g(x)$.
        
        one interesting property is $\sum_i \omega(x_i) /N =Z_f/Z_g$
        
        Cons: If $g(x)$ is not close to $f(x)$, the weight $\green{\omega(x)}$ varies violently, leading to significant errors.
        
    - **Metropolis-Hastings algorithm**
        
        (Metropolis, et al 1953; Hastings 1970) proposed the well-known Metropolis-Hasting algorithm using MCMC.
        
         
        💡 This algorithm designed a Markov Chain to sample from certain distribution
        
        $$
        \frac{dp_a}{dt}=0
        $$
        
        and
        
        $$
        \frac{dp_a}{dt}=\sum_b (\omega_{b \to a}p_b - \omega_{a\to b}p_a)
        $$
        
        so
        
        $$
        \sum_b (\omega_{b \to a}p_b - \omega_{a\to b}p_a) =0
        $$
        
        Detailed balance satisfies this 
        
        $$
        \omega_{b \to a}p_b = \omega_{a\to b}p_a  \to \\\omega_{b\to a}/\omega_{a\to b}=p_a/p_b
        $$
        
        Consider metropolis step to generate $p(x)\propto \exp[-\beta E(x)]$
        
        using the metropolis creterion, we can derive that
        
        $$
        \omega_{b\to a}/\omega_{a\to b}=p(x_a)/p(x_b)
        $$
        
        Using this, we can design a Markov process and sample from the distribution $p$, get $x_n$, and the estimation would be
        
        $$
        F_N=\frac{1}{N}\sum_n^N F(x_n)
        $$
        
        </aside>
        
    
    - **Simulated Annealing**
        
        Kirkpatrick, Gelatt, and Vecchi (1983) introduced Simulated Annealing to solve the problem of MH algorithm
        
         
        💡 We want to sample distribution $p_0$, but
        
        - start from $p_n=p_0^{\beta_n}$ using MH algorithm
        - after a few iterations, switch from $j$ to $j-1$, and repeat the above process from final state
        
        $1=\beta_0 > \beta_1 > \cdots > \beta_n$
        
        </aside>
        
    
    ---
    
     
    💡 **Assumptions**
    Given a quantity $x$ with unknown distribution probability $p_0(x)$
    
    - **(Sequence)** We have available a sequence of other distributions, given by $p_1,\cdots, p_n$, and $p_{j-1}(x)\ne 0$  wherever $p_{j}(x)\ne 0$
    - **(Evaluate)** For each distribution, we must be able to compute some function $f_j(x)$ that is proportional to  $p_j(x)$
    - **(Markov)** For each $i$ from $1$ to $n − 1$, we must be able to simulate some Markov chain transition, $T_j$, that leaves $p_j$ invariant.
    - **(Sample)** $p_n$ can be sampled easily
    </aside>
    
    A general contruction of such sequence can be
    
    $$
    f_j(x)=f_0(x)^{\beta_j}f_n(x)^{1-\beta_j}
    $$
    
    where $1=\beta_0 > \beta_1 > \cdots > \beta_n=0$
    
     
    🚧 **Exercise: Show that Simulated Annealing is a special case for Annealed Importance Sampling**
    
    </aside>
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2014.png)
    
    To prove this, simply conduct importance sampling in the extended space $(x_0,x_1,\cdots,x_{n-1})$
    
     
    🚧 **Exercise: Prove that Annealed Importance Sampling is Importance Sampling in the extended space $(x_0,x_1,\cdots,x_n)$**
    
    </aside>
    

### (Jarzynski, 1997) Free Energy Estimation

$F = -k_BT\ln{Z}$

[Equilibrium free energy differences from nonequilibrium measurements: a master equation approach](https://arxiv.org/abs/cond-mat/9707325)

> In has recently been shown that the Helmholtz free-energy difference between two equilibrium configurations of a system may be obtained **from an ensemble of finite-time nonequilibrium measurements** of the work performed in switching an external parameter of the system. Here this result is established, as an identity, within the master equation formalism.
> 
- **Jarzynski Equality  $\braket{\exp{(-\beta W)}}=\exp{(-\beta F)}$**
    
    Finite classical system that depends on an external parameter $\lambda$
    
    - a lattice of coupled classical spins, the strength of an externally applied magnetic field is $\lambda$
    - classical gas in a finite box, the volume of a box is $\lambda$
    
    Allowing the system to come to equilibrium with a heat reservoir at temperature T
    
    Switch the external parameter$\lambda$, infinitely slowly, from an initial $\lambda = 0$ to a final  value $\lambda=1$.  
    
    The system remain in **quasistatic equilibrium** with the reservoir throughout the switching process, we have
    
    $$
    W_{\red{\infty}} = \Delta F = F_1 - F_0
    $$
    
    Here $\red{\infty}$ indicates that we are going through infinite numbers of quasi-static process.
    
    But in reality, we can only switch finite times (namely $t_s$ times), so each time we modify $\lambda$ by $\Delta \lambda =t_s^{-1}$, the above equation then becames inequality
    
    $$
    \red{\braket{W}}\ge \Delta F
    $$
    
    where $\braket{\cdot}$ is the expectation over the ensemble.
    
     
    🔥 Jarzynski proved that the following result is valid for any switching time $t_s$ (Jarzynski Equality)
    
    $$
    \braket{\exp{(-\beta W)}}=\exp{(-\beta F)}
    $$
    
    </aside>
    
    > Finally, one might come away from Ref[4] with the feeling that the validity of this equation  depends directly on the properties of Hamiltonian evolution, in particular Liouville’s theorem. The treatment herein dispels this notion: **Hamilton’s equations appear nowhere in the derivation.** This point is particularly relevant in the context of numerical simulations, where the evolution of a thermostated system is often realized with the use of non-Hamiltonian equations of motion.
    > 
    - **Derivation of Jarzynski Equality(General Case)**
        
        ### Preliminaries
        
        $V$: **a space of variables** which fully describes the instaneous microscopic states at each point
        
        $\bold{z}(t)$: evolution **trajectory** of a system in phase space $**V$,** if  the system is in contact with a heat bath, then such trajectories is **stochastic.**
        
        $H_\lambda (\bold{z})$: A function that gives the total energy of the system instantaneous state $\bold{z}$, with external parameter $\lambda$.
        
        With the above notation, we define
        
        $$
        Z_\lambda(\beta)=\int d\bold{z}\exp{[-\beta H_\lambda (\bold{z})]}\\F_\lambda(\beta)=-\beta^{-1}\ln{Z_\lambda(\beta)}
        $$
        
        where $\beta$ is simply a real, positive number.
        
        We’ll focus on one thing in the following derivation:
        
        $$
        \Delta F(\beta)=F_1(\beta) -F_0(\beta)
        $$
        
         
        💡 **Assumption 1 (Uniform Switch)**
        
        The time dependence of external parameter is simply assumed to be uniform within time  $t \in [0,t_s]$
        
        $$
        \lambda(t)=t/t_s
        $$
        
        </aside>
        
        If we are given the trajectory $\bold{z}(t)$ , then **work**  $W$can be defined as
        
        $$
        W=\int_0^{t_s}dt \dot{\lambda }\frac{\partial H_{\lambda} \big(\bold{z}(t)\big)}{\partial {\lambda}}
        $$
        
        where $\dot{\lambda}=d\lambda /dt = t^{-1}_s$
        
        - Special case when $H_\lambda$  obeys Hamiltonian equation
            
            when $H$ is the Hamiltonian satisfying the Hamilton Equation:
            
            $$
            \dot{\bold{z}}=\{ \bold{z},H_\lambda\}
            $$
            
             
            🚧 **Exercise**
            Prove This for Hamiltonian Evolution
            
            </aside>
            
            $$
            W=H_1(\bold{z}(t_s))-H_0(\bold{z}(0))
            $$
            
        
         
        💡 **Assumption 2(Markov Process)**
        The evolution of our system in phase space is a Markov process with transition probability from $\bold{z'}\to \bold{z}$
        
        $$
        P(\bold{z}',t|\bold{z},t+\Delta t)
        $$
        
        </aside>
        
        we further define instantaneous transition rate
        
        $$
        R(\bold{z}',\bold{z};t)=\frac{\partial }{\partial (\Delta t)}P(\bold{z}',t|\bold{z},t+\Delta t)
        $$
        
        We can see that the dependence of $R$ on time $t$ is only through $\lambda(t)$, so we rewrite
        
        $$
        R(\bold{z}',\bold{z};t)\to R_{\lambda}(\bold{z}',\bold{z})
        $$
        
        ---
        
         
        👏 Now we have the discription for a single system, Let’s switch to an ensemble of systems!
        
        </aside>
        
        Ensemble is one core concept in statistical mechanics, where each system evolves indipendently. To describe it, we define the **time dependent distribution of the ensemble** in phase space as $f(\bold{z},t)$, then the distribution obeys
        
        $$
        \frac{\partial f}{\partial t}(\bold{z},t)=\int d\bold{z'}f(\bold{z'},t)R_\lambda (\bold{z'},\bold{z})
        $$
        
        we use a linear operator to rewrite this neatly as 
        
        $$
        \frac{\partial f}{\partial t}=\hat{R}_\lambda f
        $$
        
        This is called the MASTER EQUATION for our case.
        
         
        💡 **Assumption 3(Detailed Balance)**
        
        If the external parameter $\lambda$ is fixed, then $\bold{z}(t)$ will undergo a stochastic MC process satisfying the **detailed balanced condition** (in contact with a heat reervoir)
        
        $$
        \frac{R_\lambda (\bold{z'},\bold{z})}{R_\lambda (\bold{z},\bold{z'})}=\frac{\exp{[-\beta H_\lambda (\bold{z})]}}{\exp{[-\beta H_\lambda (\bold{z'})]}}
        $$
        
        or in a more neat format
        
        $$
        \hat{R}_\lambda\exp{[-\beta H_\lambda (\bold{z})]}=0
        $$
        
        </aside>
        
        Note that, in Thermodynamics, one often need another assumption:
        
         
        💡 **Assumption 4(Thermalization)**
        
        $$
        \lim_{t\to \infty}{\hat{U}_\lambda(t)f_0(z)}=\frac{1}{Z_\lambda}\exp{[-\beta H_\lambda (\bold{z})]}
        $$
        
        where the operator $\hat{U}_\lambda(t)=\exp{(\hat{R}_\lambda t)}$ is the evolution operator derived from ${\partial f}/{\partial t}=\hat{R}_\lambda f$
        
        This is to say that given an arbitrary initial distribuiton $f_0$, after enough time, the system will thermolize into a canonical distribution.(Which, naturally lead to the Assumption 3)
        
        </aside>
        
        But it turns out that to prove Jarzynski Equality, we don’t need **Assumption 4.**
        
         
        😎 SO EASY for Physics students! Let’s get start to derive the Jarzynski equality!
        
        </aside>
        
        ### Derivation
        
        Assuming that when $\lambda =0$, our ensemble equilibrates with the reservoir, we have
        
        $$
        f(\bold{z},0)=Z_0^{-1}\exp[{-\beta H_0(\bold{z})}]
        $$
        
        If we went through a quasi-static process, we’ll have the solution for MASTER EQUATION as (recall the property of Detailed Balance)
        
        $$
        f=Z_\lambda ^{-1} \exp{(-\beta H_\lambda)}
        $$
        
        But in our finite switching step, $f$ heavily depends on our time dependent $\lambda(t)$. What we are trying to do is evaluation of $\braket{\exp{(-\beta W)}}$ for all trajectories $\bold{z}(t)$ in the ensemble. We define 
        
        $$
        w(t)=\int_0^t dt' \dot{\lambda} \frac{\partial H_{\lambda} \big(\bold{z}(t')\big)}{\partial {\lambda}}
        $$
        
        so the work we previously defined is
        
        $$
        W=w(t_s)
        $$
        
        We define a new quantity 
        
        $$
        Q(\bold{z},t)=\braket{\exp{}[-\beta w(t)]}_{\text{subset of trajectories passing }\bold{z}}:=\braket{\exp{}[-\beta w(t)]}_\bold{z}
        $$
        
        Finally, define
        
        $$
        g(\bold{z},t)=f(\bold{z},t)Q(\bold{z},t)
        $$
        
        we have the expression of $\braket{\exp{(-\beta W)}}$ as
        
        $$
        \braket{\exp{(-\beta W)}} = \int d\bold{z}g(\bold{z},t_s)
        $$
        
        What is left here is simply finding $g$ and prove that this integral equals $\exp{(-\beta \Delta F)}$
        
        - **What’s the meaning of** $g(\bold{z},t)$ ?
            
             
            ❓ **How to get the above result?**
            
            </aside>
            
            A nice way to understand the meaning is to imagine each trajectory $\bold{z}(t)$ to be a ‘particle’ moving in the phase space, the ‘mass’ of each particle is time dependent and given by $\mu(t)=\exp{[-\beta w(t)]}$. (Begin with 1, decreasing gradually)
            
            $Q(\bold{z},t)$: Average Mass of particles found at point $\bold{z}$ at time $t$
            
            $g(\bold{z},t)$: time dependent mass density in phase space (normalized with $\int fd\bold{z}=1$)
            
            Integrating $g$ over the phase space yields the total mass of particles at time $t_s$, which is $\braket{\exp{(-\beta W)}}$.
            
             
            🔥 **Bonus Time**
            
            $g(\bold{z},t)$ can be solved using equation
            
            $$
            \frac{\partial g}{\partial t}=\Big( \hat{R}_\lambda - \beta \dot{\lambda} \frac{\partial H_{\lambda} }{\partial {\lambda}} \Big)g
            $$
            
            The mass density at $\bold{z}$ in time $t$ changes for two reasons
            
            - particle number density change $\blue{\dot{f}}=\hat{R}_\lambda f$
            - Instantanous mass change $\red{\dot{\mu}(t)}=- \beta \dot{\lambda} \frac{\partial H_{\lambda} }{\partial {\lambda}} \mu(t)$
            
            since
            
            $$
            g(\bold{z},t_s) \\ = f(\bold{z},t_s)Q(\bold{z},t_s)\\=\blue{f(\bold{z},t_s)}\red{\braket{\mu(t_s)}}_{\bold{z}}
            $$
            
            Given 
            
            $$
            g(\bold{z},0)=f(\bold{z},0)=Z_0^{-1}\exp[{-\beta H_0(\bold{z})}]
            $$
            
            we solve the equation and get
            
            $$
            g(\bold{z},t)=Z_0^{-1}\exp[{-\beta H_\lambda(\bold{z})}]
            $$
            
             
            🚧 **Exercise**
            Solve the above equation
            
            </aside>
            
            </aside>
            
        
        Since $g(\bold{z},t)=Z_0^{-1}\exp[{-\beta H_\lambda(\bold{z})}]$, we get
        
        $$
        \begin{aligned} 
        &\braket{\exp{(-\beta W)}} 
        
        \\& = \int d\bold{z}g(\bold{z},t_s)
        
        \\& =\int d\bold{z}Z_0^{-1}\exp[{-\beta H_{\lambda(t_s)}(\bold{z})}]
        
        \\& =Z_0^{-1}\int d\bold{z}\exp[{-\beta H_{1}(\bold{z})}]
        
        \\& =Z_1/Z_0
        
        \\& =\exp{(-\beta \Delta F)}
        
        \end{aligned}
        $$
        

 
🔥 **One Cool Thing**
The Jarzynski equality holds **BOTH** for Hamiltonian Revolution and Non-Hamiltonian process(e.g. **Langevin** evolution, **monte carol** evolution, molecular dynamics).

</aside>

### (Burda, Y., Grosse, R. B., and Salakhutdinov, R., 2014)   Lower Bound of AIS

[Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing](https://arxiv.org/abs/1412.8566)

> Annealed importance sampling (AIS) is widely used to estimate Markov Random Field partition functions, and often yields quite accurate results. However, AIS is **prone to overestimate the log-likelihood with little indication that anything is wrong**.
We present Reverse AIS Estimator (RAISE), a stochastic **lower bound on the log-likelihood** of an approximation to the original MRF model.
> 

 
🔥 **Lower Bound is important!**
by using RAISE and AIS in conjunction, one can judge the accuracy of one’s results by measuring the agreement of the two estiatmators.

</aside>

# Rise in Deep Learning 深度学习中的兴起

- **Strongly Related Articles**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2015.png)
    

- **Connected Papers**
    
    [Connected Papers | Find and explore academic papers](https://www.connectedpapers.com/main/2dcef55a07f8607a819c21fe84131ea269cc2e3c/Deep-Unsupervised-Learning-using-Nonequilibrium-Thermodynamics/graph)
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2016.png)
    

 
🍀 **Sincere Thanks**
The following content is highly motivated by the **blogs** of
 *Lilian Weng* (Applied AI Research Manager, OpenAI)  
*Zaixiang Zheng* (Research Scientist, ByteDance AI Lab)
*Jianlin Su* (Engineer*,*ZhuiYi AI)
*Yang Song* (PhD, Stanford)

</aside>

[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[科学空间|Scientific Spaces](https://kexue.fm/)

[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)

## Denoising Diffusion Model 去噪扩散概率模型

### (Ho et al, NeurIPS 2020)  Denoising Diffusion Probabilistic Model

[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

 
🍀 *Lilian Weng* (Applied AI Research Manager, OpenAI)  gave  a detailed introduction on the theory of DDPM.

*Zaixiang Zheng* (Research Scientist, ByteDance AI Lab) **provided many derivation details on DDPM.

*Jianlin Su* (Engineer*,*ZhuiYi AI) offered excellent derivation and comment on DDPM.

</aside>

- Big Picture of Generative Models: **GAN VAE Flow Diffusion**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2017.png)
    
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
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2018.png)
    
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
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2022.png)
    
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
    

## Score-based Diffusion Model 基于分数的扩散模型

### (Yang Song et al, ICLR 2021) Stochastic Differential Equations

[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

 
🍀 *Yang Song* (PhD, Stanford)   introduced the connection of Diffusion Model to Score-based Models.
*Jianlin Su* (Engineer*,*ZhuiYi AI) provided summary on  Score-based Models.

</aside>

- Big Picture of Generative Models (Once again)
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2028.png)
    
    **Likehood based(Explicit)**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2029.png)
    
    **Implicit (Sample from Noise without knowing the probability)**
    
     
    🚧 **Exercise**
    Is Diffusion Model explicit or implicit ?
    ????
    
    </aside>
    
- Score Function, or $\nabla \log{p}$
    
    As previously discussed, all Generative tasks boils down to **learn** probability and **sample** from it, sample can be done using ‘direct sampling’ ‘monte carol’, which lead to different models.
    
     
    🔥 Langevin Dynamics Sampling
    
    $$
    \mathbf{x}_{i+1} \gets \mathbf{x}_i + \epsilon \nabla_\mathbf{x} \log p(\mathbf{x}) + \sqrt{2\epsilon}~ \mathbf{z}_i, \quad i=0,1,\cdots, K, \mathbf{z}_i \sim \mathcal{N}(0, I)
    $$
    
    ![https://yang-song.net/assets/img/score/langevin.gif](https://yang-song.net/assets/img/score/langevin.gif)
    
    </aside>
    
    Learn $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_\mathbf{x} \log p(\mathbf{x})$ and we see 
    
    $$
     \mathbf{s}_\theta (\mathbf{x}) = \nabla_{\mathbf{x}} \log p_\theta (\mathbf{x} ) = -\nabla_{\mathbf{x}} f_\theta (\mathbf{x}) - \red{\underbrace{\nabla_\mathbf{x} \log Z_\theta}_{=0} }= -\nabla_\mathbf{x} f_\theta(\mathbf{x}). 
    $$
    
- Stohastical Differential Equation
    
     
    🔥 **Introduce Diffusion Model in one slide**
    
    ![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2030.png)
    
    </aside>
    

### Continuous Normalizing Flow

Given data $x=(x^1,\cdots, x^d) \in \mathbb{R}^d$, probability density path $p(t,x):[0,1] \times \mathbb{R}^d\to \mathbb{R}_{> 0}$ where $t\in [0,1]$.

We can design a time dependent vector field $v(t,x):[0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ and use this, we can transform $x$  via

$$
\frac{dx}{dt}=v(x,t)\\
\frac{d\ln p(x,t)}{dt}= - \nabla \cdot v
$$

This would give 

$$
x_1 = \int_{0}^1 dt v(x,t)
$$

![Untitled](Diffusion%20Models%20Inspired%20From%20Thermodynamics%205f9340ba64864202acdf9d42e8e13b32/Untitled%2031.png)

Our goal is to train $v_\theta(x,t)$ , this is done via

$$
L_{FM}=\mathbb{E}_{p(x,t)}|v_\theta - u(x)|
$$

## Flow Matching

几个问题:

1. 如何统一NF和DM？
    
    考虑连续的方程，essentially velocity
    
    如何统一所有的模型：极大似然
    
    在Flow mactching 中失败了，或许从fisher divergence出发会更好。
    
    对深度学习的理解：Loss function
    
    巨大搜索空间、高效模拟器/大量数据、明确的目标函数、问题的痛点
    
    NP-Hard，实际价值
    
    系综学习：多个同时测量
    
2. how to design velocity?
    
    Any prior knowledge can we impliment?
    
3. Application in Biology
    
    Phase space, Free energy difference, molecular generation, big molecular
    
4. What makes this faster?
    - [ ]  Will it be faster?
        - [ ]  Will it be faster than before.
        - [ ]  Sampling is slow? High resolution means accurate result?
5. Essential Idea
    - Layer-wise
    - Conditional
6. Score function and diffusion model

# Practical Model

Here we review the SOTA implementation of Diffusion Model in keras.

[Keras Implementation of Stable Diffusion](https://www.notion.so/Keras-Implementation-of-Stable-Diffusion-301e335c600144cd86bdbfb4b7f7914d?pvs=21)