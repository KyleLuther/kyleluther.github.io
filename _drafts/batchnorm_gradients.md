---
layout: post
title:  "Batch Norm Causes Exploding Gradients"
date:   2020-01-23 23:01:52 -0500
---

$$
  \def\llangle{\langle \! \langle}
  \def\rrangle{\rangle \! \rangle}
  \def\lllangle{\left \langle \!\!\! \left \langle}
  \def\rrrangle{\right \rangle \!\!\! \right \rangle}
$$

<!-- $$ \require{MnSymbol} $$ -->

$$ \require{cancel} $$

Almost everyone who trains deep nets seems to love [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf). Understandably so, as it regularly reduces training times, sensitivity to parameter initialization, and sensitivity to learning rate selection. Basically it helps "fool-proof" deep network training. And as an added benefit, it often improves final test-time performance.

However, as with many aspects of deep learning, a precise reason *why* Batch Norm confers these benefits remains elusive. [Recent work](https://openreview.net/pdf?id=SyMDXnCcF7) by Yang et. al. has taken one step forwards by examining the behavior of Batch Norm in random networks, i.e. at initialization time. They show that inserting Batch Norm into a network can actually cause gradients to explode exponentially with depth, at least at initialization time. This result may seem pretty surprising; having any quantity explode or decay exponentially with depth would only seem to make training harder, not easier.

In this post, we are going to provide an alternate derivation for the exploding gradient observation. Now, if you would be satisfied with nothing less than a perfectly rigorous, detailed, and lengthy mathematical derivation of this phenomenon you should check out their [paper](https://openreview.net/pdf?id=SyMDXnCcF7). If you're like me and just want a more intuitive explanation (but still gives quantitative predictions), stick around for the rest of this post!

(Ok I kid, I would still recommend the paper to those interested, but it is quite technical and 95 pages long with the appendix)

**TL;DR** Inserting Batch Norm into a network means that in the forward pass each neuron is divided by its standard deviation, $\sigma$, computed over a minibatch of samples. In the backward pass, gradients get divided by the same $\sigma$. In networks with ReLU nonlinearity, we can actually approximate the standard deviation as $\sigma \approx \sqrt{\frac{\pi}{\pi-1}} \approx 0.82$, at least at initialization time when the weights are random. Since this occurs at every layer, gradient norms in early layers are roughly $\left(\frac{1}{0.82}\right)^L$ times larger than gradient norms in layer $L$.

## Numerical Simulation (Gradients really do explode)
Before our calculation, let's just empirically compute gradients in a net with and without Batch Norm.

Our "vanilla net", i.e. our net without Batch Norm, will be a boring old 10 layer, feedforward, fully connected network of uniform layer width $N=1024$. We're using the ReLU nonlinearity so we'll initialize weights with [Kaiming initialization](https://arxiv.org/pdf/1502.01852.pdf) (sample every element of $W$ from a zero mean Gaussian with variance $\sqrt{2/N}$).

<p align="center">
  <img width="600" height="200" src="/assets/vanilla_net.png">
</p>

Our inputs to the net will be 512 Gaussian white noise vectors $\{ \mathbf{x}_0(t): t = 1,2,...,512 \}$ . We'll compute a linear loss over the network's outputs: $E = \sum_{t=1}^{512} \mathbf{w} \cdot \mathbf{x}_{10}(t)$. We choose $\mathbf{w}$ by drawing it from a unit Gaussian. Now we have everything we need to compute the gradients $\frac{dE}{d\mathbf{x}_l}$ at each layer $l$ in the vanilla network (We'll use Pytorch to automate this process for us).

Now let's take that exact network, loss, and set of input vectors, but insert Batch Normalization before the ReLU in every layer:

<p align="center">
  <img width="800" height="200" src="/assets/bn_net.png">
</p>

Again we can use PyTorch to compute gradients $\frac{dE}{d\mathbf{x}_l}$ at each layer $l$. Now for the results, let's visualize the gradient histograms in each layer for the Vanilla and Batch Norm net:

<p align="center">
  <img width="400" height="200" src="/assets/bn_gradient_simulation.png">
</p>

In the Vanilla network, the gradient histograms are basically the same in every layer. However, after inserting Batch Norm, the gradients actually grow with decreasing layer. In fact, the widths of these histograms grow *exponentially*. Even in this extremely simplified setting Batch Norm is doing something wierd...

## Gradients in a Batch Normalized Net
Let's review the forward and backward pass for a Batch Normalized network. For those familiar, just look at the diagrams to understand the notation and move on to the "typical" gradient section.

#### Forward Pass
<p align="center">
  <img width="400" height="150" src="/assets/bn_forward.png">
</p>

The diagram above shows the 3 core modules comprising each layer: Matrix Multiply, Batch Norm, and ReLU Nonlinearity.

$\mathbf{x}_l, \mathbf{y}_l$ and  $\mathbf{z}_l$ denote the vector outputs of each module for a single input. The [element-wise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) is denoted by $\mathbf{a} \circ \mathbf{b}$. The notation $\langle \cdot \rangle$ indicates a *minibatch average* so $\boldsymbol{\mu}_l$ and $\boldsymbol{\sigma}^2_l$ are the mean and variance of each element of $\mathbf{y}_l$ over a minibatch. Note these dependent on the minibatch!


<!-- , so $(\mathbf{a} \circ \mathbf{b})_i = (\mathbf{a})_i (\mathbf{b})_i$ -->

Because we are only doing an initialize time analysis we have made two simplifications. One, we ignore bias addition in the matrix multiply layer as biases are initialized to zero. Two, we can ignore the final part of Batch Norm which performs the transform $\mathbf{z}_l \leftarrow \boldsymbol{\gamma}_l \circ \mathbf{z}_l + \boldsymbol{\beta}_l$ because $\boldsymbol{\gamma}_l=\boldsymbol{1}$ and $\boldsymbol{\beta}_l=\boldsymbol{0}$ at initialization.



#### Backward Pass
<p align="center">
  <img width="400" height="150" src="/assets/bn_backward.png">
</p>
The diagram above shows the backwards pass through each module. Recall that the backwards pass tells us how to compute gradients with respect to a module's inputs, given the gradients with respect to the module's outputs.

In each module, the backwards pass can be derived using the chain rule, though in the case of Batch Norm it gets rather tedious. Fortunately the Batch Norm authors give us the result, which I have simplified in the diagram above. If you want more details in actually calculating the backwards pass through a Batch Norm layer, check out this [blog post](http://cthorey.github.io/backpropagation/)!

Before moving on, we'll highlight two superficial similarities  between the forwards and backwards pass through a Batch Norm module. One, division by $\boldsymbol{\sigma}_l$ occurs in each case. Two, in the same way Batch Norm centers the forwards pass, $\langle \mathbf{z}_l \rangle = \boldsymbol{0}$, it also centers gradients in the backwards pass $\langle \frac{dE}{d\mathbf{y}} \rangle = \boldsymbol{0}$.
<!--
$$\langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \cancelto{0}{\langle \frac{dE}{d\mathbf{z}} - \langle \frac{dE}{d\mathbf{z}} \rangle \rangle} - \frac{1}{\sigma} \cancelto{0}{\langle \mathbf{z} \rangle} \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$ -->

## *Typical* Gradients in a Batch Normalized Net
The backwards pass equations through a Batch Norm module are probably not the most enlightening. It's probably unclear how they relate to those histograms we plotted earlier.

Notably they depend on the precise configuration of weights. What we really want to know is *typical* behavior of the backwards pass, over the distribution of networks. Specifically we want:

$$ \text{Typical Squared Gradient: } \lllangle \left( \frac{dE}{dx_l}\right)^2 \rrrangle $$

The double brackets $\llangle \cdot \rrangle$ indicate an average over *weights*. Also note that we are really looking at at single element of the vector $\mathbf{x}$.

Conceptually the only thing we have to do to compute the typical squared gradient is average the backwards pass over weights. This is relatively simply for the matrix multiply and ReLU modules. Batch Norm will take some effort.

In this section and the ReLU section, we are just rederiving the results originally presented the original Kaiming initialization [paper](https://arxiv.org/pdf/1502.01852.pdf). Feel free to skip to the Batch Norm module if you're already familiar with these results. If you want more in depth tutorial on this here is a nice [blog post](https://pouannes.github.io/blog/initialization/).

#### Matrix Multiply Module:
Consider the backwards pass through a matrix multiply module $\frac{dE}{dx_i} = \sum_{j} W_{ji} \frac{dE}{dy_j}$. We have omitted layer indices $l$ and added vector indices $i,j$. Now, on to the "typical backpropagation" equation.

First, square both sides and average over weight configurations:

$$ \lllangle \left(\frac{dE}{dx_i}\right)^2 \rrrangle  = \sum_{jk} \lllangle W_{ji} W_{ki} \frac{dE}{dy_j} \frac{dE}{dy_k} \rrrangle $$

Second, separate the $W$ terms and $dE/dy$ terms using the gradient independence assumption (which states that forward pass quantities are independent from backwards pass quantities):

$$ \lllangle \left(\frac{dE}{dx_i}\right)^2 \rrrangle  = \sum_{jk} \llangle W_{ji} W_{ki} \rrangle \lllangle \frac{dE}{dy_j} \frac{dE}{dy_k} \rrrangle $$

Third, simplify the sum using the fact that $W_{ij}$ are independent zero mean variables with variance $2/N$ so $\llangle W_{ji} W_{ki} \rrangle = (2/N)$ if $j=k$ and 0 otherwise

$$ \lllangle \left(\frac{dE}{dx_i}\right)^2 \rrrangle  = (2/N) \sum_{j} \lllangle \left(\frac{dE}{dy_j}\right)^2 \rrrangle $$

Fourth, use the fact that the typical squared gradient at each layer is the same for all elements. This follows from the identical distribution of $W$.

$$ \boxed{ \lllangle \left(\frac{dE}{dx_{l-1}}\right)^2 \rrrangle = 2 \lllangle \left(\frac{dE}{dy_l}\right)^2 \rrrangle }  $$

#### ReLU Module:
Consider the backwards pass through the ReLU module $\frac{dE}{dz_i} = f'(z_i) \frac{dE}{dx_i}$. Let's procede like before.

**Step 1:** Square both sides and average over weights:

$$ \lllangle \left(\frac{dE}{dz_i}\right)^2 \rrrangle = \lllangle f'(z_i)^2  \left( \frac{dE}{dx_i} \right)^2 \rrrangle $$

**Step 2:** Separate the $f'$ terms and $dE/dx$ terms using the gradient independence assumption:

$$ \lllangle \left(\frac{dE}{dz_i}\right)^2 \rrrangle = \llangle f'(z_i)^2 \rrangle  \lllangle \left( \frac{dE}{dx_i} \right)^2 \rrrangle $$

**Step 3:** ReLU is a particularly simple activation. If $z_i > 0$, then $f' = 1$. Otherwise $f'=0$. This is great, since over weight configurations $z_i>0$ half the time and $z_i < 0$ the other half. So the average $\llangle f'(z_i)^2 \rrangle = 1/2$:

$$ \boxed{ \lllangle \left(\frac{dE}{dz_l}\right)^2 \rrrangle = \frac{1}{2} \lllangle \left( \frac{dE}{dx_l} \right)^2 \rrrangle } $$


#### Batch Norm Module:
In some sense, 2/3 of our work is done; in principle we just need to average the squared backwards pass equations over weights like we did before. However in a much more real sense, most of the work is still to come.

The first complication is the gradients that come from backpropagating through the minibatch mean $\mu$ and variance $\sigma$. We will make the approximation that these gradients are nearly zero, at least if there is enough "randomness" in the data and the networks are sufficiently wide.

The second complication is the division by $\sigma$, the minibatch standard deviation. One might hope it is close enough to one that we could forget about it. Unfortunately this is not so and since it is an average over samples, not weights, it will take some assumptions (and labor) to calculate.

##### Approximation 1: $\langle z \frac{dE}{dz} \rangle \approx 0$
We are going to argue that $\langle z \frac{dE}{dz} \rangle$, the term from backpropagating through $\sigma$, is nearly zero at initialization time.

To do so we'll extend the gradient independence assumption to apply over the minibatch distribution. This means that for a typical configuration of weights, forward pass quantities are independent from backward pass quantities over the minibatch distribution. With this assumption we have

$$\langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle \approx \langle \mathbf{z} \rangle \circ \langle \frac{dE}{d\mathbf{z}} \rangle = \mathbf{0}$$

The last equality follows from the fact that $\langle \mathbf{z} \rangle = \mathbf{0}$.

##### Approximation 2: $\langle \frac{dE}{dz} \rangle \approx 0$
Now we'll argue that $\langle \frac{dE}{dz} \rangle$, the term from backpropagating through $\mu$, is nearly zero at initialization time, except possibly in the last layer.

Recall that Batch Norm ensures gradients with respect to $\mathbf{y}$ are exactly zero mean: $\langle \frac{dE}{d\mathbf{y}} \rangle = \mathbf{0}$. We'll argue that this implies $\langle \frac{dE}{dz} \rangle$ is also zero mean. With our backwards pass equations, we can write gradients for $\mathbf{z}_l$ in terms of gradients for $\mathbf{y}_{l+1}$: $\frac{dE}{d\mathbf{z_l}} = f'(\mathbf{z_l}) \circ \mathbf{W}_l^T \frac{dE} {d\mathbf{y}_{l+1}}$.

Now we average both sides of this equation over samples in the minibatch. If we believe in the extended gradient independence assumption, then $f'(\mathbf{z_l})$ and $\frac{dE} {d\mathbf{y}_{l+1}}$ are independent quantities so we can simplify the minibatch-averaged sum:

$$ \left\langle \frac{dE}{d\mathbf{z}_l} \right\rangle \approx \langle f'(\mathbf{z}_l) \rangle \circ \mathbf{W}_l^T \left\langle \frac{dE} {d\mathbf{y}_{l+1}} \right\rangle = 0$$

Note that this argument doesn't really hold up in the last layer; we required a Batch Norm in layer $l+1$ to center incoming gradients to layer $l$. So if there are only $L$ layers in the network, gradient signals to $\mathbf{z}_L$ might not be centered. But we won't get too worked up if this doesnt hold for just one layer.

Combining the last two approximations, we are left with the greatly simplified backwards pass equation:

$$ \frac{dE}{d\mathbf{y}_l} \approx \frac{1}{\boldsymbol{\sigma}_l} \circ \frac{dE}{d\mathbf{z}_l} $$

We haven't even averaged anything over weights yet. This is roughly true for most networks in our ensemble.

#### Approximation 3: $\sigma^2 \approx \frac{\pi-1}{\pi}$
We're in the home stretch. Squaring then averaging the final equation over weights and then combining it with the results from the matrix multiplication and ReLU modules gives us:

$$ \lllangle \left(\frac{dE}{dx_{l-1}}\right)^2 \rrrangle \approx \lllangle \frac{1}{\sigma_l^2} \rrrangle \lllangle \left(\frac{dE}{dx_{l}}\right)^2 \rrrangle$$

All that remains is to compute $\llangle (1/\sigma_l)^2 \rrangle$. We know it has to be about $1.2$ because in our experiments the widths of the histograms grew by this much at each layer.

In this section we'll actually analytically calculate this quantity. But for some intuition, we'll look at the distribution of $\mathbf{x},\mathbf{y}$ and $\mathbf{z}$ over the minibatch samples in the 5th layer of our Batch Normalized network.

Notably, every element of $\mathbf{y}$ seems to have s

Our general strategy will be to use the fact that $\mathbf{y}_{l+1} = \mathbf{W}_{l+1} \mathbf{x}_{l}$ to compute the minibatch variance of $y$ given the minibatch variance of $\mathbf{x}_l$. We'll then argue that $z$ is a standard Gaussian and use the relation $x_l = [z_l]^+$ to compute the variance of $x$. In doing so, we will rely on two physics-style approximations.

Note that for any particular setting of weights, we can compute the variance of $y$ if we know the covariance matrix of $\mathbf{x}$
$$ \langle y^2 \rangle - \langle y \rangle^2 = \mathbf{w}^{\top} \left[ \langle \mathbf{x} \mathbf{x}^{\top} \rangle - \langle \mathbf{x} \rangle \langle \mathbf{x} \rangle^{\top} \right] \mathbf{w} = \mathbf{w}^{\top} \mathbf{C}_{xx} \mathbf{w} $$
Here is where our first approximation comes in: we are going to assume that the minibatch variance of $y$ is *self-averaging*. This means we will assume that for nearly every configuration of weights, $y$ will have roughly the same minibatch variance:
$$ \langle y^2 \rangle - \langle y \rangle^2 \approx \llangle \langle y^2 \rangle - \langle y \rangle^2 \rrangle $$
This is helpful because, averaged over weight configurations, most terms in the sum $\sum_{ij} C^{xx}_{ij} w_i w_j$ are 0. In particular when $i\neq j$ we have $\llangle w_i w_j \rrangle = 0$ and when $i=j$ we have $\llangle w_i w_j \rrangle = \sqrt{2}/N$. Since we have assumed the $x$ are identically distributed, we are left with the result:

$$ \langle y^2 \rangle - \langle y \rangle^2 = 2 [\langle x^2 \rangle - \langle x \rangle^2] $$

Now we have to compute the variance of $x$.

In particular we will apply a *mean-field* calculation: this means we will assume that, that over the minibatch distribution, elements of $x$ are independent random variables.

The first thing this does for us is that it tells us $\sigma$, the minibatch variance of $y$, is a *self-averaging* quantity: this means we assume that every $y$ has the same minibatch variance. (Actually it follows from the mean-field assumption, but im not sure how to Intuitively justify it without getting bogged down in eigenvalues of random matrix products. So let's just take it as another independent assumption.)

Since $y$ is a sum of $N$ independent random variables, it will have a gaussian distribution, at least if your network is wide enough. And since $z$ is just a centered and scaled version of $y$, it too will be Gaussian, with 0 mean and unit variance.

We now have all that we need to calculate the variance of $x$, which is simply a rectified version of $z$. In particular it is:
<!-- $$ \langle x^2 \rangle - \langle x \rangle ^2 = \frac{1}{2\pi} \int_{-\infty}^{+\infty} f(z)^2 e^{-z^2/2} dz - \left[\frac{1}{2\pi} \int_{-\infty}^{+\infty} f(z) e^{-z^2/2} \right]^2 dz $$ -->

$$ \langle x^2 \rangle - \langle x \rangle^2 = \int_{-\infty}^{+\infty} f(z)^2 \mathcal{D}z - \left[\int_{-\infty}^{+\infty} f(z) \mathcal{D}z \right]^2 $$
This notation means that each integral is an expectation over a unit Gaussian distribution.

This is great because we can easily calculate both of these terms. The first term is just $\frac{1}{2}$ (half the variance of a unit Gaussian, which by definition has variance 1). The 2nd term is $\frac{1}{2\pi}$. This follows from directly integration using the fact that $\int z e^{-z^2/2} dz = e^{-z^2/2}$. So the minibatch varaince of $x$ is just going to be $\frac{\pi}{2(\pi-1)}$. Under our mean-field assumption, we have the result that *every* $x$ has the same minibatch variance.

How does all of this relate to $\sigma$, the minibatch variance of $y$? Here is where we use our *self-averaging* approximation. We assume that not just every $x$ has the same variance, but every $y$ has the same variance. The intuitive justification is that since $y$ is the sum of $N$ random variables, each of which s

Because $y = Wx$. I think it will be easiest to swap to index notation

 We can compute easy compute the expected variance:
$$ \llangle \langle y^2_i \rangle - \langle y_i \rangle^2 \rrangle = \$$

We'll consider just a single y, and recall that every $x$ has the same distribtuion.

$$ \sigma^2 = \langle y^2 \rangle - \langle y \rangle^2 = 2 (\langle x^2 \rangle - \langle x \rangle^2) = \frac{\pi}{\pi-1}$$




#### Result: $\llangle \frac{dE}{dy} \rrangle \approx \frac{\pi}{\pi-1} \llangle \frac{dE}{dz} \rrangle$

Gradients grow in earlier layers. Of course, we can measure this.

## Normalized Preactivations Imply Very Nonlinear Input-Output Mappings
To get some intuition, we'll actually look at the forward pass of a vanilla and Batch Normalized net.

go back to the vanilla net. We'll look at the distribtuion of preactivations in later layers. Interestingly each one gets a mean value which grows with depth. Normalization completely changes this pciture.

What does this have to do with gradients? Well by normalizing, you're basically requiring the nonlinearity actually gets used. Normalizing every preactivation ensures you are now implementing some crazy nonlinear function.

<!-- ## Typical Gradients in a Vanilla Net
We want to see how the "typical scale" of gradients, roughly the width of each layer's gradient histogram, is changed by inserting Batch Norm into a randomly initialized deep network. In the simulation we saw that for vanilla nets this width is the same in each layer. This is expected. The original Kaiming paper argued this should be true.

 We'll show this first, this calculation will be used for the Batch Norm calculation anyways.

These results have already been shown

More precisely, we will be interested in how:

$$ \lllangle \left|\frac{dE}{ d\mathbf{x}_l}\right|^2 \rrrangle $$

<!-- $$ \left \llangle \left|\frac{dE}{ d\mathbf{x}_l}\right|^2 \right \rrangle $$ -->
<!--
propagates through layers of network.

Our basic strategy will be to first derive the backwards pass for each of the 3 modules in each layer of the Batch Norm nets. We will then average these over weights. -->

<!--
 Our general strategy will be:
1. Define the forward pass
2. Derive the backward pass
3. Average the backward pass over weights

We will make a number of simplifying assumptions for our calculations: we examine fully connected network with rectifying nonlinearity, initialized with Kaiming initialization. We also assume the inputs are Gaussian white noise. -->

<!-- Even after a lot of simplification, the backwards pass equations shown in the previous section are pretty ugly. They don't give a very intuitive feel for how gradients are modified by the Batch Norm layer. Also they depend on the precise configuration of $W$ which are random variables. So to gain some insight, we will compute the average over weights.

Since the W are independent random variables in each layer, we would ideally get some set of equations that look like this:

$$ \tilde{z} = dE / dx^l$$

Unfortunately, the gradients are functions of the activations so they are not independent. However, we will essentially cheat and assume these are independent variables.

Each layer consists of 3 modules. We'll work our way through each one

#### Matix Multipy
I'll assume you're mostly familiar. If you want a tutorial on how to get these, check out these other blogs!

#### ReLU Activation

#### Gradients are Preserved -->


<!-- #### Approximation 1: $\langle \frac{dE}{d\mathbf{z}} \rangle \approx 0$
We argue that gradients of preactivations should be nearly zero mean over a minibatch: $\langle \frac{dE}{d\mathbf{z}} \rangle \approx 0$. By averaging the backward pass equation for $y$ over a minibatch, it's actually easy to show that gradients w.r.t. $y$ are exactly zero:
$$ \langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \cancelto{0}{\langle \frac{dE}{d\mathbf{z}} - \langle} \frac{dE}{d\mathbf{z}} \rangle \rangle - \frac{1}{\sigma} \cancelto{0}{\langle \mathbf{z} \rangle} \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$

To derive $dE/dz$ from $dE/dy$,


#### Approximation 2: $\langle z \frac{dE}{d\mathbf{z}} \rangle \approx 0$
#### Approximation 3: $\llangle \sigma^2 \rrangle \approx \frac{\pi-1}{\pi}$
#### Approximation 4: $\llangle \frac{1}{\sigma^2} \rrangle \approx \frac{1}{\llangle \sigma^2\rrangle}$ -->



## Does this matter for Training?
