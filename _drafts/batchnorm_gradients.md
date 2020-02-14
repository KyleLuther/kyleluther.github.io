---
layout: post
title:  "Batch Norm Causes Exploding Gradients"
date:   2020-01-23 23:01:52 -0500
---

<div style="display:none">
$$
  \def\llangle{\langle \! \langle}
  \def\rrangle{\rangle \! \rangle}
  \def\lllangle{\left \langle \!\!\! \left \langle}
  \def\rrrangle{\right \rangle \!\!\! \right \rangle}
$$
$$ \require{cancel} $$
</div>

Practitioners know that inserting [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf) into your deep network is an almost fool-proof way to make the training process easier. And theorists know that *exploding gradients*, i.e. gradient signals which grow or decay exponentially with depth, generally makes the training process harder.

So it may be surprising to hear, but in [recent work](https://openreview.net/pdf?id=SyMDXnCcF7), Yang et. al showed that our beloved Batch Norm can actually *cause* exploding gradients, at least at initialization time. Before you go read the paper, I have to warn you: with the appendix, it is 95 pages of technical mathematics filled with Batch Symmetry Breaking Points, Gegenbauer expansions, and Diagonal-Off-Diagonal Semidirect operators. If this interests you, I would recommend giving it a read, as I think they get the details right and say something meaningful about Batch Norm.

However, if you'd be happy with a physics-style "back-of-the envelope" type calculation, stick around. In this post, we'll provide a shorter, hopefully more intuitive derivation for the gradient explosion phenomenon. We'll even get the same quantitative result, at least in the large batch regime. (Admittedly we are going to use a big envelope).

**TL;DR** Inserting Batch Norm into a network means that in the forward pass each neuron is divided by its standard deviation, $\sigma$, computed over a minibatch of samples. In the backward pass, gradients get divided by the same $\sigma$. At initialization time, we can actually compute $\sigma$ in wide networks with sufficiently large batch sizes. In ReLU nets, $\sigma\approx \sqrt{(\pi-1)/ \pi} \approx 0.82$.  Since this occurs at every layer, gradient norms in early layers are roughly $(1/0.82)^L$ times larger than gradient norms in layer $L$.

## Numerical Simulation (What is meant by "exploding gradients"?)
Before our calculation, let's empirically show what we mean by the statement "Batch Norm causes exploding gradients". We'll just initialize a net without Batchnorm, compute gradients, then put Batch Norm into the net and recompute gardients and see how they differ.

Our net without Batch Norm, our "vanilla net", is shown below. Its a boring old 10 layer, feedforward, fully connected network of uniform layer width $N=1024$. We're using the ReLU nonlinearity so we'll initialize weights with [Kaiming initialization](https://arxiv.org/pdf/1502.01852.pdf). This means we'll set all the biases to zero and sample every element of $W$ from a zero mean Gaussian with variance $\sqrt{2/N}$.

<p align="center">
  <img src="/assets/vanilla_net.png">
</p>

Our inputs $\{ \mathbf{x}_0 (t): t=1,2,...,512 \}$ are 512 random Gaussian vectors. We'll compute a linear loss over the network's outputs: $E = \sum_{t=1}^{512} \mathbf{w} \cdot \mathbf{x}_{10}(t)$. We choose $\mathbf{w}$ by drawing it from a unit Gaussian. Now we have everything we need to compute the gradients $dE/d\mathbf{x}_l(t)$ at each layer $l$ in the vanilla network (We'll use Pytorch to automate this process for us).

<!-- Our inputs to the net will be 512 Gaussian white noise vectors $\{ \mathbf{x}_0(t): t = 1,2,...,512 \}$ . We'll compute a linear loss over the network's outputs: $E = \sum_{t=1}^{512} \mathbf{w} \cdot \mathbf{x}_{10}(t)$. We choose $\mathbf{w}$ by drawing it from a unit Gaussian. Now we have everything we need to compute the gradients $\frac{dE}{d\mathbf{x}_l}$ at each layer $l$ in the vanilla network (We'll use Pytorch to automate this process for us). -->

Now we take that exact network, loss, and set of input vectors, but insert Batch Normalization before the ReLU in every layer:

<p align="center">
  <img src="/assets/bn_net.png">
</p>

Again we can use PyTorch to compute gradients $dE/d\mathbf{x}_l(t)$ at each layer. Now we'll visualize the gradient histograms in each layer for the vanilla and Batch Norm net. Each histogram consists of gradients for 512 inputs $\times$ 1024 features:

<p align="center">
  <img src="/assets/bn_gradient_simulation.png">
</p>

In the vanilla network, the gradient histograms are basically the same in every layer. However, after inserting Batch Norm, the gradients actually grow with decreasing layer. In fact, the widths of these histograms grow *exponentially*. It is in this sense that gradients explode.

## Gradients in a Batch Normalized Net
In this section we'll review the details of Batch Normalization and how it modifies the forward and backward pass of a neural network.

#### Forward Pass
<p align="center">
  <img src="/assets/module_forward.png">
</p>

Each layer in our normalized network contains 3 modules: Matrix Multiply, Batch Norm, and ReLU Nonlinearity. These are shown in the diagram above.

$\mathbf{x}_l, \mathbf{y}_l$ and  $\mathbf{z}_l$ denote the vector outputs of the matrix multiply, Batch Norm, and ReLU modules in layer $l$ for a single input. The [element-wise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) is denoted by $\mathbf{a} \circ \mathbf{b}$. We'll abuse notation and denote element-wise division like this $\mathbf{a}/\mathbf{b}$. The notation $\langle \cdot \rangle$ indicates a *minibatch average* so $\boldsymbol{\mu}_l$ and $\boldsymbol{\sigma}^2_l$ are the mean and variance of each element of $\mathbf{y}_l$ over a minibatch. $f$ is the ReLU nonlinearity which acts element-wise on an input vector.

Because we are only going to examine networks at initializion time, we have made two simplifications. One, we ignore bias addition in the matrix multiply layer as biases are initialized to zero. Two, we can ignore the part of Batch Norm with learnable parameters which transforms $\mathbf{z}_l \leftarrow \boldsymbol{\gamma}_l \circ \mathbf{z}_l + \boldsymbol{\beta}_l$ because $\boldsymbol{\gamma}_l=\boldsymbol{1}$ and $\boldsymbol{\beta}_l=\boldsymbol{0}$ at initialization.

#### Backward Pass
<p align="center">
  <img src="/assets/module_backward.png">
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

To do so we'll extend the gradient independence assumption to apply over the minibatch distribution. This means that for a typical configuration of weights, we assume forward pass quantities are independent from backward pass quantities over the minibatch distribution. With this assumption we have

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

All that remains is to compute $\llangle (1/\sigma_l)^2 \rrangle$, the expected inverse variance of an element of $\mathbf{y}_l$. While we already know that $\llangle (1/\sigma_l)^2 \rrangle \approx 1.2$ from our simulations (this is the factor by which the width of gradient histograms grew every layer), let's calculate it analytically.

**Step 1**: Let's start with something we do know the variance of. Because $z$ is the output of a Batch Norm module, it has to be zero-mean and unit variance over the minibatch. Since $x$ is just a rectified version of $z$, it might seem like we can relate the variance of $x$ to $z$.

Unfortunately knowing the mean and variance of $z$ is not enough to evaluate either of these integrals. It seems like a reasonable assumption however that $z$ will take a Gaussian distribution. Because $z$ is just a centered and scaled version of $y$, and $y$ is the sum of $N$ random variables. In particular we just have to integrate the unit Gaussain over the positive half:
$$ \langle x^2 \rangle - \langle x \rangle^2 = \frac{1}{\sqrt{2\pi}} \int_{0}^{+\infty} z^2 e^{\frac{-z^2}{2}} dz - \left[\frac{1}{\sqrt{2\pi}} \int_{0}^{+\infty} z e^{\frac{-z^2}{2}} dz\right]^2 = \frac{1}{2} - \frac{1}{2\pi} $$

The first term is just $\frac{1}{2}$ (half the variance of a unit Gaussian). The 2nd term is $\frac{1}{2\pi}$ which follows from direct integration: $\int z e^{-z^2/2} dz = e^{-z^2/2}$.


**Step 2:** Now we can relate the expected variance of an element of $y$ to the variance of $x$ (recall that the minibatch variance is computed over samples in the minibatch, and in general it depends on the network weights):

$$ \sigma^2 = \langle y^2 \rangle - \langle y \rangle^2 = \mathbf{w}^{\top} \left[ \langle \mathbf{x} \mathbf{x}^{\top} \rangle - \langle \mathbf{x} \rangle \langle \mathbf{x} \rangle^{\top} \right] \
\mathbf{w} = \mathbf{w}^{\top} \mathbf{C}^{xx} \mathbf{w} $$

It is simple enough to compute the average minibatch variance $\llangle \sigma^2 \rrangle$. When $i\neq j$ we have $\llangle w_i w_j \rrangle = 0$ and when $i=j$ we have $\llangle w_i w_j \rrangle = \sqrt{2}/N$. Since we have assumed the $x$ are identically distributed, we are left with the result:

$$ \llangle \sigma^2 \rrangle = 2 \llangle \langle x^2 \rangle - \langle x \rangle^2 \rrangle $$

**Step 3:** So far we've computed $\llangle \sigma^2 \rrangle$, but this has no obvious relation to $\llangle 1/\sigma^2 \rrangle$, which is the quantity we wanted. We're going to argue that $\sigma^2$, the minibatch variance of an element of $\mathbf{y}$, is a *self-averaging* quantity. This means that for nearly every configuration of weights, $\sigma^2$ is roughly the same:

$$ \langle y^2 \rangle - \langle y \rangle^2 \approx \llangle \langle y^2 \rangle - \langle y \rangle^2 \rrangle $$

This one is actually a little tricky. If you want to skip this part, thats ok.

 One can directly compute the variance (over weights) of the minibatch variance: $\llangle (\sigma^2)^2 \rrangle - \llangle \sigma^2 \rrangle^2$. And as you might imagine this gets pretty ugly. But the result is roughly:

 $$ \llangle (\sigma^2)^2 \rrangle - \llangle \sigma^2 \rrangle^2 = \sum_i \lambda^2 $$


and you'll see that it looks like. Random activations enough. Really need large batch size. (this was identified there too)


We can show this assumptions holds in our simulation, we can visualize the distribution of the first 3 elements of $\mathbf{y}$ in the 5th layer of our Batch Normalized net.

<p align="center">
  <img src="/assets/y-distribution.png">
</p>

Interestingly each element of $\mathbf{y}$ seems to have a different mean, but we can see that fluctuations around the mean appear to have identical distributions. Importantly their variance is all the same.


####Result: $\llangle \frac{dE}{dy} \rrangle \approx \frac{\pi}{\pi-1} \llangle \frac{dE}{dz} \rrangle$

## Discussion
To recap the main results:
1. Without Batch Norm gradients should be constant.
2. With Batch Norm, gradients should grow by $\sqrt{\pi/(\pi-1)}$ at each layer.
3. We also predicted that, except in the final layer, it shouldnt matter whether or not you backpropagate through the $\mu$ and $\sigma$ at initialization. So we can test this too.


#### Normalized Preactivations Imply Very Nonlinear Input-Output Mappings
Before concluding, we'll give one more way to think about the exploding gradient phenomenon: by normalizing the preactivations (the inputs to the ReLU), you are ensuring that the nonlinearity is used "more effectively" at each layer.

In this regime, the network is more likely to implement some crazy nonlinear function which massively distorts the geometry of the input space. In more formal terms, normalization the inputs means. I think this argument makes more sense if we just compare the preactivations ($y$ for vanilla net and $z$ for Batch Norm net) in our simulations.

<p align="center">
  <img src="/assets/forward_pass.png">
</p>

At this point, shameless plug. If you want to understand . I'll probably write an easier to read blog post.


#### Does this matter for Training?
I'm really not sure. It definitely makes you wonder if maybe exploding gradients aren't really as bad as they seem. The authors do show that in extremely deep networks Batch Norm made the nets untrainable.

<!-- When you consider the actual Batch Norm operation, it might seem even more surprising that it can change the backwards pass. It just centers and scales each neuron so its zero mean and unit variance. If you initialized the network reasonably, shouldn't each neuron already be pretty close to unit mean and variance anyway, how could this seemingly trivial renormalization have such a dramatic impact on the gradient backpropagation?



In this post, we are going to provide an alternate derivation for the exploding gradient observation. Now, if you would be satisfied with nothing less than a perfectly rigorous, detailed, and lengthy mathematical derivation of this phenomenon you should check out their [paper](https://openreview.net/pdf?id=SyMDXnCcF7). If you're like me and just want a more intuitive explanation (but still gives quantitative predictions), stick around for the rest of this post!

(Ok I kid, I would still recommend the paper to those interested, but it is quite technical and 95 pages long with the appendix)



If you train deep networks, you probably know that Batch Norm usually improves network training. And everyone knows exploding gradients are bad for training. Most people probably don't know that Batch Norm causes exploding gradients.

Almost everyone who trains deep nets seems to love [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf). Understandably so, as it regularly reduces training times, sensitivity to parameter initialization, and sensitivity to learning rate selection. Basically it helps "fool-proof" deep network training. And as an added benefit, it often improves final test-time performance.

However, as with many aspects of deep learning, a precise reason *why* Batch Norm confers these benefits remains elusive. [Recent work](https://openreview.net/pdf?id=SyMDXnCcF7) by Yang et. al. has taken one step forwards by examining the behavior of Batch Norm in random networks, i.e. at initialization time. They show that inserting Batch Norm into a network can actually cause gradients to explode exponentially with depth, at least at initialization time. This result may seem pretty surprising; having any quantity explode or decay exponentially with depth would only seem to make training harder, not easier.

In this post, we are going to provide an alternate derivation for the exploding gradient observation. Now, if you would be satisfied with nothing less than a perfectly rigorous, detailed, and lengthy mathematical derivation of this phenomenon you should check out their [paper](https://openreview.net/pdf?id=SyMDXnCcF7). If you're like me and just want a more intuitive explanation (but still gives quantitative predictions), stick around for the rest of this post!

(Ok I kid, I would still recommend the paper to those interested, but it is quite technical and 95 pages long with the appendix) -->
