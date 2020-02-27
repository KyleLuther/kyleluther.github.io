---
layout: post
title: "Kaiming initialization doesn't preserve neuron variance, if you compute the variance over inputs rather than weights"
date: 2020-02-18 10:00:00
keywords: Kaiming initialization, preserve variance
---

$$
  \def\llangle{\langle \! \langle}
  \def\rrangle{\rangle \! \rangle}
  \def\lllangle{\left \langle \!\! \left \langle}
  \def\rrrangle{\right \rangle \!\! \right \rangle}
$$

<p align="center">
  <img src="/assets/samplelayer_randomness.png">
</p>

*Initialize your network to preserve the mean and variance of neurons across layers.* This prescription is pervasive in the deep learning community and its easy to see why. Not only is it effective, but it is simple to implement, and simple to understand.

For ReLU nets, we implement this prescription by drawing weights from a zero mean distribution with variance $2/N$. This is the popular Kaiming initialization. Normalization schemes like Batch/Layer/Group/etc. Norm make this even easier to implement; by design they normalize neurons so they are zero mean and unit variance at init time.

But there is a subtlely here, over what distribution are the mean and variance computed? Kaiming preserves the variance when computed over *weights*. Batch Norm normalizes neurons using *minibatch* statistics. Layer Norm uses the mean and variance over a *layer*.

In this post, we'll show that neuron distributions are very different when computed over the sample distribution vs. over weights. With mild assumptions on the input distribution, we'll show that Kaiming initialization in ReLU nets preserves the variance of neurons over weights but causes the variance over samples to decay to zero with depth! [^1]

[1]: We wrote a paper describing this phenomenon. This post is hopefully a simpler derivation of the same phenomenon.


## Network vs. Sample vs. Layer Randomness
Let's start off with an experiment. We will examine neuron distributions in an ensemble of 50 layer fully connected ReLU networks of uniform width $N=1024$. We'll set their parameters using Kaiming initialization, so biases are zero and weights are drawn iid from a zero mean Gaussian with variance $2/N$. To keep things as simple as possible we'll assume inputs are Gaussian white noise.

<p align="center">
  <img src="/assets/kaiming-net.png">
</p>

In this setup, we can identify two fundamental sources of randomness, one from the weights and the other from the inputs:
$$ \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \stackrel{iid}\sim \mathcal{N}(0, 2/N) \qquad \mathbf{x}_0 \sim Q(\mathbf{x}_0)$$

The point of this post is that preactivations will have qualitatively different distributions depending on which variables we fix and which we randomize over. In this section we'll show this emperically, and in later sections we'll show it analytically.

#### Network Randomness
We will call randomness due to fluctuations in every layer's weights *network randomness*. In this section we will visualize a preactivation's distribution due to network randomness. Mathematically, this means we wish to visualize:
$$ P(y_l | \mathbf{x}_0 ) \qquad \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$
Note that this distribution is a function of the input $\mathbf{x}_0$. To empirically compute this quantity, we pick a single input $\mathbf{x}_0$ and randomly sample 1000 different networks. Below we show the empirical distribution of a single neuron in layers 1,5,10,50.

<p align="center">
  <img src="/assets/network_randomness.png">
</p>

Ok, this is boring. Just a bunch of unit Gaussians at each layer. But it's good that this is boring; Kaiming initialization is supposed to preserve the variance of each of these distributions. But there are two weird things about this computation.

#### Sample Randomness
This is where things get interesting. Here we will freeze weights in all layers and visualize a preactivation's distribution due to *sample randomness*. Mathematically, we will visualize:
$$ P(y_l | \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_{l} ) \qquad \mathbf{x}_0 \sim Q(\mathbf{x}_0) $$


To do so, we generate a single network, and this time generate 1000 different input samples. We'll compute the empirical distribution over samples of 3 neurons in layers 1,5,10,50. In general each of these distributions depends on the network.

<p align="center">
  <img src="/assets/sample_randomness.png">
</p>

Completely different! Interestingly it appears that as we look deeper and deeper in the network, the distribution of each preactivation collapses to small fluctuations around some large mean value. The network appears to be implementing a nearly constant function. The location of these mean values depend on the network weights.

#### Layer Randomness
In practice, one is not very likely to initialize 1000 different networks and compute preactivation distributions. Instead one is much more likely to initialize a single network and compute the distribution of all preactivations in a layer for a single sample.

This is equivalent to visualizing a single preactivation's distribution due to *layer randomness*, randomness in a single layer's weights. I.e. you fix the weights in layers $1,2,...,l-1$ and then visualize the distribution of a single preactivation $y_{l,i}$ over random choices in its weight vector $\mathbf{w}_{l,i}$.

$$ P(y_l | \mathbf{x}_0, \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_{l-1} ) \qquad \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$

In our experiment here are the results for a single network and input.

<p align="center">
  <img src="/assets/layer_randomness.png">
</p>

Again, pretty boring. So long as the network is sufficiently *wide*, these distributions will be close to the distributions in the previous section that we got by allowing weight in every layer to fluctuate.

We won't discuss layer randomness again, but its worth mentioning.

#### Mean-fluctuation decomposition
So it seems like Kaiming initialization exhibits some non-trivial behavior that can be seen when looking at preactivations in fixed networks. Motivated by these experiments we will decompose the value of each preactivation into the sum of two terms: its mean $\mu$ over samples and fluctuations $\nu$ around the mean $\nu$:

$$ y(w,t) = \mu(w) + \nu(w,t) $$

Importantly the mean $\mu$ is a function only of the network weights, which we have written as $w$. The fluctuations. The variance of $y$

Kaiming initialization preserves the variance of $y$ over randomenss in $w$. But the variance of $y$ has two contributions if you average over weights, fluctuations in $\mu$ and fluctuations in $\nu$. In higher layers it appears that nearly all the fluctuations in $y$ are coming from fluctuations in $\mu$.

In the rest of this post, we'll try to instead calculate fluctuations in $y$ due only to fluctuations in the input $t$.

## Kaiming init preserves variance over networks
In this section we'll show that Kaiming initialization preserves the variance of units if you randomize over weights. This is just a review of the argument given in the original Kaiminig initialization paper.

We will define the *network variance* of a preactivation as:
$$ \llangle y^2 \rrangle - \llangle y \rrangle^2 $$
<!--
$$ \llangle y^2 \rrangle = \int y_l^2 P(y_l | \mathbf{x}_0) dy_l \qquad \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$

$$ \llangle y^2 \rrangle - \llangle y \rrangle^2 = \int y_l^2 P(y_l | \mathbf{x}_0) dy_l - \left[\int y_l P(y_l | \mathbf{x}_0) dy_l\right]^2 $$ -->
where double brackets $\llangle \cdot \rrangle$ indicate an average over weights in all layers.

Two things to note. One, $y$ is zero mean over weights, $\llangle y \rrangle=0$, because elements of $W$ are zero mean. Two, every $y$ in the same layer has the same distribution, if you randomize over weights, because elements of $W$ are identically distributed.


Recall the forward pass of our fully connected net: $\mathbf{y}_l = \mathbf{W}_{l} f(\mathbf{y}_{l-1})$. Because fact elements of $\mathbf{W}$ and $\mathbf{y}_{l-1}$ have identical and independent distributions over weights, we can write the variance of an element of $\mathbf{y}_l$ as:

$$ \llangle y_l^2 \rrangle = N \; \llangle W_l^2 \rrangle \llangle f(y_{l-1})^2 \rrangle = 2 \llangle f(y_{l-1})^2 \rrangle$$

Over weights, elements of $\mathbf{y}_{l-1}$ are symmetrically distributed around zero:
$$ \llangle f(y_{l-1})^2 \rrangle = \frac{1}{2} \llangle y_{l-1}^2 \rrangle $$

The result:
$$ \llangle y_l^2 \rrangle =  \llangle y_{l-1}^2 \rrangle $$
The variance over networks of a preactivation in layer $l$ is the same as it is in layer $l-1$. Variance is preserved.

Some general things to note, averaging over weights greatly simplifies this calculation. We didnt make any assumptions about the input, nor the network width. This calculation is exact.

## Kaiming init shrinks variance over samples
In some sense we are going to do the opposite of what we did in the previous section. Here, we want to freeze the network weights and compute the variance over samples of a preactivation in each layer:
$$ \langle y_l^2 \rangle - \langle y_l \rangle^2 $$
where single brackets $\langle \cdot \rangle$ indicate an average over samples.

Unfortunately, averaging over inputs with fixed weights is much more challenging than avering over weights with fixed inputs. We will need to make

#### Intuition

#### Calculation

**Step 1:**
$$ \langle y_l^2 \rangle - \langle y_l \rangle^2 \approx \llangle \langle y_l^2 \rangle - \langle y_l \rangle^2 \rrangle $$

**Step 2:**
$$ y = \mu + \nu $$
$$ \text{sample randomness: } \; \nu \sim \mathcal{N}(0, v^2) $$
$$ \text{network randomness: } \; \mu \sim \mathcal{N}(0, m^2) $$

**Step 3:**

$$ \llangle \langle y \rangle^2 \rrangle = 2 \int \left[\int f(\mu+\nu) P(\nu) d\nu \right]^2 P(\mu) d \mu $$

**Step 4:**
$$ \int \left[\int f(\mu+\nu) P(\nu| \mu) d\nu \right]^2 P(\mu) d \mu = K(m,v)$$


## Discussion
#### Batch Normalization and Layer Normalization
#### Relationship to other works
#### Nearly Linear Networks and Edge of Chaos
#### Implications for Training
