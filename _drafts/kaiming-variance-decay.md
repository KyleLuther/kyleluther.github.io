---
layout: post

title: "Sample vs. Weight Randomness in Neural Networks"
date: 2020-02-18 10:00:00
keywords: Kaiming initialization, preserve variance
---
<!-- title: "What variance does Kaiming initialization preserve?" -->
$$
  \def\llangle{\langle \! \langle}
  \def\rrangle{\rangle \! \rangle}
  \def\lllangle{\left \langle \!\! \left \langle}
  \def\rrrangle{\right \rangle \!\! \right \rangle}
$$

<p align="center">
  <img src="/assets/samplelayer_randomness.png">
</p>*Preactivation distributions due to weight randomness (solid histograms) and sample randomness (red histogram) in random network initialized with Kaiming initialization*

*Initialize your network to preserve the mean and variance of neurons across layers.* This prescription is pervasive in the deep learning community and its easy to see why. Not only is it effective, but it is simple to implement, and simple to understand.

For nets with ReLU nonlinearity, we implement this prescription by drawing weights from a zero mean distribution with variance $2/N$ where $N$ is the fan in to a neuron in the next layer. This is known as Kaiming initialization. Normalization schemes like Batch/Layer/Group/etc. Norm make variance preservation even easier to implement; by design they normalize neurons so they are zero mean and unit variance at init time.

But there is a subtlely here, over what distribution are the mean and variance computed? Kaiming preserves the variance when computed over randomness in *weights*. Batch Norm normalizes neurons using *minibatch* statistics. Layer Norm uses the mean and variance over a *layer*.

In this post, we'll show that neuron distributions can be very different when computed over the sample distribution vs. over weights. With mild assumptions on the input distribution, we'll show that Kaiming initialization in ReLU nets preserves the variance of neurons over weights but causes the variance over samples to decay to zero with depth! [^1]

[1]: We wrote a paper describing this phenomenon. This post is hopefully a simpler derivation of the same phenomenon.


## Network vs. Sample vs. Layer Randomness
To qualitatively show that neuron distributions can be quite different depending on if you average over weights or samples,
let's start off with an experiment. We will examine neuron distributions in an ensemble of 50 layer fully connected ReLU networks of uniform width $N=1024$. We'll randomly set the nets' parameters using Kaiming initialization, so biases are zero and weights are drawn iid from a zero mean Gaussian with variance $2/N$. To keep things as simple as possible we'll assume inputs are Gaussian white noise.

<p align="center">
  <img src="/assets/kaiming-net.png">
</p>

In this setup, we can identify two fundamental sources of randomness, one from the weights:

$$ \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \stackrel{iid}\sim \mathcal{N}(0, 2/N) $$

and the other from the inputs:
$$ \mathbf{x}_0 \sim Q(\mathbf{x}_0)$$

Describe preactivations and network mroe precisely

#### Network Randomness
We will call randomness due to fluctuations in every layer's weights *network randomness*. In this section we will visualize a preactivation's distribution due to network randomness, i.e. we'll visualize:

$$ P(y_l | \mathbf{x}_0 ) \qquad \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$

Note that this distribution is a function of the input $\mathbf{x}_0$. To empirically compute this quantity, we pick a single input $\mathbf{x}_0$ and randomly sample 1000 different networks. Below we show the empirical distribution of a single neuron in layers 1,5,10,50.

<p align="center">
  <img src="/assets/network_randomness.png">
</p>

Ok, this is boring. Just a bunch of unit Gaussians at each layer. But it's good that this is boring; Kaiming initialization is supposed to preserve the variance of each of these distributions.

#### Sample Randomness
This one is more interesting. Here we will freeze weights in all layers and visualize a preactivation's distribution due to *sample randomness*. Mathematically, we will visualize:

$$ P(y_l | \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_{l} ) \qquad \mathbf{x}_0 \sim Q(\mathbf{x}_0) $$

To do so, we generate a single network, and this time generate 1000 different input samples. We'll compute the empirical distribution over samples of a neurons in layers 1,5,10,50. To show that these distributions depend on the weights in a non-trivial way, we'll repeat this process 3 times to get 3 different distributions. In general each of these distributions depends on the precise configuration of weights in the network.

<p align="center">
  <img src="/assets/sample_randomness.png">
</p>

Completely different from the network randomness setting! Interestingly it appears that as we look deeper and deeper in the network, the distribution of each preactivation collapses to small fluctuations around some large mean value. The network appears to be implementing a nearly constant function. The location of these mean values depend on the network weights.

#### Layer Randomness
In practice, one is not very likely to initialize 1000 different networks and compute preactivation distributions. Instead one would probably initialize a single network and compute the distribution of all preactivations in a layer for a single sample.

This is equivalent to visualizing a single preactivation's distribution due to *layer randomness*, randomness in a single layer's weights, with all other weights fixed. To visualize this, we'll fix the weights in layers $1,2,...,l-1$ and then visualize the distribution of a single preactivation $y_{l,i}$ over random choices in its weight vector $\mathbf{w}_{l,i}$.

$$ P(y_l | \mathbf{x}_0, \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_{l-1} ) \qquad \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$

Here are the results for a single network and input.

<p align="center">
  <img src="/assets/layer_randomness.png">
</p>

These look similar to the distributions due to network randomness. So long as the network is sufficiently *wide*, these distributions will be close to the distributions in the previous section that we got by allowing weight in every layer to fluctuate.

In the rest of this post, we will ignore any subtle differences between network and layer randomness and just focus on the difference between network and sample randomness.

<!-- ## Kaiming init preserves variance over networks
In this section we'll show that Kaiming initialization preserves the variance of units if you randomize over weights. This is just a review of the argument given in the original Kaiming initialization paper.

We will define the *network variance* of a preactivation as:

$$ \llangle y^2 \rrangle - \llangle y \rrangle^2 $$

where double brackets $\llangle \cdot \rrangle$ indicate an average over weights in all layers.

Two things to note. One, $y$ is zero mean over weights, $\llangle y \rrangle=0$, because elements of $W$ are zero mean. Two, every $y$ in the same layer has the same distribution, if you randomize over weights, because elements of $W$ are identically distributed.

Recall the forward pass of our fully connected net:  $\mathbf{y}\_{l} = \mathbf{W}\_{l} f(\mathbf{y}\_{l-1})$ . Because elements of $\mathbf{W}\_l$ and $\mathbf{y}_{l-1}$ have identical and independent distributions over weights, we can write the variance of an element of $\mathbf{y}_l$ as:

$$ \llangle y_l^2 \rrangle = N \; \llangle W_l^2 \rrangle \llangle f(y_{l-1})^2 \rrangle = 2 \llangle f(y_{l-1})^2 \rrangle$$

Over weights, elements of $\mathbf{y}_{l-1}$ are symmetrically distributed around zero:
$$ \llangle f(y_{l-1})^2 \rrangle = \frac{1}{2} \llangle y_{l-1}^2 \rrangle $$

The result:
$$ \llangle y_l^2 \rrangle =  \llangle y_{l-1}^2 \rrangle $$
The variance over networks of a preactivation in layer $l$ is the same as it is in layer $l-1$. Variance is preserved.

Some general things to note, averaging over weights greatly simplifies this calculation. Also this calculation is exact. We didnt have to make any assumptions about the input distribution or the network width.

## Kaiming init shrinks variance over samples
We want to do the opposite of what we did in the last section. We want to freeze the network weights and compute the variance over samples of a preactivation in each layer:

$$ \langle y^2 \rangle - \langle y \rangle^2 $$

where single brackets $\langle \cdot \rangle$ indicate an average over samples. Unfortunately, averaging over inputs with fixed weights is much more challenging than avering over weights with fixed inputs.

So rather than calculate the sample variance of a preactivation, which depends on the precise configuration of weights, we'll instead calculate the *network averaged* sample variance:

$$ \llangle \langle y^2 \rangle - \langle y \rangle^2 \rrangle $$

We'll later argue that the sample variance is a *self-averaging* quantity. Basically, it means that so long as our networks are wide enough, the sample variance of a preactivation in any particular weight configuration will basically be equal to its average over weight configurations:

$$ \langle y^2 \rangle - \langle y \rangle^2 \approx \llangle \langle y^2 \rangle - \langle y \rangle^2 \rrangle $$

For now, let's just take this as a given. -->

<!-- ## Calculation Time!
The original Kaiming paper showed how to calculate the preactivation mean and variance due to network randomness. They show that the mean of every preactivation is will be zero, regardless of the variance of the weights, and the variance of each preactivation will be the as the variance in the previous layer exactly when the weights have variance $2 / \text{fan in}$

This is relatively simple because every weight is iid and the result doesn't really rely on any properties of the input data. Calculating the mean and variance of neurons due to sapmle randomness is going to be much harder.  But with some assumptions (basically wide network + "random enough" inputs) we can calculate the mean and variance of neurons due to -->

## Reformat so experiments and theory are intertwined

## Preactivation Statistics arising from Network Randomness
Describing the statistics of preactivations due to network randomness (randomness in the weights due to the initialization procedure) is much simpler. That is probably why essentially every paper you'll read on initialization only considers this scenario.

We'll show some basic properties. Interestingly *none* of these are true of statistics due to sample randomness.

**1. Within a layer, every preactivation has the same distribution**

To see this, use the relationship $\mathbf{y}_l = \mathbf{W}_l \mathbf{x}_l$ and recall that every element of $\mathbf{W}_l$ has the same distribution. This implies that every element of $\mathbf{y}$ is identically distributed, over weight configurations.

**2. Every preactivation is zero mean**
Again, we use the relationship $\mathbf{y}_l = \mathbf{W}_l \mathbf{x}_l$. Because $\mathbf{W}_l$ is independent of $\mathbf{x}_{l-1}$, we can write $\llangle \mathbf{y}_l \rrangle  = \llangle \mathbf{W}_l \rrangle \llangle \mathbf{x}_l \rrangle$. Because $\mathbf{W}$ is zero mean, we have $\llangle \mathbf{y}_l \rrangle=0$.

**3. For all layers, every preactivation has the same variance if Var(W)=2/N**

Because elements of $\mathbf{W}\_l$ and $\mathbf{y}_{l-1}$ have identical and independent distributions over weights, we can write the variance of an element of $\mathbf{y}_l$ as:

$$ \llangle y_l^2 \rrangle = N \; \llangle W_l^2 \rrangle \llangle f(y_{l-1})^2 \rrangle = 2 \llangle f(y_{l-1})^2 \rrangle$$

Over weights, elements of $\mathbf{y}_{l-1}$ are symmetrically distributed around zero:
$$ \llangle f(y_{l-1})^2 \rrangle = \frac{1}{2} \llangle y_{l-1}^2 \rrangle $$

The result:
$$ \llangle y_l^2 \rrangle =  \llangle y_{l-1}^2 \rrangle $$
The variance over networks of a preactivation in layer $l$ is the same as it is in layer $l-1$. Variance is preserved.

## Preactivation Statistics arising from sample randomness

None of previous observations seem to be true. When you fix the weights, each preactivation has a different distribution over inputs, in general they are not zero mean, and the variance seems to decrease in each layer.

#### Mean-Fluctuation Decomposition
We can qualitatively describe the phenomenon. Unfortunately rigorous treatment will not be possible.

Our first step will be to decompose every preactivation into the sum of two terms, its mean $\mu$ over samples and fluctuations $\nu$ around the mean:

$$ y(w,t) = \mu(w) + \nu(w,t) $$

Note that the $\mu$ only depends on the network weights, which we indicate by $\mu(w)$, and $\nu$ depends both on the input and weights, which we indicate by $\nu(w,t)$. Now we will make two approximations.

#### Approximate Calculation (Wide net + random inputs)

If we are willing to make some assumptions, we will actually be able to calculate the precise amount by which the variance of each preactivation shrinks with layer.

Our key insight will be that the variance of each neuron is the same.

$$ \langle y^2 \rangle - \langle y \rangle^2 $$

We'll have to be really carefu

**Property 1: Within a layer, the sample variance of every neuron is the same**


If you are convinced by the experiments that this is a reasonable approximation, I suggest moving on to the next section. The justification for this is a little hairy.

This is actually quite a subtle point; within a layer, neither the sample mean $\langle y \rangle$ nor the sample 2nd moment $\langle y^2 \rangle$ are the same for all preactivations. They depend strongly on the initial weight configuration. But their difference $\langle y^2 \rangle - \langle y \rangle^2$ is the same. What gives?

Here is where we will use our wide network + random enough assumptions. Formally, we wish to show that the variance of $y$ is a *self-averaging* quantity. This means that for nearly all networks in our ensemble, the sample variance of $y$ is roughly same as the average:
$$ \langle y_l^2 \rangle - \langle y_l \rangle^2 \approx \llangle \langle y_l^2 \rangle - \langle y_l \rangle^2 \rrangle $$

To show this theoretically, let's use our formula $y_i = \sum_j W_{ij} x_j$:
$$ \langle y_l^2 \rangle - \langle y_l \rangle^2 = \mathbf{w}^T C \mathbf{w} $$

To show that $\langle y_l^2 \rangle - \langle y_l \rangle^2$ is self-averaging, we want to show that  the variance over weights of the sample variance is small:  

$$ \llangle (v^2)^2 \rrangle - \llangle v^2 \rrangle^2 \approx 0 $$

Yikes. But let's use our formula $y_i = \sum_j W_{ij} x_j$ to write $\langle y_l^2 \rangle - \langle y_l \rangle^2 = \mathbf{w}^T C \mathbf{w}$

$$ \llangle v^2 \rrangle^2 = \left[\frac{1}{N} \sum_i \lambda_i \right]^2 \qquad \llangle v^4 \rrangle = \frac{1}{N} \sum_i \lambda_i^2 $$

where $\lambda_i$ are the eigenvalues of the correlation matrix

Here is where we cheat and can invoke a mean-field approximation. If we assume that all the $x_i$ are in fact independent, and we assume we have infinitely many samples, then the covariance matrix $C$ will have small second moment.

Ok, dang

**Property 2: Fluctuations are Gaussian:**
$$ \nu_l \sim \mathcal{N}(0, v_l^2) $$

This might seem. Product can be non-Gaussian, Mean field over weights. This means that yes, everything in next layer is gaussian.


**Property 2: The mean of each neuron is Gaussian random variable:**

$$ \text{network randomness: } \; \mu_l \sim \mathcal{N}(0, m_l^2) $$

**Property 3: The sum of m and v is constant with depth**
This is just kaiming init

**Property 3: The variance follows the iterated mapping:**

$$ K(c) = c \left[ 1 + \frac{1}{\pi} \left(\sqrt{1/c^2 - 1} - \cos^{-1}(c)) \right) \right] $$
where $c^2 = \frac{m^2}{m^2+v^2}$


**Step 1:** Without any approximations, we can relate the typical sample variance of a preactivation in layer $l$ to the typical sample variance of the activation.

$$ \llangle \langle y_{l+1}^2 \rangle - \langle y_{l+1} \rangle^2 \rrangle = (2/N) \left[ \llangle \langle f(y_{l})^2 \rangle \rrangle - \llangle \langle f(y_{l}) \rangle^2 \rrangle \right] $$

The real challenge will be in dealing with the nonlinearity. Unfortunately, its not going to be valid to swap the order of expectation.

**Step 2:**
Let's assume we know the sample mean of a preactivation. Then its easy to compute $\langle f(y_{l}) \rangle$. Its just the integral:

$$ \langle f(y_{l}) \rangle = \int f(\mu + \nu) P(\nu) d\nu \qquad P(\nu) = \mathcal{N}(0, v_l^2) $$

Now we can average this over weight configurations:

$$ \llangle \langle f(y_{l}) \rangle^2 \rrangle = \int \left[\int f(\mu + \nu) P(\nu) d\nu\right]^2 P(\mu) d\mu $$

**Step 3:**
Actually computing the 2nd moment is quite easy:
$$ \llangle \langle f(y_{l})^2 \rangle \rrangle = \int \int f(\mu + \nu)^2 P(\nu) P(\mu) d \nu d\mu $$
The distribution of $(\mu+\nu)$ is a zero mean gaussian.

This is very similar to the calculation done in previous papers. In fact, we could have seen used their result to derive this. The observation is that $\llangle \langle f(y_{l})^2 \rangle \rrangle = \langle \llangle f(y_{l})^2 \rrangle \rangle$. We could integrate over weights first, then average this over inputs.

**Step 4:**
Now we have concrete expressions for both terms. We can do some algebraic manipulation to rewrite this into.

This is just the arccosine kernel.

Figure

## Discussion
So apparently the statement "Kaiming initialization preserves variance" is actually pretty subtle. Its true if you compute the variance over weight randomness. But the variance over samples, arguably the more relevant quantity in practical deep learning, actually decays with depth!

#### Batch/Layer/Group Normalization
Typically, people think of normalization schemes as ways to "reduce covariate shift, smooth loss surface, etc" but we can use our theory to understand how popular normalization schemes actually alter the initialization of a network.

Standard normalization schemes normalize preactivations:
$$ y \leftarrow \frac{y - \mu}{ \sigma } $$

Batch Normalization divides every preactivation by its *batch mean* and *batch variance*. When your batch is large, these will be closely related to the sample mean, and sample variance. This implies that every preactivation will be zero mean and unit variance, a big difference from the unnormalized case!

Layer Normalization uses statistcs over computed layer, which we've argued are closely related to network mean and variance. So this might not have such a big effect, at least in the wide network case. Is this a possible source of the discrepancy in training between layer and batch norm?

Group Normalization is something like a mix between the Batch and Layer Normalization. It normalizes with the mean and variance over subsets of neurons in a layer.

#### Implications for Training
The real question is "does any of this matter for training?" My answer is definitely YES, IT CAN MATTER. In fact the whole reason I wrote this post was that I was experimenting with various initializations for training UNets and I observed that dividing by the mean and standard deviation of each preactivation, rather than over a layer, seemed to have a noticable speedup to training.

This originally confused me, as i incorrectly thought they would be the same.

<p align="center">
  <img src="/assets/learning_curves.png">
</p>

The question of does it always matter, when does it matter? That im less sure about. Check out David Page's post for some info.

#### Random ReLU networks map all inputs to the same output
On the more theoretical side, there have been a number of works that get to the same basic conclusion. Here is an alternate interpretation

#### Random ReLU networks are nearly linear
Its mostly linear

#### Other non-linearities
