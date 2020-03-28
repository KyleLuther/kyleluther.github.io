---
layout: post

title: "Network vs. Layer vs. Sample Randomness in Neural Networks"
date: 2020-02-18 10:00:00
keywords: Kaiming initialization, preserve variance
---
<div style="display:none">
$$
  \def\llangle{\langle \! \langle}
  \def\rrangle{\rangle \! \rangle}
  \def\lllangle{\left \langle \!\! \left \langle}
  \def\rrrangle{\right \rangle \!\! \right \rangle}
$$
</div>

<p align="center">
  <img src="/assets/net_layer_sample_init/samplelayer_randomness.png">
</p>

<!-- *Preactivation distributions due to weight randomness (solid histograms) and sample randomness (red histogram) in random network initialized with Kaiming initialization* -->

>Initialize your network to preserve the variance of neurons across layers.

This prescription is pervasive in the deep learning community and its easy to see why. Not only is it effective, but it is simple to implement, and simple to understand.

For nets with ReLU nonlinearity, we implement this prescription by drawing weights from a zero mean distribution with variance $\frac{2}{\text{fan_in}}$. This is known as [Kaiming initialization](https://arxiv.org/pdf/1502.01852.pdf){:target="\_blank"}. Normalization schemes like Batch/Layer/Group/etc. Norm make variance preservation even easier to implement; by design they normalize neurons so they are zero mean and unit variance at init time.

But there is a subtlely here, over what distribution are the mean and variance computed? Kaiming preserves the variance when computed over randomness in *weights*. Batch Norm normalizes neurons using *minibatch* statistics. Layer Norm uses the mean and variance over a *layer*. Group norm uses mean and variance computed over a *group* (basically a subset of neurons in a layer).

In this post, we'll look at neuron distributions over randomness in weight configurations (network randomness), randomness in a single layer of weights (layer randomness) and randomness in samples (sample randomness). We'll see that these can be very different from each other. Kaiming initialization ensures that the variance of neurons due to network randomness is constant with depth.

With mild assumptions on the input distribution, we can actually compute the variance of neurons computed over input samples in ReLU nets initialized with Kaiming initialization. Perhaps surprisingly, we show that this variance decays to *zero* with depth. This post is closely related to a [paper](https://arxiv.org/pdf/1902.04942.pdf){:target="\_blank"} I wrote with my advisor. The calculation method is hopefully a little simpler than what we originally wrote, though perhaps less rigorous.

## Setup
We are going to examine neuron distributions in the simplest (most boring) network we can. Our network will be a 50 layer, fully connected, 1024 neuron wide, feedforward network.

<p align="center">
  <img src="/assets/net_layer_sample_init/kaiming-net.png">
</p>

The forward pass of our network is defined by:

$$ \mathbf{y}_l = \mathbf{W}_l \mathbf{x}_{l-1} \qquad \mathbf{x}_l = f(\mathbf{y}_l) $$

where $\mathbf{y}_l$ are the *pre-activations* are layer $l$ and $\mathbf{y}_l$ are the *activations* at layer $l$. The nonlinearity $f$ is the standard ReLU nonlinearity. The inputs are denoted by $\mathbf{x}_0$.

We initialize our networks with Kaiming initialization, meaning biases are zero and weights are drawn iid from a zero mean Gaussian with variance $2/N$:

$$ \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \stackrel{iid}\sim \mathcal{N}(0, 2/N) $$

We'll just assume the inputs are drawn from some arbitrary input distribution:

$$ \mathbf{x}_0 \sim Q(\mathbf{x}_0) $$

Notice that in this generic setup, there are two fundamental sources of randomness, one arising from randomness in the weights, the other arising from randomness in the samples. Standard theories of initialization deal with either a) neuron distributions due to randomness in weights only or b) neuron distributions due to randomness in both weights and samples.

In this post, we are going to see how these compare to neuron distributions in a *single network with fixed weights* where the only source of randomness is due to the samples.

## Network Randomness
We will call randomness due to fluctuations in every layer's weights *network randomness*. Specifically we will investigate the distribution of pre-activations due to network randomness:

$$ P(y_l | \mathbf{x}_0 ) \qquad \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$

Note that this distribution is a function of the input $\mathbf{x}_0$. We will indicate averages over this distribution with double brackets:

$$ \llangle y_l \rrangle = \int y_l P(y_l | \mathbf{x}_0 ) d\mathbf{W}_1 d\mathbf{W}_2 ... d\mathbf{W}_l $$


To empirically visualize this distribution, we pick a single input $\mathbf{x}_0$ and randomly sample 1000 different networks. In layers 1,5,10,50 we then pick a single neuron and visualize its distribution over the 1000 randomly generated networks:

<p align="center">
  <img src="/assets/net_layer_sample_init/network_randomness.png">
</p>

Ok, these distributions are boring. We just see a unit Gaussian at each layer.  But it's good that this is boring; Kaiming initialization is supposed to preserve the variance of each of these distributions.

Before moving on, let's show 3 simple properties of these distributions. These properties are well known in the deep learning community but I think its worth moving slowly here as *none* of these will be true when you look at neuron distributions due to sample randomness.

**1. Within a layer, every preactivation has the same distribution**

To see this theoretically, recall the relationship $\mathbf{y}\_l = \mathbf{W}\_l \mathbf{x}\_{l-1}$. Because every element of $\mathbf{W}_l$ has the same distribution and is independent of $\mathbf{x}_l$, we have that every element of $\mathbf{y}_l$ has the same distribution.

**2. Every preactivation is zero mean: $\llangle y \rrangle = 0$**

Again, we use the relationship $\mathbf{y}\_{l} = \mathbf{W}\_{l} \mathbf{x}\_{l-1}$. Because $\mathbf{W}\_l$ is independent of $\mathbf{x}\_{l-1}$, we can write $\llangle \mathbf{y}\_l \rrangle  = \llangle \mathbf{W}\_l \rrangle \llangle \mathbf{x}\_l \rrangle$. Because $\mathbf{W}\_l$ is zero mean, we have $\llangle \mathbf{y}\_l \rrangle=0$.

**3. For all layers, every preactivation has the same variance if $\llangle W\_{ij}^2 \rrangle=\frac{2}{N}$**

This is the claim that Kaiming initialization preserves neuron variance due to network randomness. We review the argument they derived in their [paper](https://arxiv.org/pdf/1502.01852.pdf){:target="\_blank"}

Because elements of $\mathbf{W}\_l$ and $\mathbf{y}\_{l-1}$ have identical and independent distributions over weights, we can write the variance of an element of $\mathbf{y}\_l$ as $\llangle y\_l^2 \rrangle = N \; \llangle W\_l^2 \rrangle \llangle f(y\_{l-1})^2 \rrangle = 2 \llangle f(y\_{l-1})^2 \rrangle$.

Over weights, elements of $\mathbf{y}\_{l-1}$ are symmetrically distributed around zero. This implies that the 2nd moment of $\text{ReLU}$ of $\mathbf{y}\_l$ is just half the 2nd moment of $\mathbf{y}\_l$: $\llangle f(y\_{l-1})^2 \rrangle = \frac{1}{2} \llangle y\_{l-1}^2 \rrangle$.

The result:

$$ \llangle y_l^2 \rrangle =  \llangle y_{l-1}^2 \rrangle $$

The variance over networks of a preactivation in layer $l$ is the same as it is in layer $l-1$. Note that iff the weight variance were larger, variance of preactivations would grow exponentially, and if the weight variance were smaller, the pre-activation variance would decay exponentially.

## Layer Randomness
Before moving to examine distributions due sample randomness, we'll examine another distribution which is closely related to network randomness, but much more convenient for the practitioner to actually measure. I am talking about *layer randomness*. This is the distribution of a neuron due to randomness in a single layer's weights, with all other weights fixed:

$$ P(y_l | \mathbf{x}_0, \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_{l-1} ) \qquad \mathbf{W}_l \sim \mathcal{N}(0, 2/N) $$

To visualize this distribution, we can sample a *single* fixed network (instead of 1000 like we did before) and now visualize the distribution of all preactivations in a layer:

<p align="center">
  <img src="/assets/net_layer_sample_init/layer_randomness.png">
</p>

These look similar to the distributions due to network randomness (though there are some finite width effects that start to show up by layer 50). So long as the network is sufficiently *wide*, these distributions will be nearly identical to the distributions in the previous section that we got by allowing weight in every layer to fluctuate.

In the rest of this post, we will ignore any subtle differences between network and layer randomness and just focus on the difference between network and sample randomness.

## Sample Randomness
Now things will get interesting. We'll visualize a preactivation's distribution due to *sample randomness*:

$$ P(y_l | \mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_{l} ) \qquad \mathbf{x}_0 \sim Q(\mathbf{x}_0) $$

This distribution is now a function of the weights in all layers (so this can be defined not just for random networks, but trained networks too). However, if we look at networks with Gaussian weights, this distribution will tend to take on certain properties.

To visualize this distribution, we generate a single network with Kaiming initialization, and this time generate 1000 different input samples. We'll visualize the empirical distribution over samples of a neurons in layers 1,5,10,50. To show that these distributions depend on the weights in a non-trivial way, we'll repeat this process 3 times to get 3 different distributions.

<p align="center">
  <img src="/assets/net_layer_sample_init/sample_randomness.png">
</p>

Completely different from the network randomness setting! Interestingly it appears that as we look deeper and deeper in the network, the distribution of each preactivation collapses to small fluctuations around some large mean value. The network appears to be implementing a nearly constant function. The location of these mean values depend on the network weights.

### Mean-Fluctuation Decomposition
We can qualitatively describe the phenomenon. Unfortunately rigorous treatment will not be possible.

Our first step will be to decompose every preactivation into the sum of two terms, its mean $\mu$ over samples and fluctuations $\nu$ around the mean:

$$ y(w,t) = \mu(w) + \nu(w,t) $$

Note that the $\mu$ only depends on the network weights, which we indicate by $\mu(w)$, and $\nu$ depends both on the input and weights, which we indicate by $\nu(w,t)$. Now we will make two approximations.

### Approximate Calculation (Wide net + random inputs)

If we are willing to make some assumptions, we can calculate the precise amount by which the variance of each pre-activation shrinks with layer.

**Approximation 1: Within a layer, the sample variance of every neuron is the same**


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

**Approximation 2: Fluctuations are Gaussian:**
$$ \nu_l \sim \mathcal{N}(0, v_l^2) $$

This might seem. Product can be non-Gaussian, Mean field over weights. This means that yes, everything in next layer is gaussian.


**Approximation 3: The mean of each neuron is Gaussian random variable:**

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

### Batch/Layer/Group Normalization
Typically, people think of normalization schemes as ways to "reduce covariate shift, smooth loss surface, etc" but we can use our theory to understand how popular normalization schemes actually alter the initialization of a network.

Standard normalization schemes normalize preactivations:
$$ y \leftarrow \frac{y - \mu}{ \sigma } $$

Batch Normalization uses the *minibatch mean* and *minibatch variance*. When your minibatch is large, these will be closely related to the sample mean, and sample variance. This implies that every preactivation will be zero mean and unit variance, a big difference from the unnormalized case!

Layer Normalization uses the *layer mean* and *layer variance*, which we've argued are closely related to network mean and variance. So this might not have such a big effect, at least in the wide network case. Is this a possible source of the discrepancy in training between layer and batch norm?

Group Normalization is something like a mix between the Batch and Layer Normalization. It normalizes with the mean and variance over subsets of neurons in a layer.

### Implications for Training
The real question is "does any of this matter for training?" My answer is definitely YES, IT CAN MATTER. In fact the whole reason I wrote this post was that while experimenting with various initializations for training UNets, I observed that normalizing pre-activations with the mean and standard deviation computed over samples (similar to Batch Norm) seemed to provide a noticeable speedup to training compared to when i used the values computed over a layer.

This originally confused me, as I incorrectly thought the two statistics would be similar. In our paper, we also trained a 10 layer "All Convolutional Net" and found that normalizing with the sample statistics at initialization outperformed normalization with layer statistics.

<p align="center">
  <img src="/assets/net_layer_sample_init/learning_curves.png">
</p>

Interestingly, just reinitializing seemed to provide a substantial fractino of the speedup provided by Batch Norm.

Now, for the question of does it always matter, and when does it matter? I'm less sure about to this. It seems like sometimes reinitializing is not always helpful. Check out David Page's [post](https://myrtle.ai/how-to-train-your-resnet-7-batch-norm/){:target="\_blank"} for some info!

#### Acknowledgements
The post originated from work done with my advisor Sebastian Seung.
