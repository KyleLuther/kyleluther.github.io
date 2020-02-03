Everyone loves Batch Norm. It often improves training, fool proofs your initialization and learning rate selection, and often even improve generalization performance. [Recent work][https://openreview.net/pdf?id=SyMDXnCcF7] showed that, perhaps surprising, Batch Normalization, can yield exploding gradients. The explanation was very technical however.

**TLDR:** Inserting Batch Norm into a network means that in the forward pass each neuron is divided by its standard deviation, $\sigma$, computed over a minibatch of samples. In the backward pass, gradients get divided by the same $\sigma$. In ReLU netwoks, we can approximate $\sigma \approx \frac{\pi}{\pi-1} \approx 0.8$. Since this occurs at each layer, gradient norms grow like $\frac{\pi}{\pi-1}^L$ with depth.

## Numerical Simulation

## Theoretical Explanation
We want to see how the "typical scale" (specifically the average squared gradient norm) is changed by inserting Batch Norm into a randomly initialized deep network. Our general strategy will be:
1. Define the forward pass
2. Derive the backward pass
3. Average the backward pass over weights

We will make a number of simplifying assumptions for our calculations: we examine fully connected network with rectifying nonlinearity, initialized with Kaiming initialization. We also assume the inputs are Gaussian white noise.

## Setup
![BatchNorm Forward Pass]({{ site.url }}/assets/bn_forward.png)
The above diagram shows one layer, $l$, of a network with Batch Norm.
The notation
$$ \langle \cdot \rangle := \text{ minibatch average } $$
For this calculation it will be important to be careful about which sorts of averages we are doing. Intuitively, averaging over weights is easier as they are IID but BN averages over samples here and this is not the same.

Additionally, this is just an initialization time analysis. BN has two more parameters, $\gamma$ and $\beta$ but these are one and zero at init time so we will ignore this.

In this section, we'll review how to compute derivatives $dE/x^l$ if we know this derivative in the layer in front. Recall our notation $\tilde{y} = dE/dy$

We can calculate gradients in a Batch Normalized network with the standard chain rule-based backprop algorithm. Assuming we know $dE/dx^l$, we can use the forward pass equation to compute:
$$ dE/dz^l = $$
$$ dE/dy^l = $$
$$ dE dx^{l-1} = $$

The point of this article is not to provide an in depth. However if you want one, check out some of these blog posts. Ill just provide a simple answer:

## "Typical" Backward Pass without Normalization
Even after a lot of simplification, the backwards pass equations shown in the previous section are pretty ugly. They don't give a very intuitive feel for how gradients are modified by the Batch Norm layer. Also they depend on the precise configuration of $W$ which are random variables. So to gain some insight, we will compute the average over weights.

Since the W are independent random variables in each layer, we would ideally get some set of equations that look like this:
$$ \tilde{z} = dE / dx^l$$

Unfortunately, the gradients are functions of the activations so they are not independent. However, we will essentially cheat and assume these are independent variables.

Each layer consists of 3 modules. We'll work our way through each one
#### Matrix Multiplication ($y^l = W x^{l-1}$):
One nice thing about averaging over $W$ is that the individual elements of vectors all have the same value: $ \llangle y_i^2 \rrangle = \frac{1}{N} \llangle |y|^2 \rrangle$. We will therefore be somewhat cavalier with our notation: When examining averages, this means any particular value will be the same.

$$ \llangle \tilde{x}_{l-1}^2 \rrangle = \sum_{jk} \llangle W_{ij} W_{ik} \rrangle \llangle (\tilde{y}^{l}_i)^2 \rrangle
$$

$$ \boxed{\llangle \tilde{x}_{l-1}^2 \rrangle = 2 \llangle \tilde{y}_{l}^2 \rrangle}$$

#### Activation ($x^l = f(z^l)$):

$$ \boxed{\llangle \tilde{z}_{l}^2 \rrangle = \frac{1}{2} \llangle \tilde{x}_{l}^2 \rrangle} $$

## "Typical" Backward Pass with Normalization:
1. Show $ \langle dE / dz \rangle \approx 0 $
2. Show $ dE / d y \approx 1/sigma dE / dz $
3. Show $ \llangle \frac{1}{\sigma}^2 \rrangle $


This one is definitely the tricky one.

#### Simplify gradient
The critical observation is this. For a random network initialized with random weights:
$$ \langle \frac{dE}{d\mathbf{z}} \rangle \approx 0 $$


We'll justify this in more depth shortly. For now, let's just provide an intuitive reason that this should hold. It's actually easy to show that this is exactly true for gradients w.r.t. $y$

$$ \langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \langle \frac{dE}{d\mathbf{z}} - \langle \frac{dE}{d\mathbf{z}} \rangle \rangle - \frac{1}{\sigma} \langle \mathbf{z} \rangle \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$

$$ \langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \cancelto{0}{\langle \frac{dE}{d\mathbf{z}} - \langle} \frac{dE}{d\mathbf{z}} \rangle \rangle - \frac{1}{\sigma} \cancelto{0}{\langle \mathbf{z} \rangle} \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$

Extending the gradient assumption
$$ \langle z \frac{dE}{d\mathbf{z}} \rangle \approx \langle z \rangle \langle \frac{dE}{d\mathbf{z}} \rangle \approx \mathbf{0} $$
Huge simplification to gradients:

$$ \frac{dE}{d\mathbf{y}} \approx \frac{1}{\sigma}  \circ \frac{dE}{d\mathbf{z}} $$

Like before we want to average this weights

$$ \llangle \left|\frac{dE}{d\mathbf{y}}\right|^2 \rrangle \approx \llangle \frac{1}{\sigma}^2 \rrangle \llangle \frac{dE}{d\mathbf{z}} \rrangle $$

What is the typical sigma??
#### What is $\llangle \frac{1}{\sigma^2} \rrangle $
In some sense this is the heart of the calculation. We'll use a simulation at this point.

The critical observation is that the variance of each $y$ is nearly the same and it is roughly $ 0.8 $. At this point we could pack things up. Fair enough, but let's try to intuitively see why this is going on.

Our essential modeling assumption will be that $x$ at every layer are IID variables. With this assumption, well get

1. Every $z$ is unit Gaussian over samples
2. $x=f(z)$ so the variance of every $x$ can be computed via a simple gaussian integral
3. $y=wx$ so the expected variance of $y$is twice that of $x$
4. fluctuations of $y$ are small so we replace $\llangle 1 / sigma \rangle \approx \frac{1}{ \langle \sigma^2 \rangle }$










## Normalized Preactivations Imply Exploding Gradients?

## Does this matter?
