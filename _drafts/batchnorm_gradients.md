Everyone loves Batch Norm. It often improves training, fool proofs your initialization and learning rate selection, and often even improve generalization performance. [Recent work][https://openreview.net/pdf?id=SyMDXnCcF7] showed that, perhaps surprising, Batch Normalization, can yield exploding gradients. The explanation was very technical however.

**TL;DR** Inserting Batch Norm into a network means that in the forward pass each neuron is divided by its standard deviation, $\sigma$, computed over a minibatch of samples. In the backward pass, gradients get divided by the same $\sigma$. In wide ReLU networks, we can approximate $\sigma \approx \frac{\pi}{\pi-1} \approx 0.8$. Since this occurs at each layer, gradient norms grow like $\frac{\pi}{\pi-1}^L$ with depth.

## Numerical Simulation in Random Networks
Before the actual calculation, let's just run a simulation to show that inserting Batch Norm into a random network actually can cause gradients to explode.

We'll initialize a boring old 10 layer, feedforward, fully connected network of uniform width $N=1024$. We'll use the ReLU activation function, and initialize with Kaiming initialization: biases are set zero and elements of $W$ are drawn i.i.d. from a zero-mean Gaussian with variance $\sqrt{2/N}$. We'll generate 100 Gaussian white noise input vectors $\{\mathbf{x}_0(t):t=1,2,...,128\}$ and generate a random linear loss function: $E = \sum_{t=1}^{128} \mathbf{w} \cdot \mathbf{x}_{L}(t) \quad \mathbf{w} \sim \mathcal{N}(0,1)$


<!-- ![Vanilla Architecture](/assets/vanilla_net.png) -->

We'll use PyTorch to automatically compute the derivatives $dE/d\mathbf{x}_l(t)$ at each layer $l$ and for every input $t$. This gives us 128x1024 numbers at each layer. We visualize the histogram of these 128x1024 gradients at each layer in the left plot.

We then take that exact network, and insert Batch Normalization before the ReLU in every layer, and recompute gradients using the same loss function. We visual the histogram of these 128x1024 gradients at each layer in the right plot.

<!-- ![Gradient Simulation](/assets/bn_gradient_simulation.png =400x200) -->
<p align="center">
  <img width="400" height="200" src="/assets/bn_gradient_simulation.png">
</p>

In the vanilla network, the gradient histograms are basically the same in every layer. However, after inserting Batch Norm, the gradients actually grow with decreasing layer. In fact, the widths of these histograms grow *exponentially*.

This suggests that after inserting Batch Norm, the network outputs $\mathbf{x}_L$ are much more sensitive to changes in earlier layers than higher layers. This seems odd. Intuitively it seems like having an exponential gradient scaling seems undesirable for training. Yet Batch Norm seems to improve training speed across such an enormous range of tasks, which suggests maybe the gradient scaling isn't actually that important.

I really am puzzled by this observation and I'm really not sure how to relate gradient scales to training performance. However, I can at least try to understand *why* gradients explode in the first place.

## Typical Gradients in a Vanilla Net
We want to see how the "typical scale" of gradients, roughly the width of each layer's gradient histogram, is changed by inserting Batch Norm into a randomly initialized deep network. In the simulation we saw that for vanilla nets this width is the same in each layer. This is expected. The original Kaiming paper argued this should be true.

 We'll show this first, this calculation will be used for the Batch Norm calculation anyways.

These results have already been shown

More precisely, we will be interested in how:

$$ \llangle \left|\frac{dE}{ d\mathbf{x}_l}\right|^2 \rrangle $$
propagates through layers of network.

Our basic strategy will be to first derive the backwards pass for each of the 3 modules in each layer of the Batch Norm nets. We will then average these over weights.

<!--
 Our general strategy will be:
1. Define the forward pass
2. Derive the backward pass
3. Average the backward pass over weights

We will make a number of simplifying assumptions for our calculations: we examine fully connected network with rectifying nonlinearity, initialized with Kaiming initialization. We also assume the inputs are Gaussian white noise. -->

Even after a lot of simplification, the backwards pass equations shown in the previous section are pretty ugly. They don't give a very intuitive feel for how gradients are modified by the Batch Norm layer. Also they depend on the precise configuration of $W$ which are random variables. So to gain some insight, we will compute the average over weights.

Since the W are independent random variables in each layer, we would ideally get some set of equations that look like this:
$$ \tilde{z} = dE / dx^l$$

Unfortunately, the gradients are functions of the activations so they are not independent. However, we will essentially cheat and assume these are independent variables.

Each layer consists of 3 modules. We'll work our way through each one

#### Matix Multipy
I'll assume you're mostly familiar. If you want a tutorial on how to get these, check out these other blogs!

#### ReLU Activation

#### Gradients are Preserved


## Typical Gradients with Batch Norm
In some sense, 2/3 of the work is done. We just need to extend the argument in the previous section to determine the impact of the Batch Norm layer. In a much more practical sense, most of the work is still to come as backpropagating through the normalization layer is much more complicated.

For this calculation we have to be careful in distinguishing two sorts of averages. We indicate a *minibatch* average using with single brackets $\langle \cdot \rangle$ and as before indicate a *weight* average using double brackets $\llangle \cdot \rrangle$

![BatchNorm Forward Pass](/assets/bn_forward.png)

The above diagram shows one layer of a net with Batch Norm. The forward and backward pass are shown as well for a single input vector. Note that because this is just an initialization time analysis, we can ignore the two parameters in Batch Norm, $\gamma$ and $\beta$, but these are one and zero at init time so we will ignore this. Also for numerical reasons, BatchNorm actually defines $\sigma$ as the variance plus some $\epsilon$ just to avoid possible division by zero.

If you want a derivation for those gradients, check out these blogs!

A complication not present in the vanilla case is the gradients that come by backpropagating through the minibatch mean $\mu$ and variance $\sigma$.

First, We'll argue that in wide random nets, these actually don't really contribute substantially to the gradient and we can basically ignore these. Second, we'll show how to estimate $\sigma$ in ReLU nets. We'll combine these results, along with the result that gradient norms are preserved by the combination of the matrix multiply and ReLU modules, to show that in this setting, gradients are amplified at each layer.

<!-- #### Approximation 1: $\langle \frac{dE}{d\mathbf{z}} \rangle \approx 0$
We argue that gradients of preactivations should be nearly zero mean over a minibatch: $\langle \frac{dE}{d\mathbf{z}} \rangle \approx 0$. By averaging the backward pass equation for $y$ over a minibatch, it's actually easy to show that gradients w.r.t. $y$ are exactly zero:
$$ \langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \cancelto{0}{\langle \frac{dE}{d\mathbf{z}} - \langle} \frac{dE}{d\mathbf{z}} \rangle \rangle - \frac{1}{\sigma} \cancelto{0}{\langle \mathbf{z} \rangle} \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$

To derive $dE/dz$ from $dE/dy$,


#### Approximation 2: $\langle z \frac{dE}{d\mathbf{z}} \rangle \approx 0$
#### Approximation 3: $\llangle \sigma^2 \rrangle \approx \frac{\pi-1}{\pi}$
#### Approximation 4: $\llangle \frac{1}{\sigma^2} \rrangle \approx \frac{1}{\llangle \sigma^2\rrangle}$ -->

#### Approximation 1: $\langle \frac{dE}{dz} \rangle \approx \langle z \frac{dE}{dz} \rangle \approx 0$
We are going to argue that $\langle \frac{dE}{dz} \rangle$, the term from backpropagating through $\mu$, and $\langle z \frac{dE}{dz} \rangle$, the term from backpropagating through $\sigma$, are nearly zero at initialization time, except possibly in the last layer.

We extend  the gradient independence assumption to assume that, over the minibatch distribution, forward pass quantities are independent from backward pass quantities. With this assumption we have: $\langle z \frac{dE}{dz} \rangle \approx \langle z \rangle \langle \frac{dE}{dz}  \rangle$. Basically the gradient w.r.t. $z$ is independent of the value of $z$ itself.

Now we need to show that the average gradient $\langle \frac{dE}{dz}  \rangle$ is roughly zero. Without any assumptions, we can show that gradients $\frac{dE}{dy}$ are exactly zero mean in a Batch Normalized net:
$$\langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \cancelto{0}{\langle \frac{dE}{d\mathbf{z}} - \langle} \frac{dE}{d\mathbf{z}} \rangle \rangle - \frac{1}{\sigma} \cancelto{0}{\langle \mathbf{z} \rangle} \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$
Now we can use the backwards pass equations to write the gradient for $z_l$ using the gradient for $y_{l+1}$:
$$\frac{dE}{dz_i} = \sum_{j} f'(z_i) W^T_{ij} dE/dy_j $$

Now here comes the extendend gradient independence assumption. If $dE/dy_j$ are independent of $W$ and $f'$, then we can write this sum as:
$$ \langle \frac{dE}{dz_i} \rangle \approx \sum_{j} \langle f'(z) \rangle W^T_{ij} \langle dE/dy_j \rangle = 0$$

Note that this explanation doesn't really hold up in the last layer ($dE/dz$ was zero only if there was a Batch Norm layer afterwards to center gradients).

#### Approximation 2: $\sigma^2 \approx \frac{\pi-1}{\pi}$
Calculating $\sigma$ is in some ways the heart of our whole calculation. In the previous section we argued that $dE/dy \approx \frac{1}{\sigma} dE/dz$. We also argued that the combination of the matrix multiplication layer and nonlinearity layer preserve gradient norms on average. So the key thing determining how the typical size of gradients backpropagates through layers of a network is goign to be $\sigma$. For ReLU we'll see that $\sigma = 0.8$ so gradients grow exponentially by a factor of $1/0.8$

To do this, we're going to make two a physics-style assumptions. First we'll do is assume that $\sigma$, the minibatch variance of $y$, is a *self-averaging* quantity: this means we assume that every $y$ has the same minibatch variance. This will be helpful as we can compute the average

. (Actually it follows from the mean-field assumption, but im not sure how to Intuitively justify it without getting bogged down in eigenvalues of random matrix products. So let's just take it as another independent assumption.)


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

## Normalized Preactivations Imply Exploding Gradients?
To get some intuition, we'll actually look at the forward pass of a vanilla and Batch Normalized net.

go back to the vanilla net. We'll look at the distribtuion of preactivations in later layers. Interestingly each one gets a mean value which grows with depth. Normalization completely changes this pciture.

What does this have to do with gradients? Well by normalizing, you're basically requiring the nonlinearity actually gets used. Normalizing every preactivation ensures you are now implementing some crazy nonlinear function.


## Does this matter for Training?
