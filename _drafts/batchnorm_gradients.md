Everyone loves Batch Norm. It often improves training, fool proofs your initialization and learning rate selection, and often even improve generalization performance. [Recent work][https://openreview.net/pdf?id=SyMDXnCcF7] showed that, perhaps surprising, Batch Normalization, can yield exploding gradients. The explanation was very technical however.

**TL;DR** Inserting Batch Norm into a network means that in the forward pass each neuron is divided by its standard deviation, $\sigma$, computed over a minibatch of samples. In the backward pass, gradients get divided by the same $\sigma$. In ReLU netwoks, we can approximate $\sigma \approx \frac{\pi}{\pi-1} \approx 0.8$. Since this occurs at each layer, gradient norms grow like $\frac{\pi}{\pi-1}^L$ with depth.

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

#### Approximation 1: $\frac{dE}{dy} \approx \frac{1}{\sigma} \frac{dE}{dz}$
Essentially we are going to argue that the terms that come from backpropagating through $\mu$ and $\sigma$ don't actually matter at initialization time. First we'll show this with numerical simulation.

We can also provide a handwavy arguement.

To argue this we extend to the gradient independence assumption to assume that over the minibatch distribution, forward pass quantities are independent from backward pass quantities. Speficially we will assume $\langle z \frac{dE}{dz} \rangle \approx \langle z \rangle \langle dE/dz \rangle$ and $\langle f'(z) dE/dy \rangle \approx \langle f'(z) \rangle \langle dE/dy \rangle$.

With this assumption the 3rd term, which comes from backpropagating through $\sigma$, is roughly zero. (Batch Norm explicity requires $\langle z \rangle=0$):
$$ \langle z \frac{dE}{dz} \rangle \approx \langle z \rangle \langle dE/dz \rangle = 0$$

We can also argue the 2nd term is zero. Without any assumptions, we can show that gradients w.r.t. $y$ are exactly zero:
$$\langle \frac{dE}{d\mathbf{y}} \rangle = \frac{1}{\sigma} \cancelto{0}{\langle \frac{dE}{d\mathbf{z}} - \langle} \frac{dE}{d\mathbf{z}} \rangle \rangle - \frac{1}{\sigma} \cancelto{0}{\langle \mathbf{z} \rangle} \circ \langle \mathbf{z} \circ \frac{dE}{d\mathbf{z}} \rangle = 0$$

We can write $\frac{dE}{dz_i} = \sum_{j} f'(z) W^T_{ij} dE/dy_j$. Now comes the handwaving. We are going to assume that $f'(z) \perp dE/dy$. This can be seen as an extension of the gradient independence assumption. so we can average this over the minibatch:
$$ \langle \frac{dE}{dz_i} \rangle \approx \sum_{j} \langle f'(z) \rangle W^T_{ij} \langle dE/dy_j \rangle = 0$$

Note that this explanation doesn't really hold up in the last layer ($dE/dz$ was zero only if there was a Batch Norm layer afterwards to center gradients).

 It's understandable if you don't like these assumptions. Isn't the reason backprop is effective is the fact that $dE/dz$ and $z$ are correlated? Yes, but basically we hope there is enough "randomness" here.

#### Approximation 2: $\sigma^2 \approx \frac{\pi-1}{\pi}$
In some sense this is the heart of the calculation. If $\sigma$ is typically less than 1, gradients will be amplified at each layer. Again, we'll just use a simulation to show this at first.

Our essential assumption will be analogous to the mean-field assumptions. This time, we assume that over the sample distribution elements of $x$ are independent random variables. This implies that $z$ is a gaussian, and we know it is zero mean and unit variance.



We'll use a simulation at this point.

The critical observation is that the variance of each $y$ is nearly the same and it is roughly $ 0.8 $. At this point we could pack things up. Fair enough, but let's try to intuitively see why this is going on.

Our essential modeling assumption will be that $x$ at every layer are IID variables. With this assumption, well get

#### Result: $\llangle \frac{dE}{dy} \rrangle \approx \frac{\pi}{\pi-1} \llangle \frac{dE}{dz} \rrangle$

1. Every $z$ is unit Gaussian over samples
2. $x=f(z)$ so the variance of every $x$ can be computed via a simple gaussian integral
3. $y=wx$ so the expected variance of $y$is twice that of $x$
4. fluctuations of $y$ are small so we replace $\llangle 1 / sigma \rangle \approx \frac{1}{ \langle \sigma^2 \rangle }$


## Normalized Preactivations Imply Exploding Gradients?

## Does this matter?
<!-- #### Matrix Multiplication ($y^l = W x^{l-1}$):
One nice thing about averaging over $W$ is that the individual elements of vectors all have the same value: $ \llangle y_i^2 \rrangle = \frac{1}{N} \llangle |y|^2 \rrangle$. We will therefore be somewhat cavalier with our notation: When examining averages, this means any particular value will be the same.

$$ \llangle \tilde{x}_{l-1}^2 \rrangle = \sum_{jk} \llangle W_{ij} W_{ik} \rrangle \llangle (\tilde{y}^{l}_i)^2 \rrangle
$$

$$ \boxed{\llangle \tilde{x}_{l-1}^2 \rrangle = 2 \llangle \tilde{y}_{l}^2 \rrangle}$$

#### Activation ($x^l = f(z^l)$):

$$ \boxed{\llangle \tilde{z}_{l}^2 \rrangle = \frac{1}{2} \llangle \tilde{x}_{l}^2 \rrangle} $$ -->
