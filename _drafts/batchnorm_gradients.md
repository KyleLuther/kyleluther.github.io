Everyone loves Batch Norm. It often improves training, fool proofs your initialization and learning rate selection, and often even improve generalization performance. [Recent work][https://openreview.net/pdf?id=SyMDXnCcF7] showed that, perhaps surprising, Batch Normalization, can yield exploding gradients. The explanation was very technical however.

**TLDR:** Inserting Batch Norm into a network means that in the forward pass each neuron is divided by its standard deviation, $\sigma$, computed over a minibatch of samples. In the backward pass, gradients get divided by the same $\sigma$. In ReLU netwoks, we can approximate $\sigma \approx \frac{\pi}{\pi-1} \approx 0.8$. Since this occurs at each layer, gradient norms grow like $\frac{\pi}{\pi-1}^L$ with depth.

## Setup
We want to see how the "typical scale" of gradients is changed by inserting Batch Norm into a randomly initialized deep network. Our general strategy will be:
1. Define the forward pass
2. Derive the backward pass
3. Average the backward pass over weights

We will make a number of simplifying assumptions for our calculations: we examine fully connected network with rectifying nonlinearity, initialized with Kaiming initialization. We also assume the inputs are Gaussian white noise.

## Forward Pass with Batch Norm
![BatchNorm Forward Pass]({{ site.url }}/assets/bn_forward.png)
The above diagram shows one layer, $l$, of a network with Batch Norm.
The notation
$$ \langle \cdot \rangle := \text{ minibatch average } $$
For this calculation it will be important to be careful about which sorts of averages we are doing. Intuitively, averaging over weights is easier as they are IID but BN averages over samples here and this is not the same.

Additionally, this is just an initialization time analysis. BN has two more parameters, $\gamma$ and $\beta$ but these are one and zero at init time so we will ignore this.

## Backward Pass with Batch Norm
In this section, we'll review how to compute derivatives $dE/x^l$ if we know this derivative in the layer in front. Recall our notation $\tilde{y} = dE/dy$

We can calculate gradients in a Batch Normalized network with the standard chain rule-based backprop algorithm. Assuming we know $dE/dx^l$, we can use the forward pass equation to compute:
$$ dE/dz^l = $$
$$ dE/dy^l = $$
$$ dE dx^{l-1} = $$

The point of this article is not to provide an in depth. However if you want one, check out some of these blog posts. Ill just provide a simple answer:

## "Typical" Backward Pass
Even after a lot of simplification, the backwards pass equations shown in the previous section are pretty ugly. They don't give a very intuitive feel for how gradients are modified by the Batch Norm layer. Also they depend on the precise configuration of $W$ which are random variables. To gain some insight, we will compute the average over weights.

We would like some set of equations that look like this:
$$ dE/ dy^l = dE / dx^l$$

Unforunately, the gradients are in general function of the activations, this means. However, we will essentially cheat and assume these are independent variabels.

Module 1

## Normalized Preactivations Imply Exploding Gradients?

## Does this matter?
