---
layout: post

title: "Non-negative Quadratic Programming: Local Optima, Copositive Matrices, Permitted and Forbidden Sets"
date: 2020-02-18 10:00:00
keywords: Copositive matrices, forbidden sets, permitted sets, non-negative quadratic programming
---

Consider the following optimization:

$$ \min_{\mathbf{x} \geq 0} \; \mathbf{b}^{\top} \mathbf{x} + \frac{1}{2} \mathbf{x}^{\top} \mathbf{M} \mathbf{x} $$

where $\mathbf{M}$ is a symmetric $n \times n$ matrix and $\mathbf{x} \geq 0$ is a non-negativity constraint on every element of $\mathbf{x}$.

This might show up in ..


Without the non-negativity constraints, this would be in some ways a boring optimization problem. If all the eigenvalues of $\mathbf{M}$ are non-negative, i.e. the objective looks like a bowl, then this is a convex optimization problem. Further we can write the solution as $\mathbf{x} = -\mathbf{M}^{+} \mathbf{b}$. And if $\mathbf{M}$ has any negative eigenvalues, then $\min$ is undefined.

With the non-negativity constraints, things get more interesting. If all the eigenvalues of $\mathbf{M}$ are non-negative, this is still a convex problem, though a closed-form solution is generally not available.

But when $\mathbf{M}$ is indefinite,  there is now the possibility that the constraints prevent $\mathbf{x}$ from diverging to $-\infty$. In this setting, the objective is non-convex (it looks like a saddle), and finding global minima is known to be NP-HARD.

In this post we'll review the conditions on $M$ for minima to exist. Then we'll examine some properties of local minima in the non-convex setting.

When $M$ is copositive is a necessary and sufficient condition. Local minima must be part of permitted sets. Basically any submatrix of $M$ must be positive semi-definite.

In their (paper)[https://papers.nips.cc/paper/1793-permitted-and-forbidden-sets-in-symmetric-threshold-linear-networks.pdf], Hahnloser and Seung show that there is actually a relatively simple condition.

In particular, when $M$ is copositive, the min is defined. And Further, Regardless of $b$, there exist forbidden sets. Basically, the submatix defined by the solution must be positive semi-definited.


## Preliminaries: What does the objective look like?
Let's ignore the non-negativity constraint for now. What does the objective look like as a function of $\mathbf{x}$? With some arithmetic, we can rewrite the objective function:

$$ \mathbf{b}^{\top} \mathbf{x} + \frac{1}{2} \mathbf{x}^{\top} \mathbf{M} \mathbf{x} = \mathbf{a} + \frac{1}{2} (\mathbf{x}-\mathbf{x_0})^{\top} \mathbf{M} (\mathbf{x}-\mathbf{x_0}) $$

where $\mathbf{x}_0,\mathbf{a}$. Basically the slope $b$ is simply determining where the singular critical point of the quadratic form is. The "defining" characteristic, the curvature, is all contained in the matrix $M$.

Without the constraints, we have to have that this objective looks like a bowl. If there are any directions that slope downwards, then $\mathbf{x}.

 $\mathbf{M}$

When $\mathbf{M}$ is positive definite, contains only positive eignvalues. \mathbf{x} M x > 0.

Necessary and Sufficient is that $\mathbf{M}$ is positive definite,

## Condition for Existence of a Minimum: M is Copositive
Adding the non-negativity constraints. In other words, $M$ being positive definite is a sufficient condition for the optimization to be defined for all $\mathbf{b}$. But unlike, it turns out that it is not a necesary one.

What is wrong with this picture work when you have non-negativity constraints?

It's easiest to see with some graphics.

What if the eigenvalues have negative elements?

## Finding Local Minima: Projected Gradient Descent

## Properties of Local Minima: Forbidden and Permitted Sets
Let's think about the previous example. The directions of negative curvature, the loss decreases. Its going to do so until it hits a wall.

Suppose you have

#### Projected Gradient Descent


Key intuition: if there is a negative eigenvalue, then x will try to remove it, it cant be a local max.


## Examples
