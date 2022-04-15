---
layout: page
title: About
permalink: /about/
---
![image](/assets/circular_profile.png){:height="200px" width="200px" style="float: right"}

Hi!

I'm a 6th year physics PhD student at Princeton advised by [Sebastian Seung](https://twitter.com/sebastianseung?lang=en){:target="\_blank"}. My current research interests lie at the intersection of deep learning, computer vision, and computational neuroscience. Feel free to research out if you want to get in touch!

[email](mailto:kluther@princeton.edu) &nbsp; \| &nbsp;  [CV](/assets/CV.pdf) &nbsp; \| &nbsp; [Google Scholar](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F5e0yPGmYrQrZ9lske_v4RPq7xURWD5Z9iJGyfnmTQL4rYTaBSksBIrwBWBx732XmQAtC4IklkW_Y7KQPO32WMjzxA06w&user=JX_K0-QAAAAJ) &nbsp; \| &nbsp; [GitHub](https://github.com/KyleLuther)
{: style="text-align: center"}

### Research
<!-- ![image](/assets/sparsecoding_overview.png){:height="100px" width="225px" align="left" style="padding-right: 25px"}
[Sensitivity of sparse codes to image distortions](https://ieeexplore.ieee.org/abstract/document/9048957)  
**Kyle Luther**, H. Sebastian Seung  
Arxiv (accepted and soon to appear in Neural Computation), 2022

We use convolutional networks to segment 3D volumetric microscopy images. This work uses the metric learning/dense voxel embedding strategy: learn to assign a vector to every pixel such that pixels from the same neuron are assigned the same vector, and pixels from different neurons are assigned different vectors. -->

![image](/assets/metriclearning3d_overview.png){:height="130px" width="225px" align="left" style="padding-right: 25px"}
[Learning and segmenting dense voxel embeddings for 3D neuron reconstruction](https://ieeexplore.ieee.org/abstract/document/9048957)  
Kisuk Lee, Ran Lu, **Kyle Luther**, H. Sebastian Seung  
IEEE Transactions on Medical Imaging, 2021

We use convolutional networks to segment 3D volumetric microscopy images. This work uses the metric learning/dense voxel embedding strategy: learn to assign a vector to every pixel such that pixels from the same neuron are assigned the same vector, and pixels from different neurons are assigned different vectors.

![image](/assets/reexamining_overview.png){:height="225px" width="225px" align="left" style="padding-right: 25px"}
[Reexamining the principle of mean-variance preservation for neural network initialization](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033135)  
**Kyle Luther**, H. Sebastian Seung  
Physical Review Research, 2020  

The centering component of Batch Normalization has a dramatic impact on network initialization properties. We prove that in a deep & wide ReLU network, all inputs are mapped to very similar outputs at initialization time. By centering every layer with Batch Norm, feature collapse is prevented.

<br clear="all" />

![image](/assets/cgame_overview.png){:height="150px" width="225px" align="left" style="padding-right: 25px"}
[Unsupervised learning by a softened correlation game: duality and convergence](https://ieeexplore.ieee.org/abstract/document/9048957)  
**Kyle Luther**, Runzhe Yang, H. Sebastian Seung  
The Fifty-Third Asilomar Conference on Signals, Systems & Computers, 2019

Recent models have posed cortical learning a two player zero-sum game: excitatory neurons compete with inhibitory neurons to maximize/minimize an objective. We show empirically that the learning dynamics of this game do not converge when the inhibitory neurons learn too slowly. We also provide a fast algorithm which is guaranteed to converge to a min-max point.

![image](/assets/metriclearning_overview.png){:height="225px" width="225px" align="left" style="padding-right: 25px"}
[Learning metric graphs for neuron segmentation in electron microscopy images](https://ieeexplore.ieee.org/abstract/document/9048957)  
**Kyle Luther**, H. Sebastian Seung  
IEEE 16th International Symposium on Biomedical Imaging, 2019

We use convolutional networks to segment microscopy images of brain tissue. Unlike previous approaches which detect boundaries and generate segments by filling in boundaries, we use a metric learning approach to assign all pixels from a neuron the same vector, while assigning different vectors to pixels from different objects.

<!-- ### Code -->
