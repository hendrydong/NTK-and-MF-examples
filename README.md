# NTK-and-MF-examples



## Description

This repo contains some examples of mean-field regime (MF) [6] and neural tangent kernel regime (NTK) [4,5], which are mathematical models of neural networks. Under different conditions, the dynamics of neural network is prone to behave as one of them. This repo and [1] introduces the differences between these two regimes and tries to illustate the conditions of them. For simplicity, we only consider the two-layer NN case, which is written as

![](http://latex.codecogs.com/gif.latex?f([u,\theta],x)=\frac{\alpha}{m}\sum_{j=1}^mu_jh(\theta_j,x),)

where input ![](http://latex.codecogs.com/gif.latex?x\in\mathbb{R}^d), feature learning parameter  ![](http://latex.codecogs.com/gif.latex?\theta_j\in\mathbb{R}^d),   importance parameter ![](http://latex.codecogs.com/gif.latex?u_j\in\mathbb{R}^k), and scaling factor ![](http://latex.codecogs.com/gif.latex?\alpha).



### NTK

Our NTK part focus on the priciple of linearization, which is the core idea of tangent kernel. When the distance between initialization and optimized NN  ![](http://latex.codecogs.com/gif.latex?\Vert\theta-\tilde\theta\Vert) is not large, it can be analysed with NTK regime. This circumstance indeed happens when ![](http://latex.codecogs.com/gif.latex?m) and ![](http://latex.codecogs.com/gif.latex?\alpha) are large. 

Also, we have pointed out some other factors may also changes ![](http://latex.codecogs.com/gif.latex?\Vert\theta-\tilde\theta\Vert), such as learning rate, momentum and even the initalization of ![](http://latex.codecogs.com/gif.latex?\tilde\theta). And we have illustrate the failure of linear approximation of some practical NN. Thus, some practical tricks may break the NTK regimes and need more investigation for further study.

More detailed information are in [1,4,5] and "./NTK/"

### MF

Unlike NTK, the MF does not restrict ![](http://latex.codecogs.com/gif.latex?\Vert\theta-\tilde\theta\Vert), but assume the i.i.d. property between each  ![](http://latex.codecogs.com/gif.latex?\theta_j). This relax the search space of each particle ![](http://latex.codecogs.com/gif.latex?\theta_j) by compromising the correlation between them. More importantly, this property is also an ideal case for practical NNs, consistent with tricks such as Dropout, Batch Normalization. However, the theoretical analysis of MF are still preliminary compared with NTK.

In this repo, we are focusing on the distributional change of particles and investigating the effective neurons of real NNs.

More detailed information are in [1,2,3,6] and "./MF/MF.ipynb"


### Repopulation

Feature repopulation is a consequence of MF regime [2,3,7]. In particular, rather than learning the tangent space (parameters of trained NN are very closed to initialization), MF regime moves the whole distribution of NN parameters. 

We compared the feature produced by repopulated distribution and initialized distribution and find the effetiveness of the first one.

Codes are provided in "./Repopulation".



## References

[1] C. Fang, H. Dong, T. Zhang. Mathematical Models of Overparameterized Neural Networks. *Proceedings of the IEEE*, 2021.

[2] C. Fang, H. Dong, T. Zhang. "Over parameterized two-level neural networks can learn near optimal feature representations". *arXiv preprint arXiv:1910.11508*, 2019.

[3] C. Fang, J. D. Lee, P. Yang, and T. Zhang, "Modeling from features: a mean-ﬁeld framework for over-parameterized deep neural networks". *arXiv preprint arXiv:2007.01452*, 2020.

[4] A. Jacot, F. Gabriel, and C. Hongler, "Neural tangent kernel: Convergence and generalization in neural networks".  *Advances in neural information processing systems*, 2018.

[5] S. S. Du, J. D. Lee, H. Li, L. Wang, and X. Zhai, "Gradient descent ﬁnds global minima of deep neural networks". *International Conference on Machine Learning*, 2019.

[6] S. Mei, A. Montanari, and P.-M. Nguyen, "A mean ﬁeld view of the landscape of two-layer neural networks". *Proceedings of the National Academy of Sciences*, vol. 115, no. 33, pp. E7665–E7671, 2018.

[7] W. Zhang, Y. Gu, C. Fang, J. Lee, and T. Zhang, "How to Characterize The Landscape of Overparameterized Convolutional Neural Networks".  *Advances in neural information processing systems*, 2020.



## Citations

The visualization and illustration in this repo came primarily out of research in [Statistics and Machine Learning Research Group](http://statml.hkust.edu.hk/) at HKUST. 

For detailed description you can refer to 

[Mathematical Models of Overparameterized Neural Networks](https://arxiv.org/abs/2012.13982)

If you find it helpful, you can cite

```
@ARTICLE{fang2021mathematical,
  author = {Cong Fang, Hanze Dong, Tong Zhang},
  journal={Proceedings of the IEEE}, 
  title = {Mathematical Models of Overparameterized Neural Networks},
  year={2021},
  
}
```



## Contact
If you meet any problem in this repo, please describe them and contact:

Hanze Dong: A [AT] B, where A=hdongaj, B=ust.hk.

