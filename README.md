
# Variational and Wasserstein Image Autoencodes 

The purpose of this repository is to compare two types of regularized autoencoders: Variational Autoencoder (VAE) and Wasserstein Autoencoder (WAE). 

Regularized autoencoders add an extra loss that enforces the latent distribution to be closer to the given latent prior (e.q. standard normal distribution). This results in easier sampling, stable training and other benefits. The main differences between VAE and WAE are 1) how we measure the discrepancy from our prior and 2) whether we need a stochastic encoder.

VAEs are the most popular type of regularized autoencoders. On the other hand, WAE may be a better choice for certain problems, where Wasserstein distance is more suitable. Additionally, WAE-MMD variant is slightly easier to implement as there's no need to have an encoder that behaves differently during the training and inference. 

Note that: all models here are implemented from scratch in Tensorflow - except for MMD loss, which is based on the original WAE implementation: 
https://github.com/tolstikhin/wae . The code is based on my implementation from 2021 refactored to be compatible with TF 2.13.  

## Overview 

This readme is separated into the following parts: 
1) Variational Autoencoders - loss and training algorithm.
2) Wasserstein Autoencoders - loss and training algorithm.
3) Setup instructions (work in progress).
4) Experiments - VAE and WAE compared on MNIST. 
5) References to related papers and repositories. 

Unlike most of literature on regularized autoencoders, I'll start with the final training algorithms and then briefly explain the theory behind them. 

## Variational Autoencoders 

<!-- add an images from the paper -->

Assuming we have a vanilla autoencoder, we can turn it into VAE with two modifications: 1) make the encoder stochastic and 2) add an extra regularization.

#### Stochastic encoder

We want the output of VAE encoder to be a random variable $z \sim N(\mu_x, \sigma_x^2 I)$, where both $\mu_x$ and $\sigma_x^2$ depend on the input image $x$. The easiest way to implement VAE encoder is to add an extra output to the encoder neural network. Both outputs can be represented with a single feature vector. Two outputs are then interpreted as: 
- $\mu_x$: mean vector  
- $\log(\sigma_x)$: log variance vector  

The final VAE encoder output is: 
- $z=\mu_x+\sigma_x\epsilon$, where $\epsilon \sim N(0, I)$ 

Note that for inference with a trained model, we usually make the encoder deterministic by only using the mean value $z=\mu_x$. 

#### Latent Regularization

In order to get latent distribution that's close to our prior, we add an additional loss term. Our new loss function will have form: $L = L_{rec} + L_{reg}$, where $L_{rec}$ is the same reconstruction loss (MSE) as used in vanilla autoencoders. 

The regularization loss $L_{reg}$ is the following KL-divergence between our output distribution $N(\mu,\sigma^2I)$ and prior distribution: 
<!--
$$L_{reg} = D_{KL}( N(\mu, \sigma ^2 I), N(0, I))$$
$$=\frac{1}{2} \sum_{i=0}^k \left(\sigma_i^2 + \mu_i^2 - 2 \log(\sigma_i) - 1\right)$$
-->

$$L_{reg} = D_{KL}( N(\mu, \sigma ^2 I), N(0, I))=\frac{1}{2} \sum_{i=0}^k \left(\sigma_i^2 + \mu_i^2 - 2 \log(\sigma_i) - 1\right)$$

Minimization of the KL-divergence leads to approximately normal distribution of our latent vectors after training. The regularization loss is easy to compute as the encoder neural network outputs $\mu$ and $\log(\sigma)$. 

#### VAE Training Algorithm

When we combine the two modifications, we get the following training algorithm.

VAE Training Step:
 1. Sample image x from the given training dataset and random noise vector $\epsilon \sim N(0, I)$
 2. Evaluate encoder neural network: $\mu,log(\sigma)=\mathrm{Encoder}(x)$
 3. This gives us the latent vector: $z = \mu + \sigma * \epsilon$
 5. Evaluate decoder neural network to get reconstructed image: $\hat x = \mathrm{Decoder}(z)$
 6. Compute reconstruction error: $L_{rec} = \mathrm{MSE}(x, \hat x)$
 7. Compute regularization error: $L_{reg} = \frac{1}{2} \sum { \left(\sigma_i^2 + \mu_i^2 - 2 \log(\sigma_i) - 1\right) }$  
 8. Update encoder and decoder parameters. 

Note that in practice, we usually sample a batch of images instead of a single image.  

### VAE - More Details

In the following text, I'll briefly explain motivation behind VAE and how to derive the training algorithm. 

#### Problem Statement

We start with the following generative model, where we see the given image as a random variable X:

$$ P(X; \theta) = \int {P(X | z; \theta) } P(z) dz $$

This tells us how likely is the image given the model with parameters $\theta$. On the right side, we add conditioning on the latent vector $z$ with it's prior distribution $P(z)$. We can see $P(X | z; \theta)$ as a decoder, where $\theta$ are it's weights. Our goal will be to find $\theta$ that maximizes this probability over all observed data ([Maximum_likelihood_estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)).

<!-- 
This model corresponds to sampling latent vectors from our prior (e.q. normal distribution) and applying a decoder with distribution P(X | z; \theta).     
Ideally, P(X; \theta) should be high for images similar to our training dataset and close to zero for images that are not similar (e.q. random noise).  
-->

<!--
Let's consider the following generative model:
$$ P(X; \theta) = \int {P(X | z; \theta) } P(z) dz $$
Where $X$ is a random image, $z$ is latent vector with prior distribution $P(z)$ and $\theta$ model parameters.
This gives us the probability of an image $X$ given the generative model with parameters $\theta$. 
-->

#### Approximate Inference (Encoder)

Maximum likelihood for latent variable models such as $P(X; \theta)$ is not generally tractable. The problem is integration over all possible latent vectors $z$ to compute $P(X; \theta)$. Instead of directly optimizing $P(X; \theta)$, we will optimize it's approximation. 

<!-- 
Optimizing $P(X; \theta)$ is not generally tractable. The problem is that to compute $P(X; \theta)$, we would need to compute an integral over all possible values of latent vector $z$. 
-->

A naive solution to estimate the integral would be sampling random vectors $z$. The problem here is that for most latent vectors, $P(X | z)$ will be close to zero. Thus, a large number of samples would be required, intractable for larger latent dimensions.

A better solution is sampling only $z$ that likely produced the given image $X$. This way $P(X | z)$ will be larger and a smaller number of $z$ samples required. 
In order to find such $z$, we'll train a stochastic encoder that given an input image $X$ generates latent vector $z$ with high $P(X | z)$. We define $Q(z| X )$ as the distribution of encoder outputs given the image X.  

<!-- 
We define a stochastic encoder that takes input image $X$ and generates latent vector from a distribution  $Q(z| X )$, i.e. that depends on the given image. This will allow us to generate z with high $P(X | z)$. 

--> 

#### Evidence Lower Bound 

Instead of maximul likelihood estimate approach that would maximize $\log P(X)$, we will maximize it's lower bound, which is easier to compute. 
Evidence Lower Bound (ELBO) is defined as follows:

$$ 
L_{ELBO} = \log P(X) - 
D_{KL} (Q(z | X) || P(z | X)) 
$$

The first term is log-likelihood we want to maximize, the second term is KL-divergence that pushes our encoder distribution closer to our prior. 
Instead of computing ELBO directly as defined above, we use the following form:
$$\log P(X) - D_{KL} (Q(z | X) || P(z | X)) = \\ E_{z \sim Q}[ log P(X | z)]- D_{KL} (Q(z | X) || P(z))$$


<!--We will assume $N(0,I)$ prior, that's most commonly used. --> 

The right side of the equation is something we can compute during the training. If we assume $P(X | z)$ having a normal distribution, $log P(X | z)$ leads to MSE reconstruction loss. The second term $D_{KL} (Q(z | X) || P(z))$ is the latent regularization term as described above.  Therefore, by maximizing ELBO, we 1) minimize reconstruction loss and 2) minimize distance from our prior distribution $P(z)$. 

#### Reparameterization trick

The last problem, we need to solve is to move all non-determinism into our input and make our model deterministic in order to be able to compute gradients. 

This can be done by expresing z as: $z_{x, \epsilon} = \mu_x + \sigma_x * \epsilon$, where $\epsilon \sim N(0, I)$

<!-- We start with sampling random image X and noise $\epsilon \sim N(0, I)$. After that the rest of the computation is deterministic. 
We exppress z as: 
$$z_{x, \epsilon} = \mu_x + \sigma_x * \epsilon$$

-->

Now, we can rewrite our ELBO loss in the final form, which gives us our VAE training algorithm:

$$L_{ELBO} = E_{X}[ E_z [log P(X | z)] - D_{KL} (Q(z | X) || P(z))  ]$$
. 

## Wasserstein Autoencoders 

Wasserstein Autoencoders (WAE) use a different approach to regularize latent space. Similarly to VAE, we are given a prior distribution (usually normal dist.), but KL-divergence is replaced by Wasserstein distance and encoder is not required to be stochastic. There're also two types of Wasserstein Autoencoders WAE-MMD and WAE-GAN with different training algorithms. 


### Wasserstein Distance

The intuition behind Wasserstein distance (also known as Earth Mover’s distance) is the minimum energy cost required to move a pile of dirt in the shape of one probability distribution into the shape of the other distribution. 

The distance between two probability distributions $P_X$ and $P_G$ is based on the optimal transport problem, which is defined as follows: First, we consider a distribution $\Gamma$, which is joint distribution of $P_X$ and $P_G$ (i.e. it's marginals are $P_X$ and $P_G$) and a cost function $c(x,y)$ that gives us the distance between two samples. We define the overall mean as:    
$$C(\Gamma)=\mathbb{E}_{(X,Y) \sim \Gamma} [c(X, Y)] $$

From all possible joint distributions of $P_X$ and $P_G$, we want to find the one that matches $P_X$ and $P_G$ as closely as possible.
$$W_c(P_X, P_G)=\mathrm{inf}_\Gamma C(\Gamma) $$

We can see that, when $P_X=P_G$, the join distribution $\Gamma$ that minimizes the cost is the one that has X=Y. In this case the cost will be zero. When $c(x,y) = |x-y|$, $W_c$ corresponds to the example with moving pile of dirt. 

Finally, p-Wasserstein distance is defined as $W_p(P_X, P_G) = (W_c(P_X, P_G))^{1/p}$, where $c(x, y) = d^p(x, y)$

<!-- $$W_c(P_X, P_G)=\mathrm{inf}_{\Gamma \in P}{\mathop{\mathbb{E}}_{(X,Y) \sim \Gamma} [c(X, Y)]}$$ -->

We can compare Wasserstein and KL divergence on a simple example of two pairs of probability distributions. In this example both pairs have the same KL-divergence, but different Wasserstein distance. On the left, we have two distributions with high Wasserstein distance (to transform the first distribution into second we would need to move a significant amount of probability mass from "1" to "5"). On the right, two distributions have small Wasserstein distance (as moving probability mass from "1" to "2" requires less effort). 

<center>
<img src="images/Wasserstein-vs-KL.png" width="40%" style="align: center">
</center>

Wasserstein distance is a weaker measure, which means it's more "sensitive" to differences between two probability distributions. Depending on the application, Wasserstein distance can be better choice than KL and can lead to more informed gradients. 

### Two Types of Wasserstein Autoencoders

There're two most common ways to train Wasserstein: WAE-MMD and WAE-GAN. Each uses a different approach to estimate Wasserstein distance. WAE-MMD has more stable training (like VAE), WAE-GAN may produce more visually appealing images. 

#### WAE-GAN

WAE-GAN uses adversarial training. This requires an additional neural network - latent discriminator that's trained together with encoder and decoder. The discriminator takes latent code z and predicts whether it's an encoded training image or sampled from our prior distribution. It usually has the same number of layers as encoder and decoder. 

WAE-GAN usually results in higher perceptual quality of generated images, but the downsides are less stable training and the need to add an extra neural network. 

<!-- ![alt text](images/wae-gan.png) -->

#### WAE-MMD 

WAE-MMD is easer to implement as it only adds an extra penalty to our loss function - Maximum Mean Discrepancy (MMD). MMD between the latent and prior distribution can be estimated from the produced latent codes (encoded training images) and random vectors sampled from our prior. 

WAE-MMD is suatable for higher dimensional latent spaces, has stable training (as VAE) and it's easy to implement - no need to add a discriminator or have stochastic encoder. On the other hand, quality of sampled images might be slightly lower compared to WAE-GAN.

<!-- 
TODO explain more details
The usual choise for our kernel is a RBF kernel. (formula)
-->

#### Training 

This part is not finished yet - More details will be added later. 

<img src="images/wae-gan-mmd.png" width="80%">

<!-- 
<center>
<img src="images/wae-gan.png" width="45%">
<img src="images/wae-mmd.png" width="45%">
</center>


- two allgorithms 
- compare all methods also with VAE
- theory (briefly, read more sources  )
   - distance
   - overlapping gaussians 
   - duality + gan training
   - MMD loss explain briefly in more details  
- literature 
Sources 

https://kowshikchilamkurthy.medium.com/wasserstein-distance-contraction-mapping-and-modern-rl-theory-93ef740ae867

 
 -->

# Project Structure

In this part I'm going to explain imlepentation of two models. 

Work in progress ... 


## Setup

```
git clone ...
cd image-autoencoders
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

```

TODO Docker setup

Note that this project is based on my older code using an older version of tensorflow. There might be a better way to implement certain parts now. 


# Experiments

In the following experiemnts I compare VAE, WAE and Vanilla autoencoder with different latent dimensions. The models are compared on MNIST dataset from three main aspects: reconstruction, sampling and interpolations. All results are shown on the test set that is not used for training. 


## Stacking Autoencoders

Instead of autoencoding images directly into latent vectors, we use two stage approach: 1) train a fully convolutional autoencoder (convAE) to encode the original images into lower resolution images and 2) train an another autoencoder (stackedAE) to encode smaller images into latent vectors.  

### Motivation

The main advantage of stacking autoencoders is faster
training. This is specially significant when training multiple models with different latent dimension and type of latent regularization. 

We only need to train convAE once, which is the only compute intentive training that requires a GPU. After than stackedAE can be trained in significantly shother time on CPU.   

This way we can train 18 different models (with different latent dimension and regularization type) in just few hours. It also allows us to tune hyper-parameters in a shorter time. 

<!-- TODO visualize the stacking (img from wiki?) -->


### Convolutional Autoencoder

The fully convolutional autoencoder (convAE) is trained for 100k steps on MNIST dataset. Images 28x28 are encoded into 7x7x10 low resolution tensors (latent images). 

From sample test reconstructions, we can see that the reconstruction here is almost perfect.  


![alt text](images/res1-convae.png)

After training, we encoder all MNIST images and all further training are done on latent 7x7x10 images. 

### Stacked (Latent) Autoencoder 

Stacked Autoencoder has a simple architecture - both encoder and decoder consist of three fully connected layers. There're no convolutional layers. 

In the following experiments, we compare three different types of latent regularization:
Variational autoencoder (VAE), Wasserstein autoencoder (WAE) and classical 
unregularized vanilla autoencoder. 

## Reconstruction

### Mean Reconstruction Error

First, we compare reconstruction error for all types of models and multiple latent dimensions. The reconstruction error is an average test mean squared error. 

<center>
<img src="images/res2-reconstruction.png" width="60%">
</center>

We can see that Vanilla autoencoder has the lowest error. This is expected as it only reconstruction error is minimized during the training.

For most latent dimensions, VAE has lower reconstruction eror, which might be caused by setting too strong regularization weight for WAE. (A better comparison would be plotting 
multiple WAE models with different regularization strenght.)

### Reconstructions on a Small Sample 

Let's compare reconstructed images using different model types. 

<center>
<img src="images/res2-sample-dim2.png" width="80%">
<img src="images/res2-sample-dim3.png" width="80%">
<img src="images/res2-sample-dim8.png" width="80%">
<img src="images/res2-sample-dim21.png" width="80%">
</center>

We can see that models with smaller latent dimension 
do not always reconstruct digits correctly. This gets better with more latent parameters. 

Note that: lower reconstruction error does not always lead to better models. Models with higher latent dimension may fit unwanted noise. 

## Sampling

Another important property of autoencoders is being able to generate random images - in our case images of digits. 

Both VAE and WAE are trained to have approximately normal distribution of the latent space. To sample random images, we can sample z from $N(0, I)$ and apply the model encoder. 

### Sample Generated Images

Unlinke regularized autoencoders, sampling using vanilla autoencoder is more difficult as the latent distribution is not restricted. 

![alt text](images/res3-random-sample-dim2.png)
![alt text](images/res3-random-sample-dim3.png)
![alt text](images/res3-random-sample-dim5.png)
![alt text](images/res3-random-sample-dim8.png)
![alt text](images/res3-random-sample-dim21.png)


We can see that the most of generated images using vanilla AE are either noisy or just an empty image as the sampled latent vectors do not correspond to any meaningful images. 

Quality of the generated images for VAE and WAE seem comparable. 

Small latent dimensions usually lead to more common / average images. With increasing dimensionality, AE can generate more uncommon shapes (see the results for dim=5,8). When the latent dimension is too high, he sampling becomes harder as most of the latent space 
does not represent anything meaningful due to curse of dimensionality (https://en.wikipedia.org/wiki/Curse_of_dimensionality). 

### Latent Distribution

To get a better understanding of generative capabilities, we can plot latent distributions. 

We encode the whole test set using different models for latent dimensions 2 and 3. 

![alt text](images/res4-zdistr-dim2.png)

In case of three dimensions, we plot all pairs of axis. For comparison we also plot a sample from normal distibution. 

![alt text](images/res4-zdistr-dim3.png)


We can see that the latent distribution of VAE and WAE look more similar to the sample from normal distribution as expected.

## Interpolation

One the most important properties of autoencoders is ability to interpolate between two images. 

In order to interpolate two images, we encode both of them into latent vectors, interpolate latent vectors and for each interpolated latent vectors generate an intermediate image. 


### How to Compare Interpolations

For MNIST digits, it's not obvious what should be wanted behaviour here. 
 * Do we prefer all interpolated images to be valid digits?
 * Or would we prefer having smooth transition between them? 

Example: 
 * Say we want to interpolate between "1" and "0". Is it preferable to have the interpolated image that's a valid "6" digit or something that's look more like something between "1" and "0"?

Depending on the application we might have differet preferences. This behaviour can be influenced by our choice of regularization and the number of latent parameters. 

### Sample Interpolations

In the following, we plot two randomly selected mnist digits  "7" (left) and "2" (right). All images between them are interpolations using VAE, WAE and Vanilla autoencoder.   

Generally, with smaller latent dimension, it might be difficult to avoid getting different digits when interpolaring between two digits. Notice that when interpolating between "7" and "2", we often get "3". 

![alt text](images/res5-interp-dim2.png)
![alt text](images/res5-interp-dim3.png)

 In higher dimensions, interpolations are
usually smooth transition from one number to the another one. 

![alt text](images/res5-interp-dim5.png)
![alt text](images/res5-interp-dim8.png)
![alt text](images/res5-interp-dim13.png)

When the latent dimension is too high. We have a similar problem as with random sampling - "holes" in the latent space that does not correspond to any images similar to our dataset. 

![alt text](images/res5-interp-dim21.png)


## Final Thoughts

### Latent Dimension

There's a trade-off between good reconstruction and sampling capabilities. Depending on the application with might prefer smaller or larger latent dimension. 

With higher latent dimension, we usually get near perfect reconstruction, but almost all generated samples will be  noisy as only a tiny fraction of our latent space with encode something meaningful.    

When using small latent dimension, most of random samples will corespond to images similar to our training dataset. On the other hand, we will not be able to represent all images from our dataset with too few latent parameters. Usually, only most common images will be reconstructed correctly. Sometimes this can be wanted behavior and can help us to remove noise.  

### VAE vs WAE

When it comes to VAE vs WAE. From the results, it's not clear which one leads to better results. From practical perspective, WAE is easier to implement as it does not require stochastic encoder, only an extra term in the loss function is needed. 

## References 

1. [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
2. [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558)
3. [Wasserstein Auto-Encoders (original repository)](https://github.com/tolstikhin/wae)

