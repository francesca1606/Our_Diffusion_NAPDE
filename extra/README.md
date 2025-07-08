### EXTRA - DUAL BRANCH

In this folder you can find code to implement a different model than the one trained used in the end but that we believe that with a more dedicated and thorough treatment could lead to great results. 

The model in question is a dual encoding branches - single decoding branch `UNet` with cross attention layers. The main idea is to the two encoding branches of the network the high-passed and low-passed part of the input noise hoping to achieve a better reconstruction of the signal at low frequencies which seems the part of the spectrum where the model struggles the most. 

We think it can be implemented using a classical formulation by filtering the noise at time `t` or also in a conditional fashion by giving to one branch a numerically simulated signal and to the other the noise. 

Inspiration for this was taken from [Dual-branch interactive cross-frequency attention network for deep feature
learning](https://www.sciencedirect.com/science/article/abs/pii/S0957417424012727) and [DI‑UNet: dual‑branch interactive U‑Net for skin cancer image
segmentation](https://link.springer.com/article/10.1007/s00432-023-05319-4)