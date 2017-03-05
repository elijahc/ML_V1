# RNN

##### Resources
- [Predicting sequences of vectors(regression) in Keras using LSTM](http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/)
- [TimeDistributed Wrapper in Keras](https://github.com/fchollet/keras/blob/master/keras/layers/wrappers.py#L45)
### Thoughts moving forward:
---


#### _Attention_
Use "neural machine translation" inspired model with attention that uses an attention mask that demarcates regions of the prior spike activity important particular neuron or future timestep.

##### Resources
- [Attention Models in Keras](https://github.com/fchollet/keras/issues/2067) (Forum)
- [How to add Attention on top of a RNN (Text Classification)](https://github.com/fchollet/keras/issues/4962) (Forum)

##### References
- Show, Attend, Tell: Neural Image Caption Generation with Visual Attention - ([PDF](https://arxiv.org/pdf/1502.03044))
- Neural Machine Translation by Jointly Learning to Align and Translate - ([PDF](https://arxiv.org/pdf/1409.0473))

---


#### _Latent Variable_
Latent, shared, random variable can account for large amount of variance in V1 ([Ecker et al 2014](http://www.cell.com/neuron/abstract/S0896-6273(14)00104-4))

RNN's which have a Latent Random Variable at each time step parameterized by the prior hidden layer (h<sub>t-1</sub>) work better than vanilla RNN's for highly structured sequence prediction ([J .Chung et al 2015](https://arxiv.org/abs/1506.02216))


##### Resources
- [ Building Autoencoders in Keras ](https://blog.keras.io/building-autoencoders-in-keras.html) (Keras Blog)
- [Example VAE implementation in keras](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py) (Github)
- [Theano implementation of VRNN](https://github.com/jych/nips2015_vrnn) from J. Chung et al (Github)
---
