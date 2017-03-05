
## CNN -> spiketrain
##### Rationale
Recently, making generative models based solely on pre-trained deep convolutional neural nets are capable of inferring complex structures about the images and how we percieve them.
In some cases even, these pretrained image feature extractions (sometimes called annotations) once trained, have been able to exceed state of the art in some computer vision domains (e.g. [Deep Gaze II](https://arxiv.org/abs/1610.01563) using VGG)


##### Resources
- [Using pretrained CNN models for things](pretrained_models_readme.md)


##### Design

| Block 1        | Shape                |
| -------------- | -------------------- | ------------- |
| Input: Img     | (224,244,3)          |               |
| fetch n layers | (14,14,512)          |               |
| concatenate    | (14,14,1536)         |               |
| Output: Tensor | (14,14,1536)         |               |


| Block 2 (conv) | Shape                |
| -------------- | -------------------- | ------------- |
| Conv1 (16,1,1) | (14,14,16)           |               |
| BatchNorm1     | (14,14,16)           |               |
| Conv2 (32,1,1) | (14,14,32)           |               |
| BatchNorm2     | (14,14,32)           |               |
| Conv3 (2,1,1)  | (14,14,2)            |               |
| BatchNorm3     | (14,14,2)            |               |
| Conv4 (1,1,1)  | (14,14,1)            |               |

| Block 3 (fc)   | Shape                |
| -------------- | -------------------- | ------------- |
| Flatten        | (196, )              |               |
| Dense(4096)    | (4096, )             |               |
| Dense(2048)    | (2048, )             |               |
| Dense(n)       | (n, )                |               |

##### References
- DeepGaze II: Reading fixations from deep features trained on object recognition - ([PDF](https://arxiv.org/pdf/1610.01563))
