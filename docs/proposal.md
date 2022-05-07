# Project Proposal

## 1. Motivation & Objective

Forgeries in media could be harmful in lots of aspects, including politics, military, security, etc. Previously, classical computer vision techniques were enough to deal with media manually manipulated by humans. However, recently advanced deep learning methods have been used to forge or falsify media. For instance, deepfakes, which uses artificial intelligence to swap people’s faces in media, is one of those sophisticated techniques that classical computer vision techniques cannot cope with. Thus, we aim to derive new learning-based methods to detect fake videos created by deepfakes.


## 2. State of the Art & Its Limitations

Most prior work designed models to detect deepfake videos based on two principles: one is that deepfake videos have visual artifacts across frames, and the other is that temporal features may not be consistent after the synthesis process of deepfakes. Moreover, a variety of network architectures have been proved useful and achieved high classification accuracy in previous researches. However, most work just exploited single modality. Since we have several potential features to use to detect deepfake videos, a natural question emerges: can we design multimodal deepfake detection methods to further increase the classification accuracy or to deal with even more advanced deepfakes?


## 3. Novelty & Rationale

We aim to propose a multimodal deepfake method, making use of both visual and speech information in the input video.  Mittal et al. utilized the similarity between face and speech modalities with the help of affective cues and their model worked pretty well on DeepFakeTIMIT and DFDC datasets. Plus, some previous work has shown that lip-speech synchronization is a good feature to use when detecting deepfake videos. Since we believe Mittal’s method and lip-speech synchronization are complementary, we plan to design a proper architecture to effectively combine them together and further improve the classification accuracy compared with Mittal’s work.


## 4. Potential Impact

If the project is successful, one direct impact is that the classification accuracy rate of deepfake videos would be improved to another step. If the accuracy does not outperform previous work, our project will also be meaningful since it provides an example of how to combine different modalities to detect deepfake videos as effectively as possible. It may inspire other researchers and indirectly lead up to better models.
In general, the progress in detecting deepfake videos would benefit lots of areas, including politics, military, and security, etc.


## 5. Challenges

* Training models to detect fake videos requires a large quantity of computing. 
* Appropriate network architectures have to be designed to effectively combine methods using different modalities.


## 6. Requirements for Success

* Abilities to realize which features are more useful and could be put together to train the model.
* Skills to design, implement, and analyze deep learning networks.
* Efficient computing resources, such as GPU.


## 7. Metrics of Success

We plan to apply the Area Under Curve (AUC) metric on benchmark datasets and compare results with those from previous work.


## 8. Execution Plan

Generally speaking, key tasks consist of selecting features to use, designing neural networks, training the model using Pytorch, analyzing the network and the performance, and tuning the network (parameters).

As a team, we would do the project cooperatively. Each team member would participate in each single task. But we claim different emphasis for each member:

* Ruoye Wang: analyze the model and performance
* Jinchen Wu: implement the model using Pytorch and train the model
* Weijian Zhang: determine which features to use and design the architecture


## 9. Related Work

### 9.a. Papers

First, many prior works split the video into frames and study the connection between them, for example, Li et al.[1] proposed to use DNN to detect artifacts observed during the face warping step of the generation algorithms; Yang et al.[2] used the discontinuity of head pose in synthetic videos as a basis for recognition. In addition, different network structures have been tried by many previous works: for example, the capsule network structure proposed by Nguyen et al.[3] and XceptionNet proposed by Rossler et al.[4]. Previous researchers have also noticed that temporal coherence is not enforced effectively in the synthesis process of deepfakes, such as Guera and Delp et al.[5] who found that deepfake videos contain intra-frame consistencies.

Our idea comes from the paper Emotions Don't Lie by Mittal et al.[6], in which the authors exploit the congruence between facial and verbal emotions by constructing a double Siamese neural network consisting of extracted expression features and expression vectors, and extracted vocal features and vocal emotion vectors. However, this article has some shortcomings, such as people deliberately concealing their expressions in some specific cases. Therefore, we decided to reproduce this article and add some other micro-expression features to the network, such as blink patterns[7], to try to improve the accuracy.


### 9.b. Datasets

In the pre-training phase, we use the CMU-MOSEI[8] database to pre-train the Video/Audio Perceived Emotion Embedding, and then DFDC[9] and DF-TIMIT[10] are used because they are new datasets containing both audio and video, which is what we need; and they are large datasets as well as do not contain the model generating fake video’s information.


### 9.c. Software

First we need to use OpenFace[11] and pyAudioAnalysis[12] to get the features of face and voice respectively, after that we will finish the construction and training of the network on Colab, as well as the testing and analysis afterwards.


## 10. References

[1] Yuezun Li and Siwei Lyu. 2018. Exposing deepfake videos by detecting face warping artifacts. arXiv preprint arXiv:1811.00656 (2018). https://arxiv.org/abs/1811.00656

[2] Xin Yang, Yuezun Li, and Siwei Lyu. 2019. Exposing deep fakes using inconsistent head poses. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 8261–8265. https://arxiv.org/abs/1811.00661

[3] Huy H Nguyen, Junichi Yamagishi, and Isao Echizen. 2019. Capsule-forensics: Using capsule networks to detect forged images and videos. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2307–2311. https://arxiv.org/abs/1810.11215

[4] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. 2019. Faceforensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE International Conference on Computer Vision. 1–11. https://arxiv.org/abs/1901.08971

[5] David Güera and Edward J Delp. 2018. Deepfake video detection using recurrent neural networks. In 2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). IEEE, 1–6. https://ieeexplore.ieee.org/document/8639163

[6] Mittal, T., Bhattacharya, U., Chandra, R., Bera, A., & Manocha, D. (2020). Emotions Don't Lie: A Deepfake Detection Method using Audio-Visual Affective Cues. ArXiv, abs/2003.06711. https://arxiv.org/abs/2003.06711

[7] Jung, T., Kim, S., & Kim, K. (2020). DeepVision: Deepfakes Detection Using Human Eye Blinking Pattern. IEEE Access, 8, 83144-83154. https://ieeexplore.ieee.org/document/9072088

[8] CMU-MOSEI http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/

[9] DFDC https://ai.facebook.com/datasets/dfdc/

[10] DF-TIMIT https://www.idiap.ch/en/dataset/deepfaketimit

[11] OpenFace https://cmusatyalab.github.io/openface/

[12] pyAudioAnalysis https://github.com/tyiannak/pyAudioAnalysis


