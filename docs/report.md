# Table of Contents
* [Abstract](#Abstract)
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Multimedia forgery threats people's lives from lots of perspectives in contemporary society. It is necessary to derive new and advanced fake video detection models to deal with learning-based video falsification techniques, such as deepfake. We propose an audio-visual based multimodal deepfake detection framework, combining contrastive loss and cross-entropy loss. We further add an LSTM layer to the vanilla framework and test our models on DFTIMIT and DFDC datasets. Our methods achieved perfect performance on DFTIMIT and comparable results on DFDC. We also analyzed the effects to the model performance using different loss combinations.

# 1. Introduction
## 1.1 Motivation & Objective
Multimedia forgeries threaten data and identity security not only in ordinary people's lives but also in important fields such as politics and military. Previously, classical computer vision techniques were sufficient to detect manually manipulated media. However, recent progress in deep learning methods eases the falsification of media content, one of which is Deepfake that uses artificial intelligence to swap people’s faces in videos and successfully survives classical detection techniques. Thus, we aim to derive new learning-based methods to detect fake videos created by Deepfake techniques.
## 1.2 State of the Art & Its Limitations
Previous deepfake detection work could be divided into three categories: exploiting intra-frame visual artifacts, exploiting inter-frame inconsistencies, and exploiting multimodal features. Intra-frame visual artifacts include unnatural shapes of eyes or mouths, head poses, eye blinkings, etc. while inter-frame inconsistencies include abnormal movements of the contour and so on. These two types of features could be exploited to distinguish between real and fake videos using unimodal methods.

However, unimodal methods have the following limitations. Most unimodal techniques only have the performance of around 60% on the famous DFTIMIT and DFDC datasets using the AUC metric. Some features should be able to be utilized together to detect deepfake videos in a complementary manner. Additionally, they could not detect deepfake videos with audio manipulated.

Meanwhile, multimodal methods leveraging multiple visual features or audio-visual features have been derived and proved to have better performance. In general, methods based on audio-visual features tend to exploit features more complementarily and efficiently and achieve better performance than unimodal methods, especially on the challenging DFDC dataset.
## 1.3 Transition from the midterm proposal
At first, we planned to design and implement a multimodal framework based on the paper Emotions don’t Lie[5]. We’d like to add lip-speech synchronization detection to the proposed framework that measures the similarity of visual emotion and audio emotion between real and fake videos. The basic idea is that visual and audio emotions inferred from real videos would be close to each other in some space while those inferred from fake videos would be far away. However, we ran into several problems:
* The first one is that the learning framework is a double Siamese network that requires the inputs of a real and fake video pair. This might cause the result that the model learns how the fake video was generated rather than some crucial features. As a result, the classifier might not generalize well.
* The second problem is that the framework is very complicated. Not only it is hard to implement, but we think it’s computationally consuming and the training would be hard to converge. So we decided to change our approach.
## 1.4 Rationale
We think we do not have to learn emotional audio-visual coherence explicitly and we assume visual and audio features extracted from real videos would be close to each other in some space while those extracted from fake videos would be in a distance. Chung and Zisserman et al.[7] employed contrastive loss to deal with lip-sync issues. We borrowed the concept of contrastive loss and wanted to use it to measure the similarity of visual and audio features.

In the work of Chugh et al.[8], their experimental results showed that the imposition of cross-entropy loss to contrastive loss would facilitate learning discriminative features.

To extract visual features, we exploited 3D-CNNs similar to those in [9]. We extracted 13 audio MFCC[10] features using ResNet architectures.
## 1.5 Novelty
Based on the inspirations above, we proposed an audio-visual based multimodal architecture that measures the similarity between the audio modal and the visual modal, and used it to distinguish between real and fake videos. We Investigated how different combinations of contrastive loss and cross-entropy loss would affect the performance of the classification. Also, we combined LSTM with the audio-visual based architecture and evaluated the results. Moreover, we tested the performance of different frameworks on similar sizes of training samples on DFTIMIT and DFDC datasets.
## 1.6 Potential Impact
* Might inspire framework designs on audio-visual based models in this area.
* Be able to discern more general and advanced deepfake videos.
* Provide insight on how LSTM would influence the performance of the similarity-measuring deepfake detection system.
* Bring a deeper understanding of the two famous audio-visual based deepfake detection datasets: DFTIMIT and DFDC.
## 1.7 Challenges: What are the challenges and risks?
* Training models to detect fake videos requires a large quantity of computing.
* Appropriate network architectures have to be designed to effectively combine methods using different modalities.
* Training and testing samples should have been well selected and pre-processed.
## 1.8 Requirements for Success
* Abilities to recognize useful and usable features in model training.
* Skills to design, implement, train, analyze, and tune the deep neural networks.
* Efficient computing resources, such as GPU and Google Colab.
## 1.9 Metrics of Success
We plan to apply the Area Under Curve (AUC) metric to our models on benchmark datasets and compare results with those from previous work.

# 2. Related Work
## 2.1 Unimodal methods:
* Intra-frame visual artifacts: 
Li et al.[1] proposed to use DNN to detect artifacts observed during the face warping step of the generation algorithms; Yang et al.[2] used the discontinuity of head pose in synthetic videos as a basis for recognition
* Inter-frame inconsistencies:
Guera and Delp et al.[3] and Sabir et al.[4] both found that deepfake videos contain intra-frame consistencies and combined CNN and LSTM to detect them using different architectures.
## 2.2 Multimodal methods:
* Mittal et al.[5] exploit the congruence between facial and verbal emotions by constructing a double Siamese neural network consisting of extracted expression features and expression vectors, and extracted vocal features and vocal emotion vectors.
* Hosler et al.[6] leverage LSTM networks that predict emotion from audio and video Low-Level Descriptors (LLDs) and classify videos as authentic or deepfakes through an additional supervised classifier trained on the predicted emotion.
# 3. Technical Approach
## 3.1 Dataset
Since the creation of deepfake techniques, a series of datasets have been generated to train and test deepfake detection models. Because our model has to extract features both from visual and audio information, it is necessary to select visual-audio based datasets. Among them, two famous ones are DeepfakeTIMIT (DFTIMIT[11]) and DFDC[12].
### 3.1.1 DFTIMIT
DFTIMIT dataset contains in total 320 fake videos of 32 different people. Each of the subjects has 10 videos. Each video in DFTIMIT is manipulated by operating face swapping from real videos in VIDTIMIT[14] database using FS-GAN, an open-source GAN-based method. Similarly, VIDTIMIT contains in total 320 real videos of 32 different people. Only the visual channel has been manipulated while the audio channel remains authentic. In addition, they used two sizes of FS-GAN models, resulting two series of fake videos: High Quality (HQ) and Low Quality (LQ), respectively. We found that the HQ videos have more obvious facial details while the LQ videosl are more blurred. So we decided to involve the HQ results to the training set. Each video is of 512 × 384 resolution with a 25 fps frame rate, and of ≈ 4𝑠 duration.
### 3.1.2 DFDC
In DFDC dataset, there are over 100,000 videos (real and fake videos both included). The details of the manipulations have not been disclosed. The manipulations exist in either the audio channel or the visual channel or both. The videos are of ≈ 10𝑠 durations, each with an fps of 30. So there are ≈ 300 frames per video.
### 3.1.3 Principles of selecting data
Our method requires the dataset to have both audio and visual channels. The reason why we chose these two datasets is that they are the only two datasets that have both audio and visual information among common deepfake datasets.
In DFTIMIT, all camera angles are frontal. We selected 3 out of 32 subjects as the test set, which ensures that our method is not learning face identities but fake features.
In DFDC, due to the limitation of computing equipment, we only selected training data of almost the same size as the DFTIMIT dataset. Each folder in DFDC consists of videos from a single subject. We selected subjects from different folders, covering as many identities as possible. We also selected videos with manipulations on different channels and different fake effects. After that, we removed the incomplete face data. Finally, we used 623 videos for training, and 62 for testing.
## 3.2. Architecture
### 3.2.1 The Vanilla Multimodal Framework
The flow chart of our vanilla multimodal framework could be seen in Figure 1 below.
<p align = "center">
<img width="779" alt="截屏2022-06-18 17 15 12" src="https://user-images.githubusercontent.com/105074735/174460817-46a24eb5-a119-419a-b4e1-8850d68dd9f0.png">
</p>
<p align = "center">
Figure 1. The Vanilla Multimodal Framework
</p>
The format of the input videos is mp4 before the pre-processing step. We extract visual and audio information from the videos and cut each video into 1-second segments using the ffmpeg toolkit. Then, we use the S3FD model to crop faces and python-speech-features to get 13 MFCC features. Each visual and audio segment would be fed into the visual stream network and the audio stream network, respectively. In the visual stream, the input is a sequence of visual frames with spatiotemporal features in it. It goes through the 3D-CNNs network and the output is a 1024-dimensional feature vector. In the audio stream, the input is a sequence of MFCC feature maps. It goes through the ResNet network and the output is also a 1024-dimensional feature vector. Then, the contrastive loss L1 (1) is computed by measuring the distance between the two feature vectors. Moreover, two cross-entropy losses L2 (3) and L3 (4) are computed and the final loss L (5) would be the weighted sum of the three losses. In our experiment, we chose the weights all to be 1.
<p align = "center">
<img width="536" alt="截屏2022-06-18 18 44 52" src="https://user-images.githubusercontent.com/105074735/174462434-cf132c60-57a7-42f6-b3f4-1addf006c058.png">
</p>
<p align = "center">
<img width="544" alt="截屏2022-06-18 18 45 27" src="https://user-images.githubusercontent.com/105074735/174462446-be98143f-0954-40e7-b6db-8d9cef2c6d77.png">
</p>
<p align = "center">
<img width="541" alt="截屏2022-06-18 18 45 52" src="https://user-images.githubusercontent.com/105074735/174462454-a56730da-3f68-4ae2-90cf-5d75bb3065c8.png">
</p>
<p align = "center">
<img width="537" alt="截屏2022-06-18 18 46 13" src="https://user-images.githubusercontent.com/105074735/174462461-f4f02b7b-bdcf-40e1-b6f3-efc742eb0725.png">
</p>
<p align = "center">
<img width="532" alt="截屏2022-06-18 18 46 33" src="https://user-images.githubusercontent.com/105074735/174462472-57eb121e-3554-4483-99cb-86dbac77dfc7.png">
</p>
In the training process, we calculated the loss and did the back-propagation using PyTorch. In the predicting process, we calculated the distance between visual and audio feature vectors and averaged the distances among all the segments of a single video to get the dissimilarity score of a single video. We computed the dissimilarity score for both real and fake videos of the training set, and the midpoint between the median values for the real and fake videos is used as a threshold value. If the dissimilarity score of a testing video is bigger than the threshold, it would be classified as fake. Otherwise, it’s real.

### 3.2.2 The Multimodal Framework with LSTM
In addition to the vanilla network, we notice that LSTM is useful with respect to detecting temporal inconsistencies in deepfake videos. We’d like to combine LSTM with our current architecture. The intuition is that we suppose the coherence of the segments in a single video would make the dissimilarity score more distinguishable between real and fake videos. Thus, we designed a new architecture, adding an LSTM layer.

The flow chart of our multimodal framework with LSTM could be seen in Figure 2 below.
<p align = "center">
<img width="786" alt="截屏2022-06-18 18 52 17" src="https://user-images.githubusercontent.com/105074735/174462619-eaeb38e4-2e70-4dd5-a020-260ea68a55db.png">
</p>
<p align = "center">
Figure 2. The Multimodal Framework with LSTM
</p>

## 3.3 Platform
During pre-processing, we used ffmpeg to cut each video into 1-second segments. Then we used S3FD model to crop faces and python_speech_features to get the MFCC features of each segment.

During training and testing, we used PyTorch to build our network and implement it.

All of our experiments were run on Google Colab using one GPU.
# 4. Evaluation and Results
Our experiments were implemented logically from three aspects: The first is the performance of our methods on different datasets. The second is for the same dataset, we used different frameworks for comparison. The third is that for each framework, we compared the effects of different loss combinations.
## 4.1 Results of the vanilla framework on DFTIMIT
We applied the vanilla framework to DFTIMIT. We used box plots to visualize the scores of real and fake videos. The visualization could be viewed in Figure 3 below.
<p align = "center">
<img width="739" alt="截屏2022-06-18 19 28 07" src="https://user-images.githubusercontent.com/105074735/174463337-2ebc3443-3de8-4058-9583-6cf347b3c203.png">
</p>
<p align = "center">
Figure 3. Results of the vanilla framework on DFTIMIT
</p>
The greater the difference in score distribution, the better the results. Label 0 represents fake while label 1 represents real. From the results, our method is very effective for DFTIMIT, which can perfectly separate fake and real videos. Additionally, we can see that the score resulting from adding L3 loss is more scattered. We would explain this in the experiments on the DFDC dataset later. In conclusion, our method can fully cope with this simple and small dataset.

## 4.2 Results of the vanilla framework on DFDC
On DFDC, we also performed the same experiment and the results are shown below in Figure 4.
<p align = "center">
<img width="733" alt="截屏2022-06-18 19 30 54" src="https://user-images.githubusercontent.com/105074735/174463393-3870ba63-fd6c-4c38-a484-18e82975238e.png">
</p>
<p align = "center">
Figure 4. Results of the vanilla framework on DFDC
</p>
We initially used the combination of cross-entropy loss of video and audio and contrastive loss. But we found that the result was not very good as you can see in the left plot. We thought this was because there were not many audio modifications in the DFDC data set. So the cross-entropy loss of audio, on the contrary, might affect the judgment of the model. In the case that the audio channel is almost not manipulated in the fake videos, the audios of the real video and the fake video are theoretically of no obvious difference. Therefore, we decided to remove L3 and found that the results were significantly improved.

Afterward, in order to further compare the influence of each loss, we trained the network using only L1. It could be seen that the results were not very good, which showed that the cross-entropy loss did help the model to better judge the authenticity of the video. Only using contrastive loss is not enough.

## 4.3 Results of the framework with LSTM on DFDC
Considering that there should be time consistency between the segments of videos, we added an LSTM layer to the original framework. It is hoped that this can facilitate distinguishing real and fake videos. The results are shown in Figure 5.
<p align = "center">
<img width="524" alt="截屏2022-06-18 20 47 27" src="https://user-images.githubusercontent.com/105074735/174464991-7e706e61-c2e7-4714-85a5-1d9170f78062.png">
</p>
<p align = "center">
Figure 5. Results of the framework with LSTM on DFDC
</p>
The two plots further validate the hypothesis that adding L3 loss affects the results negatively. However, even the results of this LSTM-based model without L3 did not achieve what we wanted. The AUC did not increase but decreased compared with the vanilla model. We also tried to use the output of LSTM for prediction directly, but the results were not much different.

We think the main reasons are as follows. First, we are limited by the computation equipment. We had to reduce the batch size and the input resolution of the face images when running the network with LSTM. We reduced the batch size to 4, and the face image resolution from 224 × 224 to 96 × 96. Secondly, we found that the loss of the LSTM-based model was small during training. This was because the output of the top layer of LSTM was bounded in -1 and 1 due to the use of the Tanh function. Thus, the back propagation might have little impact on the previous CNN network. Gradient vanishing problems were likely to occur, causing the problem of barely learning.

## 4.4 Comparison of our model and other SOTA methods
In Table 1, we summarize the AUC scores of our model and other SOTA deepfake detection methods on DFTIMIT and DFDC datasets. Our method achieved perfect performance on DFTIMIT and comparable results on DFDC.
<p align = "center">
<img width="842" alt="截屏2022-06-18 21 15 19" src="https://user-images.githubusercontent.com/105074735/174465588-1f6ea1dd-c91f-4d6b-a320-fd399fade686.png">
</p>
<p align = "center">
Table 1. Comparison of our model and other SOTA methods
</p>

# 5. Discussion and Conclusions
## 5.1 Advantages
* Achieved perfect performance on DFTIMIT and decent results on DFDC.
* Analyzed the effects to the performance using different loss combinations.
## 5.2 Disadvantages
* The size of training samples is relatively small due to computation limit.
* The performance of the LSTM based network is not good as expected.
## 5.3 Future directions
* Test our models on bigger and more SOTA datasets.
* Improve the architecture of the LSTM based network.
* Try different advanced feature extraction networks and analyze the results.
* Improve the techniques used in data preprocessing, e.g. face alignment, overlapping, etc.
# 6. References
[1] Yuezun Li and Siwei Lyu. 2018. Exposing deepfake videos by detecting face warping artifacts. arXiv preprint arXiv:1811.00656 (2018). https://arxiv.org/abs/1811.00656

[2] Xin Yang, Yuezun Li, and Siwei Lyu. 2019. Exposing deep fakes using inconsistent head poses. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 8261–8265. https://arxiv.org/abs/1811.00661

[3] David Güera and Edward J Delp. 2018. Deepfake video detection using recurrent neural networks. In 2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). IEEE, 1–6.

[4] Ekraam Sabir, Jiaxin Cheng, Ayush Jaiswal, Wael AbdAlmageed, Iacopo Masi, and Prem Natarajan. 2019. Recurrent convolutional strategies for face manipulation detection in videos. Interfaces (GUI) 3 (2019), 1.

[5] Mittal, T., Bhattacharya, U., Chandra, R., Bera, A., & Manocha, D. (2020). Emotions Don't Lie: A Deepfake Detection Method using Audio-Visual Affective Cues. ArXiv, abs/2003.06711. https://arxiv.org/abs/2003.06711

[6] Hosler, Brian, et al. "Do deepfakes feel emotions? A semantic approach to detecting deepfakes via emotional inconsistencies." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[7] Joon Son Chung and Andrew Zisserman. 2017. Out of Time: Automated Lip Sync in the Wild. 251–263. https://doi.org/10.1007/978-3-319-54427-419

[8] Chugh, Komal, et al. "Not made for each other-audio-visual dissonance-based deepfake detection and localization." Proceedings of the 28th ACM international conference on multimedia. 2020.

[9] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. 2017. Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? CoRR abs/1711.09577
(2017). arXiv:1711.09577 http://arxiv.org/abs/1711.09577

[10] Nelson Mogran, Hervé Bourlard, and Hynek Hermansky. 2004. Automatic Speech Recognition: An Auditory Perspective. Springer New York, New York, NY, 309–338. https://doi.org/10.1007/0-387-21575-1_6

[11] Pavel Korshunov and Sébastien Marcel. 2018. Deepfakes: a new threat to face recognition? assessment and detection. arXiv preprint arXiv:1812.08685 (2018).

[12] BrianDolhansky,RussHowes,BenPflaum,NicoleBaram,andCristianCanton Ferrer. 2019. The Deepfake Detection Challenge (DFDC) Preview Dataset. arXiv preprint arXiv:1910.08854 (2019).

[13] GitHub-shaoanlu/faceswap-GAN:Adenoisingautoencoder+adversarial losses and attention mechanisms for face swapping. https://github.com/shaoanlu/ faceswap-GAN. (Accessed on 02/16/2020).

[14] Conrad Sanderson. 2002. The vidtimit database. Technical Report. IDIAP.

[15] H. H. Nguyen, J. Yamagishi, and I. Echizen. 2019. Capsule-forensics: Using Capsule Networks to Detect Forged Images and Videos. In ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2307–2311.

[16] F. Matern, C. Riess, and M. Stamminger. 2019. Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations. In 2019 IEEE Winter Applications of Computer Vision Workshops (WACVW). 83–92.
