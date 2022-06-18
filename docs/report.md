# Table of Contents
* [Abstract](#Abstract)
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Provide a brief overview of the project objhectives, approach, and results.

# 1. Introduction
## Motivation & Objective
Multimedia forgeries threaten data and identity security not only in ordinary people's lives but also in important fields such as politics and military. Previously, classical computer vision techniques were sufficient to detect manually manipulated media. However, recent progress in deep learning methods eases the falsification of media content, one of which is Deepfake that uses artificial intelligence to swap people’s faces in videos and successfully survives classical detection techniques. Thus, we aim to derive new learning-based methods to detect fake videos created by Deepfake techniques.
## State of the Art & Its Limitations
Previous deepfake detection work could be divided into three categories: exploiting intra-frame visual artifacts, exploiting inter-frame inconsistencies, and exploiting multimodal features. Intra-frame visual artifacts include unnatural shapes of eyes or mouths, head poses, eye blinkings, etc. while inter-frame inconsistencies include abnormal movements of the contour and so on. These two types of features could be exploited to distinguish between real and fake videos using unimodal methods.

However, unimodal methods have the following limitations. Most unimodal techniques only have the performance of around 60% on the famous DFTIMIT and DFDC datasets using the AUC metric. Some features should be able to be utilized together to detect deepfake videos in a complementary manner. Additionally, they could not detect deepfake videos with audio manipulated.

Meanwhile, multimodal methods leveraging multiple visual features or audio-visual features have been derived and proved to have better performance. In general, methods based on audio-visual features tend to exploit features more complementarily and efficiently and achieve better performance than unimodal methods, especially on the challenging DFDC dataset.
## Transition from the midterm proposal
At first, we planned to design and implement a multimodal framework based on the paper Emotions don’t Lie[5]. We’d like to add lip-speech synchronization detection to the proposed framework that measures the similarity of visual emotion and audio emotion between real and fake videos. The basic idea is that visual and audio emotions inferred from real videos would be close to each other in some space while those inferred from fake videos would be far away. However, we ran into several problems:
* The first one is that the learning framework is a double Siamese network that requires the inputs of a real and fake video pair. This might cause the result that the model learns how the fake video was generated rather than some crucial features. As a result, the classifier might not generalize well.
* The second problem is that the framework is very complicated. Not only it is hard to implement, but we think it’s computationally consuming and the training would be hard to converge. So we decided to change our approach.
## Rationale
We think we do not have to learn emotional audio-visual coherence explicitly and we assume visual and audio features extracted from real videos would be close to each other in some space while those extracted from fake videos would be in a distance. Chung and Zisserman et al.[7] employed contrastive loss to deal with lip-sync issues. We borrowed the concept of contrastive loss and wanted to use it to measure the similarity of visual and audio features.

In the work of Chugh et al.[8], their experimental results showed that the imposition of cross-entropy loss to contrastive loss would facilitate learning discriminative features.

To extract visual features, we exploited 3D-CNNs similar to those in [9]. We extracted 13 audio MFCC[10] features using ResNet architectures.
## Novelty
Based on the inspirations above, we proposed an audio-visual based multimodal architecture that measures the similarity between the audio modal and the visual modal, and used it to distinguish between real and fake videos. We Investigated how different combinations of contrastive loss and cross-entropy loss would affect the performance of the classification. Also, we combined LSTM with the audio-visual based architecture and evaluated the results. Moreover, we tested the performance of different frameworks on similar sizes of training samples on DFTIMIT and DFDC datasets.
## Potential Impact
* Might inspire framework designs on audio-visual based models in this area.
* Be able to discern more general and advanced deepfake videos.
* Provide insight on how LSTM would influence the performance of the similarity-measuring deepfake detection system.
* Bring a deeper understanding of the two famous audio-visual based deepfake detection datasets: DFTIMIT and DFDC.
## Challenges: What are the challenges and risks?
* Training models to detect fake videos requires a large quantity of computing.
* Appropriate network architectures have to be designed to effectively combine methods using different modalities.
* Training and testing samples should have been well selected and pre-processed.
## Requirements for Success
* Abilities to recognize useful and usable features in model training.
* Skills to design, implement, train, analyze, and tune the deep neural networks.
* Efficient computing resources, such as GPU and Google Colab.
## Metrics of Success
We plan to apply the Area Under Curve (AUC) metric to our models on benchmark datasets and compare results with those from previous work.

# 2. Related Work
## Unimodal methods:
* Intra-frame visual artifacts: 
Li et al.[1] proposed to use DNN to detect artifacts observed during the face warping step of the generation algorithms; Yang et al.[2] used the discontinuity of head pose in synthetic videos as a basis for recognition
* Inter-frame inconsistencies:
Guera and Delp et al.[3] and Sabir et al.[4] both found that deepfake videos contain intra-frame consistencies and combined CNN and LSTM to detect them using different architectures.
## Multimodal methods:
* Mittal et al.[5] exploit the congruence between facial and verbal emotions by constructing a double Siamese neural network consisting of extracted expression features and expression vectors, and extracted vocal features and vocal emotion vectors.
* Hosler et al.[6] leverage LSTM networks that predict emotion from audio and video Low-Level Descriptors (LLDs) and classify videos as authentic or deepfakes through an additional supervised classifier trained on the predicted emotion.
# 3. Technical Approach
## Dataset
Since the creation of deepfake techniques, a series of datasets have been generated to train and test deepfake detection models. Because our model has to extract features both from visual and audio information, it is necessary to select visual-audio based datasets. Among them, two famous ones are DeepfakeTIMIT (DFTIMIT[11]) and DFDC[12].
### DFTIMIT

### DFDC

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References
