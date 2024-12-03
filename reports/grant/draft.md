# Introduction
Musical instruments classification is a Musical Information Retrieval (MIR) task which consists of determining the instruments present in a recording. This topic has been extensively studied in the literature, and for monophonic recordings (containing only one instrument), state-of-the-art models reach often almost $100\%$. However, few articles have addressed the issue of identifying individual instruments of the same type, let alone from the violin familiy specifically.

The goal of this paper is to review and compare different ways of tackling this task, from data collection to data processing. Guidelines regarding recording sessions are discussed and a Long-Time version of MFCCs is introduced.

# Experiment
## Dataset
During the Bilbao Project, thirteen violins were built in order to relate their material and geometrical characteristics with their tonal quality \citep{fritzBilbaoProjectSearching2021}. These violins have been played in 2019 by twenty-three professional violinists, each of them having recorded a scale on each violin and a short musical excerpt on a violin of their choice. The recordings were made under the same conditions in a large rehearsal room at the Bilbao conservatory, keeping the distance between the player and the microphone constant. Our dataset thus consists of $13 \times 23$ scales plus $1 \times 23$ musical excerpts.

[Figure]

Six of those thirteen Bilbao violins (violins number 1, 4, 5, 9, 11 and 13) were brought to the 2024 Villfavard Workshop and were recorded again. They were played freely by four new players in a small room, under rather different conditions than during the 2019 recordings.

[Figure]

## Features
We limit the study to long-term audio descriptors, as we can observe that when violinists try new instruments they often give a global description of the sound they produce.
We compare three audio descriptors : Long-Term Average Spectra (has been used in [] in order to compare the sound quality of violins), Mel Frequency Cepstral Coefficients (have been largely used in speaker [] and instrument [] classification) and Long-Term Cepstral Coefficients (have been introduced in [] in order to tackle violin identification).

## Classification
We compare the results of three popular classification algorithms : K-Nearest Neighbours, Support Vector Machines and Multilayer Perceptron. These three classifiers use different learning strategies and thus will give different results on our data.

K-Nearest Neighbours is a method that finds the closest training points to a new test point and predicts its label from them. Since the prediction is made directly from the training data, this method is non-parametric (or non-generalizing), which is an advantage when the decision boundary is irregular.

Support Vector Machines is a supervised learning method used for classification. It works by finding an optimal hyperplane that maximizes the distance between each class in the training data. This algorithm is computationally expensive but generally has good generalisation properties.

Multilayer perceptron is a supervised learning method that learns a function $f : \mathbb{R}^n \to \mathbb{R}^o$ using training data. To do so, it uses layers of neurons. Each neuron transforms the result from the previous layer by forming a weighted linear summation $w_0 + w_1 x_1 + \dots + w_n x_n$ followed by a non-linear activation function. This method has demonstrated effectiveness in various domains, as the non-linear activation functions can model complex relationships in data while hidden layers can learn hierarchical representations from input data. However, MLPs are prone to overfitting, especially on small datasets.

# Results
[Table : Results using scales as training data and muscial excertps as test data]
[Table : Results using half players as training data and the other half as test data]
[Table : Results using 2019 recordings as training data and 2024 recordings as test data]

# Conclusion
Our results show that MFCCs can effectively distinguish individual violins beyond the variabilities induced by the players, enabling violin recognition from recordings. Future work will focus on correlating MFCCs with perceptual evaluations of violin timbre by musicians, providing insights into how objective metrics align with subjective listening experiences.