#let ASAblue = rgb(0, 48, 86)
#let ASAlightblue = rgb(0, 167, 225)

#set page(
  margin: 1in,
  header: line(stroke: 2pt + ASAblue, length: 100%),
  footer: line(stroke: 2pt + ASAblue, length: 100%)
)

#set text(font: "EB Garamond",)

#set heading(numbering: "1.A.1")
#show heading: it => {
  let levels = counter(heading).get()
  let deepest = if levels != () {
    levels.last()
  } else {
    1
  }

  if it.level == 1 {
    show: block
    
    set text(16pt, weight: "bold", fill: ASAblue)
    if it.numbering != none {
      box(width: 0.3in, numbering("1.", deepest))
    }
    h(7pt, weak: true)
    smallcaps(it.body)
  }
  if it.level == 2 {
    show: block
    set text(16pt, weight: "bold", fill: ASAlightblue)
    box(width: 0.3in, numbering("A.", deepest))
    h(7pt, weak: true)
    smallcaps(it.body)
  }
  if it.level == 3 {
    show: block
    set text(1em, weight: "bold", fill: ASAblue)
    box(width: 0.3in, numbering("i.", deepest))
    h(7pt, weak: true)
    it.body
  }
}

#set par(
  first-line-indent: 1em,
  justify: true,
)

= Introduction

Musical instrument classification is a fundamental task in Musical Information Retrieval (MIR), focusing on identifying the types of instruments present in a recording, such as violins, cellos, or pianos. This topic has been extensively studied in the literature, and for monophonic recordings (containing only one instrument), state-of-the-art models often achieve nearly 100% accuracy. 
However, few articles have addressed the issue of identifying individual instruments of the same type, let alone from the violin familiy specifically (for example, identifying violin A versus violin B).
Only a handful of studies (@lukasik2010, @wang2020, @yokoyama2022a) have tackled this challenge, showing that cepstral coefficients provide sufficient discriminative information about individual violin timbre to enable effective classification. @lukasik2010 found that long-term features could lead to a more stable classification. Nevertheless, these studies relied on limited datasets, where not all violins were played by multiple violinists. This raises an important question: do these algorithms recognize the instruments themselves, or simply the unique playing styles of the performers?

This paper aims to address this challenge by reviewing and comparing various approaches to identifying individual violins. In order to achieve this, two datasets were collected, comprising recordings of multiple violinists playing a set of violins. Various features and machine learning algorithms were then analyzed, ultimately providing retrospective guidelines for recording sessions — for example, identifying which musical excerpts are most effective for capturing the timbral differences between violins.
The remainder of this paper is organized as follows: @methodology describes the methodology, including data collection, feature extraction, data exploration, and classification using machine learning. The experimental results are presented and analyzed in @results. Finally, @conclusions provides concluding remarks and outlines potential directions for future work.

= Methodology <methodology>
This section outlines the methodology employed in this study, covering data collection, processing, and the machine learning algorithms applied. 

==  Datasets
Two datasets were collected for this study, each consisting of recordings where a group of violinists played a set of violins. The violins and violinists sets were different in those two experiments. The recordings durations for each player and each violin in both datasets are presented in @f:dataset.

=== Bilbao Dataset
During the Bilbao Project, thirteen violins were built in order to relate their material and geometrical characteristics with their tonal quality @fritz2019. These violins have been played in 2019 by twenty-three professional violinists, each of them having recorded a 3-octave G chromatic scale (as in @f:scale) on each violin and a short musical excerpt on a violin of their choice. Our dataset thus consists of $13 times 23$ scales plus $1 times 23$ musical excerpts. The recordings were made under the same conditions in a large rehearsal room at the Bilbao conservatory, keeping the distance between the player and the microphone (Zoom H4n Pro) constant and using a sample rate of 48kHz.

#figure(
  image("../figures/scale.png"),
  caption: [3-octave G chromatic scale played by the participants of the recording.]
)<f:scale>

Six of those thirteen Bilbao violins (violins number 1, 4, 5, 9, 11 and 13) were brought to the 2024 Villfavard Workshop and were recorded again. They were played freely by four new players, in a small room, under rather different conditions than during the 2019 recordings. Recordings were made using a Zoom H5 with a sample rate of 48kHz.

=== CNSM Dataset
In September 2024, we conducted an experiment at the Paris Conservatoire (CNSMDP) involving thirteen violinists and three violins. The participants were invited to freely explore each instrument before recording a chromatic scale (@f:scale) and selected excerpts from the classical violin repertoire, including pieces by Bach (Allemande), Mozart (Concerto No. 3), Tchaikovsky (Concerto), Sibelius (Concerto), and Glazunov (Concerto). The recordings were made under the same conditions in a large recording studio at the CNSM, keeping the distance between the player and the microphone (a pair of DPA 4006) constant and using a sample rate of 48kHz.

#figure(
  grid(
    columns: (1fr, 1fr),
    image("../figures/class_weights.svg"),
    image("../figures/class_weights_cnsm.svg"),
  ),
  caption: [Recording time available with respect to players and with respect to violins]
)<f:dataset>

==  Features
We limit the study to long-term audio descriptors, as we can observe that when violinists try new instruments they often give a global description of the sound they produce. The following features have been compared for the classification task :

=== Long Time Average Spectra (LTAS)
The Long Time Average Spectra (LTAS) of a recording is obtained by dividing the input signal into overlapping segments, then calculating the windowed DFT of each segment and finally averaging the power of those DFTs. These steps are shown in @f:ltas :

#figure(
  image("../figures/ltas.svg"),
  caption: "LTAS computation steps",
  gap: 0cm,
)<f:ltas>

LTAS has been used in @buen2005 in order to compare the tonal quality of violins. More specifically, the sound of old Italians violins (Stradivari/Guarneri) and modern violins has been compared. The author concludes that differences between these two groups can be shown using LTAS


=== Long-Term Cepstral Coefficients (LTCC)
LTCC have been introduced in @lukasik2010 for Indivudial Instrument Identification. Their calculation is similar to that of MFCCs, except that a Mel-filterbank is not applied and that the final step is given by an Inverse Discrete Fourier Transform. These steps are shown in @f:ltcc :

#figure(  
  image("../figures/ltcc.svg"),
  caption: "LTCC computation steps",
  gap: 0cm,
)<f:ltcc>


=== Mel-Frenquency Cepstral Coefficients (MFCC)

MFCCs are obtained by mapping the frequencies of a spectrum onto a nonlinear mel-scale (a perceptual scale of pitches judged by listeners to be equal in distance from one another), taking the log, and then compute the DCT of the result. Here, instead of calculating the MFCCs on overlapping segments, we use a LTAS as our spectra as we want features with a long-term meaning. These steps are shown in @f:mfcc :

#figure(  
  image("../figures/mfcc.svg"),
  caption: "MFCC computation steps",
  gap: 0cm,
)<f:mfcc>

MFCC are a set of features that has been extensively used for Automatic Speaker Recognition and for Instruments Classification (@eronen2000, @deng2008).

== Classifiers

We compare the results of three popular classification algorithms : K-Nearest Neighbours, Support Vector Machines and Multilayer Perceptron. These three classifiers use different learning strategies and thus will give different results on our data.

=== K-Nearest Neighbors
K-Nearest Neighbours is a method that predicts the label of a new point based on the most frequent label among its K closest neighbors in the feature space. Since the prediction is made directly from the training data, this method is non-parametric (or non-generalizing), which is an advantage when the decision boundary is irregular.

=== Support Vector Machines
Support Vector Machines is a supervised learning method using for classification. It works by finding an optimal hyperplane that maximizes the distance between each class in the training data. This algorithm is computationally expensive but generally has good generalisation properties.

=== Multilayer Perceptron (MLP)
A Multilayer Perceptron is a type of neural network that maps inputs to outputs through multiple layers of interconnected neurons, using learned weights and biases to perform complex transformations. This method has demonstrated effectiveness in various domains, as the non-linear activation functions can model complex relationships in data while hidden layers can learn hierarchical representations from input data. However, MLPs are prone to over-fitting, especially on small datasets.


= Results & Discussions <results>
This section examines the influence of hyperparameters on the classification task and evaluates performance across different dataset subsets. Classifier scores were computed for a grid of hyperparameters. To analyze the effect of each hyperparameter, the mean and standard deviation of the scores were calculated while varying only that hyperparameter and keeping all others constant.

== Dataset hyperparameters

=== Influence of sample rate

// #grid(
//   columns: 2,
//   align: horizon,
//   gutter: 2em,
//   figure(
//     image("../figures/influence_sr.svg"),
//     caption: [Influence of the sampling rate of the dataset on the accuracy of the algorithms]
//   ),
//   par[
//     The results indicate that high sampling rates are not necessary to achieve high accuracy. This suggests that the low-frequency components of the spectra provide sufficient discriminative information for accurate predictions. Consequently, using a relatively low sampling rate (e.g., 8000 Hz) is a more efficient choice, reducing storage requirements and computation time without compromising performance.
//   ]
// )

#figure(
  image("../figures/influence_sr.svg"),
  caption: [Influence of the sampling rate of the dataset on the accuracy of the algorithms]
)<finfluence_sr>

The results in @finfluence_sr indicate that high sampling rates are not necessary to achieve high accuracy. This suggests that the low-frequency components of the spectra provide sufficient discriminative information for accurate predictions. Consequently, using a relatively low sampling rate (e.g., 8000 Hz) is a more efficient choice, reducing storage requirements and computation time without compromising performance.

=== Influence of sample duration
#figure(
  image("../figures/influence_sample_duration.svg"),
  caption: [Influence of the sample duration on the accuracy of the algorithms]
)<f_influence_sample_duration>

Analysis of @f_influence_sample_duration reveals that longer sample durations lead to higher accuracy. This can be attributed to the use of long-term features, which require a wide range of played notes to effectively capture the violins' signature spectra. However, sample durations exceeding 40 seconds do not appear to provide additional improvements in accuracy.

== Features hyperparameters

=== Influence of feature
#figure(
  image("../figures/influence_feature.svg"),
  caption: [Influence of the sampling rate of the dataset on the accuracy of the algorithms]
)<f_influence_feature>

The results in @f_influence_feature indicates that MFCCs and LTAS seem to be better features to distinguish violins from their sound.

=== Influence of the number of coefficients (for MFCCs and LTCCs)
#figure(
  image("../figures/influence_n_coeff.svg"),
  caption: [Influence of the number of coefficients kept in the calcultion of LTCCs and MFCCS]
)<f_influence_n>

In the computation of LTCCs and MFCCs, using fewer cepstral coefficients results in a smoother spectral representation, as only the first coefficients of the spectral transform are retained. The results in @f_influence_n show that a high number of cepstral coefficients is not required to achieve good prediction performance. This indicates that even a smoothed representation of the spectra provides sufficient discriminative information for accurate classification.

== Classifiers hyperparameters

=== Influence of the classifier
#figure(
  image("../figures/influence_clf.svg"),
  caption: [Influence of the sampling rate of the dataset on the accuracy of the algorithms]
)<f_influence_clf>

The results in @f_influence_clf show that all three classification algorithms achieve a similar accuracy, approximately 90%. Since these algorithms have different inherent disadvantages, this consistency suggests that the accuracy obtained is largely independent of the choice of classifier. The results indicate that KNNs and SVMs outperform MLPs. This suggests that the dataset size may be insufficient for MLPs to perform optimally. Additionally, the features appear to be sufficiently well-separated in the feature space, allowing simpler algorithms to achieve better results.

= Conclusions <conclusions>

In the paper, we challeged the identification of individual violins based on their unique timbre, using machine learning techniques. 
By assessing the influence of multiple parameters on the classification accuracy, this study established that long-term MFCCs combined with a simple KNN are powerful enough to tackle this task. The low-frequency content is discriminative enough to allow a sample rate of only 8 kHz, providing that the each sample is long enough (30 seconds or longer).
Future work will focus on correlating MFCCs with perceptual evaluations of violin timbre by musicians, providing insights into how objective metrics align with subjective listening experiences.

#bibliography("references.bib", title: "References")

= Appendix