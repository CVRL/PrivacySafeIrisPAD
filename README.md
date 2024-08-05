# Privacy-Safe Iris Presentation Attack Detection #
Official GitHub repository of the paper: Mahsa Mitcheff, Patrick Tinsley, and Adam Czajka, "Privacy-Safe Iris Presentation Attack Detection," [IJCB](https://ijcb2024.ieee-biometrics.org), Buffalo, NY, September 15-18, 2024
<br><br>

![pipiline](https://github.com/CVRL/PrivacySafeIrisPAD/blob/main/pipiline.png)

Overview of the pipeline of privacy-safe, synthetic data-only iris presentation attack detection (PAD) training and validation. TCL and noTCL denote images of irises with and without contact lenses, respectively. After training generative models *Step 1*, we exclusively use synthetically-generated data (mimicking irises both with and without textured contact lenses) to train iris PAD as usual (*Step 3*). The iris matcher is used (in *Step 2*) to exclude synthetic samples that are too close to non-synthetic samples used for generative models training, which prevents the leakage of identity information from the training set into the generated samples. Resulting iris PAD methods are tested on regular (non-synthetic) data composed of bona fide and fake samples (*Step 4*).

## Table of contents
* [Abstract](#abstract)
* [Step1: Training GAN Models on the Authentic Iris Samples](#gan-training)
   * [Weight Acquisition](#gan-weights)
   * [Synthetic Samples Acquisition](#gan-samples)
* [Step2: Image Synthesis and Identity Leakage Mitigation](#gan-synthesizing)
* [Step3: Training of Iris PAD Models with Synthetic Data Only](#pad-tarining)
  * [Environments Requirements](#requirements)
* [Step4: Testing of Iris PAD Models Used in the Paper with Unseen Authentic Data](#pad-testing)
  * [Weight Acquisition](#PAD-weights)
* [Citation](#citation)
* [Acknowledgment](#acknowledgment)

<a name="abstract"/></a>
## Abstract

This project proposes a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid ``identity leakage,'' the generated samples that accidentally matched those used in the model's training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible. The source codes and generative models trained for this work are offered along with the paper.

<a name="gan-training"/></a>
## Step1: Training GAN Models on the Authentic Iris Samples

“Authentic TCL” and “Authentic noTCL” collections, *ND3D*, *ND Cosmetic Contacts*, and *BXGRID* datasets published by the University of Notre Dame, were used to train generative models synthesizing Synthetic TCL and Synthetic noTCL samples (used later in Step
3), respectively. We employed a class-conditional Style-GAN2 model to generate Synthetic TCL iris samples. For each textured contact lens brand, we defined a separate class. Synthetic noTCL iris images were generated using an unconditional StyleGAN2 model. The training code was adopted from the NVIDIA repository [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) [StylaGAN2](https://github.com/NVlabs/stylegan2?tab=readme-ov-file), specifically using the StyleGAN2 and StyleGAN2-ADA configurations. 

Please refer to Table 1, cited in the paper, for more information on train and test datasets.

https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/dataset_tool.py


<a name="gan-weights"/></a>
### Weight Acquisition

- Pre-trained StyleGAN Model Trained on the Authentic noTCL Samples: Download the weight for StyleGAN model trained on the authentic noTCL samples from [noTCL GAN](https://notredame.box.com/s/lgdanm3tb16e1t3ear7n4glhxeci8a6r).

- Pre-trained StyleGAN Model Trained on the Authentic TCL samples: Download the weight for StyleGAN model trained on the authentic noTCL samples from [TCL GAN] [Model](https://notredame.app.box.com/folder/278643866297).



<a name="gan-synthesizing"/></a>
## Step2: Image Synthesis and Identity Leakage Mitigation]



**Note:** Instructions of how to request a copy of the synthetic iris dataset can be found at [dataset](https://notredame.app.box.com/folder/258825225412).


<a name="pad-tarining"/></a>
## Step3: Training of Iris PAD Models with Synthetic Data Only

The code processes cropped iris images both with and without contact lenses as input, generating a PA score ranging from 0 to 1. A score of 0 indicates the sample without a contact lens, while a score of 1 signifies the sample with a contact lens.

<a name="requirements"/></a>
### Requirements
To run the code you need to install Pytorch, Numpy, Scipy, Pillow. Create a conda environment as below: 

```conda create —name dNetPAD```

```conda activate dNetPAD```

```conda install pytorch torchvision -c pytorch```

```conda install -c anaconda numpy``` 

```conda install -c anaconda scipy```

```conda install -c anaconda pillow``` 



## Training PAD Models
To train the models and validate it on your validation data during the the training, run the below code

```python train.py -csvPath csvFilePath  -datasetPath datasetImagesPath -method modelName -outputPath resultPath```

The format of the dataset CSV file is as below:
<br>train,notcl,image1.png
<br>train,tcl,image2.png
<br>test,notcl,image3.png
<br>test,tcl,image4.png


<a name="pad-testing"/></a>
## Step4: Testing of Iris PAD Models Used in the Paper with Unseen Authentic Data
<a name="PAD-weights"/></a>

### Weight Acquisition(#PAD-weights)
Download our best pre-trained model from [Models](https://notredame.app.box.com/folder/278643866297).

After training the models, select the one with the highest accuracy on the validation set to evaluate its performance on unseen data by running the below code

```python test.py -csvPath csvFilePath -modelPath bestModelPth  -trainData "synthetic" -model modelName -results  resultPath -scoreFile "score.csv"```

A CSV file containing PA scores will be generated in the same folder as the images.


<a name="citation"/></a>
### Citation

Research paper summarizing the paper:


<a name="acknowledgment"/></a>
### Acknowledgment
___________________________________________________________________________________________


The PAD code was adopted from [DeNetPAD](https://github.com/iPRoBe-lab/D-NetPAD/tree/master).


