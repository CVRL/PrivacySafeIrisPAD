
# <center> Privacy-Safe Iris Presentation Attack Detection (IJCB) 2024 </center>

![pipiline](https://github.com/CVRL/PrivacySafeIrisPAD/blob/main/pipiline.png)

# Abstract

This project proposes a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid ``identity leakage,'' the generated samples that accidentally matched those used in the model's training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible. The source codes and generative models trained for this work are offered along with the paper.

# Data Used for Training the StyleGAN models
All of authentic TCL and noTCL samples were adopted from repository published by the University of Notre Dame while uthentic TCL samples for training the GAN model was adopted from ND3D and “ND Cosmetic Contacts and noTCL was adopted from BXGRID dataset.

# Generating Synthetic Iris Samples

“Authentic TCL” and “Authentic noTCL” collections were used to train generative models synthesizing Synthetic TCL and Synthetic noTCL samples (used later in Step
3), respectively. We employed a class-conditional Style-GAN2 model to generate Synthetic TCL iris samples. Fo each textured contact lens brand, we defined a separate class. Synthetic noTCL iris images were generated using an unconditional StyleGAN2 model. The training code was adopted from the NVIDIA repository [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) [StylaGAN2] (https://github.com/NVlabs/stylegan2?tab=readme-ov-file), specifically using the StyleGAN2 and StyleGAN2-ADA configurations. 


# Experiment 
To evaluate a potential gap between the performance of the iris PAD methods trained traditionally (with authentic samples) and trained solely with synthetic samples, we designed two experiments:

*1. First Experment: *  the iris PAD models were trained exclusively on synthetic data and tested on iris PAD benchmarks.
* 2. Second Experment:*  the iris PAD models were trained solely on authentic data (including images of irises with and without textured contact lenses), and tested on the iris PAD benchmarks; to ensure fair comparisons.


# Description on how to use the code

The code processes cropped iris images both with and without contact lenses as input, generating a PA score ranging from 0 to 1. A score of 0 indicates the sample without a contact lens, while a score of 1 signifies the sample with a contact lens.

# Requirement
To run the code you need to install Pytorch, Numpy, Scipy, Pillow. Create a conda environment as below: 

```conda create —name dNetPAD```

```conda activate dNetPAD```

```conda install pytorch torchvision -c pytorch```

```conda install -c anaconda numpy``` 

```conda install -c anaconda scipy```

```conda install -c anaconda pillow``` 


# Training
```python train.py -csvPath csvFilePath  -datasetPath datasetImagesPath -method modelName -outputPath resultPath```

The format of the dataset CSV file is as below:
<br>train,notcl,image1.png
<br>train,tcl,image2.png
<br>test,notcl,image3.png
<br>test,tcl,image4.png

# Testing
After training the model, select the one with the highest accuracy on the validation set to evaluate its performance on unseen data

```python test.py -csvPath csvFilePath -modelPath bestModelPth  -trainData "synthetic" -model modelName -results  resultPath -scoreFile "score.csv"```

A CSV file containing PA scores will be generated in the same folder as the images.

___________________________________________________________________________________________
**Note:** Instructions of how to request a copy of the dataset can be found at [dataset] (https://cvrl.nd.edu/projects/data/). 


The code was adopted from [DeNetPAD](https://github.com/iPRoBe-lab/D-NetPAD/tree/master).
