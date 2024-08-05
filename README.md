# Privacy-Safe Iris Presentation Attack Detection #
Official GitHub repository of the paper: Mahsa Mitcheff, Patrick Tinsley, and Adam Czajka, "Privacy-Safe Iris Presentation Attack Detection," [IJCB](https://ijcb2024.ieee-biometrics.org), Buffalo, NY, September 15-18, 2024
<br><br>

![pipiline](https://github.com/CVRL/PrivacySafeIrisPAD/blob/main/pipiline.png)

Overview of the pipeline of privacy-safe, synthetic data-only iris presentation attack detection (PAD) training and validation. TCL and noTCL denote images of irises with and without contact lenses, respectively. After training generative models *Step 1*, we exclusively use synthetically-generated data (mimicking irises both with and without textured contact lenses) to train iris PAD as usual (*Step 3*). The iris matcher is used (in *Step 2*) to exclude synthetic samples that are too close to non-synthetic samples used for generative models training, which prevents the leakage of identity information from the training set into the generated samples. Resulting iris PAD methods are tested on regular (non-synthetic) data composed of bona fide and fake samples (*Step 4*).

## Table of contents
* [Abstract](#abstract)
* [Source Code for StyleGAN Models](#gan-code)
* [Steps for Image Synthesis](#gan-synthesizing)
    * [Trained StyleGAN Model Weights](#gan-weights)
    * [Generating Samples](#gan-samples)
* [Accessing Synthetic Iris Images](#samples)
* [Training Iris PAD Models with Synthetic Data](#pad-tarining)
  * [Environments Requirements](#requirements)
* [Testing Iris PAD Models with Unseen Data](#pad-testing)
* [Citation](#citation)
* [Acknowledgment](#acknowledgment)

<a name="abstract"/></a>
### Abstract

This project proposes a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid *identity leakage*, the generated samples that accidentally matched those used in the model's training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible. The source codes and generative models trained for this work are offered along with the paper.

<a name="gan-code"/></a>
### Source Code for StyleGAN Models

To train generative models for synthesizing synthetic TCL and noTCL samples, we used code from NVIDIA's [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StylaGAN2](https://github.com/NVlabs/stylegan2?tab=readme-ov-file) repositories, specifically utilizing the StyleGAN2 and StyleGAN2-ADA configurations.

- We employed a class-conditional [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) model to generate synthetic TCL iris samples, defining a separate class for each textured contact lens brand.
  
- For synthetic noTCL iris images, we used an unconditional [StylaGAN2](https://github.com/NVlabs/stylegan2?tab=readme-ov-file) model."

___________________________________________________________________________________________

<a name="gan-synthesizing"/></a>
### Steps for Image Synthesis

<a name="gan-weights"/></a>
#### Trained StyleGAN Model Weights

To generate noTCL and TCL samples using our pre-trained StyleGAN models, first you need to download the weights from the links below:

- Pre-trained StyleGAN Model Weights for Authentic noTCL Samples [Pre-trained noTCL GAN](https://notredame.box.com/s/oe1ez0hu3tn0x93meujlk7epsjsskfbp). 

- Pre-trained StyleGAN Model Weights for Authentic TCL samples [Pre-trained TCL GAN](https://notredame.app.box.com/file/1613090265358?s=v3kg037hy05luyui4a8emqrzqs1522k7).

<a name="gan-samples"/></a>
#### Generating Samples
After downloading the weights, run this code [generate GAN samples](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/generate.py) in order to generate synthetic noTCL and TCL iris samples using our pre-trained models. Please refer to NVIDIA's github repository for more information on how to use the code [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main).

For example, to generate samples from the first class condition of TCL irises, use the code below. Adjust the class argument from 1 to 7 as needed. For noTCL, you do not need to set the class argument.

```python generate.py --network=network-snapshot-conditional-025000.pkl --seeds=0-1000 --outdir=/generated_samples/condition1 --class=1``` 
  
___________________________________________________________________________________________

<a name="samples"/></a>
### Accessing Synthetic Iris Images

Instructions on how to request a copy of the synthetic iris dataset used in this paper can be found at [dataset](https://notredame.app.box.com/folder/258825225412).

___________________________________________________________________________________________

<a name="pad-tarining"/></a>
### Training Iris PAD Models with Synthetic Data

To train the PAD models run the below code 

```python train.py -csvPath csvFilePath  -datasetPath datasetImagesPath -method modelName -outputPath resultPath```

The format of the dataset CSV file is as below:
<br>train,notcl,image1.png
<br>train,tcl,image2.png
<br>test,notcl,image3.png
<br>test,tcl,image4.png

The code processes cropped iris images both with and without contact lenses as input, generating a PA score ranging from 0 to 1. A score of 0 indicates the sample without a contact lens, while a score of 1 signifies the sample with a contact lens.

**Note:** The PAD code was adopted from [DeNetPAD](https://github.com/iPRoBe-lab/D-NetPAD/tree/master).

<a name="requirements"/></a>
#### Environments Requirements
To run the code you need to install Pytorch, Numpy, Scipy, Pillow. Create a conda environment as below: 

```conda create —name dNetPAD```

```conda activate dNetPAD```

```conda install pytorch torchvision -c pytorch```

```conda install -c anaconda numpy``` 

```conda install -c anaconda scipy```

```conda install -c anaconda pillow``` 

___________________________________________________________________________________________

<a name="pad-testing"/></a>
### Testing Iris PAD Models with Unseen Data
To test your data on our pre-trained PAD model, first download the models from [Pre-trained PAD Models](https://notredame.app.box.com/folder/278643866297).

After downloading the trained models, run the code below on your dataset to evaluate the models' performance on unseen data.

```python test.py -csvPath csvFilePath -modelPath bestModelPth  -trainData "synthetic" -model modelName -results resultPath -scoreFile "score.csv"```

A CSV file containing PA scores will be generated in the same folder as the images.

___________________________________________________________________________________________

<a name="citation"/></a>
### Citation

Research paper summarizing the paper:
```
@Article{Privacy-Safe IPAD,
  author    = {Siamul Karim Khan and Patrick J. Flynn and Adam Czajka},
  journal   = {...},
  title     = {{Mahsa Mitcheff, Patrick Tinsley, and Adam Czajka}},
  year      = {2024},
  issn      = {...},
  month     = {...},
  number    = {...},
  pages     = {...},
  volume    = {...},
  abstract  = {...},
  doi       = {...},
  keywords  = {iris pad;genrative models},
  publisher = {...},
}
```

___________________________________________________________________________________________

<a name="acknowledgment"/></a>
### Acknowledgment
This material is based upon work partially supported by the National Science Foundation under Grant No. 2237880. Any opinions, findings, and conclusions
or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

