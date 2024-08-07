# Privacy-Safe Iris Presentation Attack Detection #
Official GitHub repository of the paper: Mahsa Mitcheff, Patrick Tinsley, and Adam Czajka, "Privacy-Safe Iris Presentation Attack Detection," [IJCB](https://ijcb2024.ieee-biometrics.org), Buffalo, NY, September 15-18, 2024
<br><br>

![pipiline](https://github.com/CVRL/PrivacySafeIrisPAD/blob/main/pipiline.png)

Overview of the pipeline of privacy-safe, synthetic data-only iris presentation attack detection (PAD) training and validation. TCL and noTCL denote images of irises with and without contact lenses, respectively. After training generative models *Step 1*, we exclusively use synthetically-generated data (mimicking irises both with and without textured contact lenses) to train iris PAD as usual (*Step 3*). The iris matcher is used (in *Step 2*) to exclude synthetic samples that are too close to non-synthetic samples used for generative models training, which prevents the leakage of identity information from the training set into the generated samples. Resulting iris PAD methods are tested on regular (non-synthetic) data composed of bona fide and fake samples (*Step 4*).

## Table of contents
* [Abstract](#abstract)
* [Source Code and Weights for StyleGAN Models](#gan-code)
* [Generating Synthetic Iris Samples (using our trained StyleGAN Models)](#gan-samples)
* [Accessing Synthetic Iris Samples (used in this paper)](#samples)
* [Training and Evaluating Iris PAD Models with Synthetic Data](#pad-tarining-evaluating)
* [Citation](#citation)
* [Acknowledgment](#acknowledgment)

<a name="abstract"/></a>
### Abstract

This project proposes a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid *identity leakage*, the generated samples that accidentally matched those used in the model's training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible. The source codes and generative models trained for this work are offered along with the paper.

<a name="gan-code"/></a>
#### Source Code and Weight for StyleGAN Models

NVIDIA's [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) was used to synthesize TCL iris samples and weights of our trained model can be downloaded from here [Trained StyleGAN Model Weights on Authentic noTCL Samples](https://notredame.box.com/s/oe1ez0hu3tn0x93meujlk7epsjsskfbp). 

NVIDIA's [StylaGAN2](https://github.com/NVlabs/stylegan2?tab=readme-ov-file) was used to synthesize noTCL iris samples and weights of our trained model can be downloaded from here [Trained StyleGAN Model Weights on Authentic TCL Samples](https://notredame.app.box.com/file/1613090265358?s=v3kg037hy05luyui4a8emqrzqs1522k7).

___________________________________________________________________________________________


<a name="gan-samples"/></a>
#### Generating Synthetic Iris Samples (using our trained StyleGAN Models
After downloading the weights, use this [code](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/generate.py) to generate synthetic noTCL and TCL iris samples with our pre-trained models. 

The example code below uses the pre-trained model *network-snapshot-conditional-025000.pkl* to generate 1,000 synthetic TCL iris images, saving them to the */generated_samples/condition1* directory. Adjust the *--class* argument from 1 to 7 to generate samples for different contact lens brands.

```python generate.py --network=network-snapshot-conditional-025000.pkl --seeds=0-1000 --outdir=/generated_samples/condition1 --class=1``` 


Please refer to NVIDIA's github repository for more information on how to use the code [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main).
  
___________________________________________________________________________________________

<a name="samples"/></a>
### Accessing Synthetic Iris Samples (used in this paper)

Instructions on how to request a copy of the synthetic iris dataset used in this paper can be found at [dataset](https://notredame.app.box.com/folder/258825225412).

___________________________________________________________________________________________

<a name="pad-tarining-evaluating"/></a>
### Training and Evaluating Iris PAD Models with Synthetic Data

To train the PAD models using the synethtic sample run the below code 

```python train.py -csvPath csvFilePath  -datasetPath datasetImagesPath -method modelName -outputPath resultPath```

After training your model models, use the code below to evaluate the models' performance on unseen data.

```python test.py -csvPath csvFilePath -modelPath bestModelPth  -trainData "synthetic" -model modelName -results resultPath -scoreFile "score.csv"```


**Note:** The PAD code was adopted from [DeNetPAD](https://github.com/iPRoBe-lab/D-NetPAD/tree/master) and please refer to this repo for more information about environment and how to prepare your train and test sets. We adopted the code to work with DenseNet, Resnet, and vision transformer (ViT) model and included augmentation pipeline in the [data loader](https://github.com/CVRL/PrivacySafeIrisPAD/blob/main/dataset_Loader.py) 


___________________________________________________________________________________________

<a name="citation"/></a>
### Citation

Research paper summarizing the paper:
```
@Article{Privacy-Safe IPAD,
  author    = {Mahsa Mitcheff, Patrick Tinsley, and Adam Czajka},
  journal   = {...},
  title     = {Privacy-Safe Iris Presentation Attack Detection},
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

