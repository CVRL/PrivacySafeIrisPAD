# Privacy-Safe Iris Presentation Attack Detection #

Official GitHub repository for the paper: Mahsa Mitcheff, Patrick Tinsley, Adam Czajka, "Privacy-Safe Iris Presentation Attack Detection," IEEE/IAPR International Joint Conference on Biometrics, Buffalo, NY, September 15-18, 2024 **([ArXiv](https://arxiv.org/abs/2408.02750) | IEEEXplore)**

![pipiline](https://github.com/CVRL/PrivacySafeIrisPAD/blob/main/pipiline.png)

Overview of the pipeline of privacy-safe, synthetic data-only iris presentation attack detection (PAD) training and validation. TCL and noTCL denote images of irises with and without contact lenses, respectively. After training generative models *Step 1*, we exclusively use synthetically-generated data (mimicking irises both with and without textured contact lenses) to train iris PAD as usual (*Step 3*). The iris matcher is used (in *Step 2*) to exclude synthetic samples that are too close to non-synthetic samples used for generative models training, which prevents the leakage of identity information from the training set into the generated samples. Resulting iris PAD methods are tested on regular (non-synthetic) data composed of bona fide and fake samples (*Step 4*).

### Table of contents
* [Abstract](#abstract)
* [Source Code and Weights for StyleGAN Models](#gan-code)
* [Generating Synthetic Iris Samples (using our trained StyleGAN models)](#gan-samples)
* [Accessing Synthetic Iris Samples (used in this paper)](#samples)
* [Training and Evaluating Iris PAD Models with Synthetic Data](#pad-tarining-evaluating)
* [Citation](#citation)
* [Acknowledgment](#acknowledgment)

<a name="abstract"/></a>
### Abstract

This project proposes a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid *identity leakage*, the generated samples that accidentally matched those used in the model's training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible. The source codes and generative models trained for this work are offered along with the paper.
___________________________________________________________________________________________
<a name="gan-code"/></a>
### Source codes and weights for StyleGAN models

NVIDIA's [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) was used to synthesize **TCL** iris samples (that is, irises with textured contact lenses) and the weights of our trained model are available at [this link](https://notredame.box.com/s/ai6ta1ocfmb37bk6gxy9owvbsrndf2zz). 

NVIDIA's [StylaGAN2](https://github.com/NVlabs/stylegan2?tab=readme-ov-file) was used to synthesize **noTCL** iris samples (that is, irises without contact lenses) and the weights of our trained model are available at [this link](https://notredame.box.com/s/l52ym2rgeii6volqeroqy2zb98d5juvm).

___________________________________________________________________________________________
<a name="gan-samples"/></a>
### Generating synthetic iris samples (using our trained StyleGAN models)
After downloading our StyleGAN2 weights, use [NVIDIA's code](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/generate.py) to generate synthetic noTCL and TCL iris samples with our pre-trained models. 

The example code below uses the pre-trained model *network-snapshot-conditional-025000.pkl* to generate 1,000 synthetic TCL iris images, and saves them to the */generated_samples/condition1* directory. Adjust the *--class* argument from 1 to 7 to generate samples representing different contact lens brands.

```
python generate.py --network=network-snapshot-conditional-025000.pkl --seeds=0-1000 --outdir=/generated_samples/condition1 --class=1 
```

Please refer to NVIDIA's github repository for more information on how to use the code [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main).
___________________________________________________________________________________________
<a name="samples"/></a>
### Dataset of synthetic iris samples used in this paper

Instructions on how to request a copy of the synthetic iris dataset used in this paper can be found at [the CVRL webpage](https://cvrl.nd.edu/projects/data/).
___________________________________________________________________________________________
<a name="pad-tarining-evaluating"/></a>
### Training and evaluating iris PAD models with synthetic data

To train the PAD models using the synethtic samples use the following command:

```
python train.py -csvPath csvFilePath  -datasetPath datasetImagesPath -method modelName -outputPath resultPath
```

After training your model models, use the following command to evaluate the models' performance on unseen data:

```
python test.py -csvPath csvFilePath -modelPath bestModelPth  -trainData "synthetic" -model modelName -results resultPath -scoreFile "score.csv"
```

**Note:** The PAD model code was adapted from [DNetPAD GitHub repo](https://github.com/iPRoBe-lab/D-NetPAD/tree/master). For more information about the environment setup and how to prepare your training and test sets, please refer to the original DNetPAD GitHub repo. For this paper, we modified the original DNetPAD code to work with DenseNet, ResNet, and Vision Transformer (ViT) models, and included an augmentation pipeline in the data loader.
___________________________________________________________________________________________
<a name="citation"/></a>
### Citation

If you find this work useful in your research, please cite the following paper:
```
@inproceedings{mitcheff2024privacysafeirispresentationattack,
      title={Privacy-Safe Iris Presentation Attack Detection}, 
      author={Mahsa Mitcheff, Patrick Tinsley and Adam Czajka},
      year={2024},
      booktitle={IEEE International Joint Conference on Biometrics},
}
```
___________________________________________________________________________________________

<a name="acknowledgment"/></a>
### Acknowledgment
This material is based upon work partially supported by the National Science Foundation under Grant No. 2237880. Any opinions, findings, and conclusions
or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

