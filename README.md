
# Privacy-Safe Iris Presentation Attack Detection (IJCB) 2024

[pipiline][pipeline-v4.pdf]

This paper proposes a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid ``identity leakage,'' the generated samples that accidentally matched those used in the model's training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible. The source codes and generative models trained for this work are offered along with the paper.

# Conclusion

This paper proposed the framework in which exclusively synthetically-generated iris images were used to build the entire iris PAD method detecting textured contact lenses. This study demonstrates that it is possible to train effective iris PAD models without using any authentic data, collected from human subjects. To achieve this goal we trained unconditional generative models synthesizing iris images without contact lenses, and conditional generative models synthesizing images of irises wearing contact lenses offered by seven different manufacturers. By applying an ``identity leakage'' mitigation mechanism in the pipeline, the proposed framework offers an advantage of reducing privacy concerns associated with using iris data from authentic subjects. As a result, we obtained privacy-safe iris PAD methods that perform comparably well when tested on all the existing benchmarks offering iris images with and without textured contact lenses (benchmarks used in models training were excluded from testing to avoid bias).



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
**Note:** nstructions of how to request a copy of the dataset can be found
at [dataset] (https://cvrl.nd.edu/projects/data/). 


The code was adopted from [DeNetPAD](https://github.com/iPRoBe-lab/D-NetPAD/tree/master).
