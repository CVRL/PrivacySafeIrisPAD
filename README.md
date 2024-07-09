# Description

The code processes cropped iris images both with and without contact lenses as input, generating a PA score ranging from 0 to 1. A score of 0 indicates the sample without a contact lens, while a score of 1 signifies the sample with a contact lens.

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
