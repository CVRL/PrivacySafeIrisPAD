import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    device = torch.device('cuda')
    parser.add_argument('-imageFolder', default='/TempData/Iris_Split/',type=str)
    parser.add_argument('-csvPath', required=True, default= '/TempData/Iris_Split/test_tcl_live.csv', type=str)
    parser.add_argument('-scoreFile', required=True, default= 'score.csv', type=str)
    parser.add_argument('-model', default='DenseNet',type=str)
    parser.add_argument('-results', default='',type=str)
    parser.add_argument('-modelPath',  default='Model/D-NetPAD_Model.pth',type=str)
    parser.add_argument('-nClasses', default= 2, type=int)
    parser.add_argument('-trainData', default='synthetic', type=str)

    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load weights of the model
    weights = torch.load(args.modelPath, map_location=device)

    print("Loading the Model ...")    
    if args.model == 'DenseNet':
        img_size = 224
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args.nClasses)

    elif args.model == 'resnet':
        img_size = 224
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.nClasses)

    elif args.model == 'inception':
        img_size = 229
        #model = models.inception_v3(pretrained=True, aux_logits=False)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.nClasses)

    elif args.model == 'convnext':
        img_size = 224
        model = models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT')
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, args.nClasses)

    elif args.model == 'vit':
        img_size = 224
        model = models.vit_b_16(pretrained=True)
        num_ftrs = model.heads[-1].in_features
        model.heads[-1] = nn.Linear(num_ftrs, args.nClasses)

    # if you are using the best_model.pth
    model.load_state_dict(weights['state_dict'])
    # if not using the best model
    #model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    imagesScores=[]

    print("********* Calculating the Score **********")
    with open(args.csvPath, 'r') as f:
         csvFile = csv.reader(f)
         for imgFile in csvFile:
            fPath = os.path.join(args.imageFolder, imgFile[0])
            image = Image.open(fPath)
            # Image transformation
            tranformImage = transform(image)
            image.close()
            tranformImage = tranformImage.repeat(3, 1, 1) # for NIR images having one channel
            tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
            tranformImage = tranformImage.to(device)

            # Get model prediction score for the image, move it to cpu and convert it to numpy
            output = model(tranformImage)
            output = output.detach().cpu().numpy()

            # Select the class index with the highest score
            class_index = np.argmax(output)
            # Select the corresponding score (since output is a numpy array we choose 0 to get the content)
            PAScore = output[0, class_index]

            # ****** Normalize output score between [0,1] 
            PAScore = np.minimum(np.maximum((PAScore+15)/35,0),1)

            if class_index == 0:
                PAScore = PAScore * 100
            else:
                PAScore = (1 - PAScore ) * 100

            #print([imgFile, PAScore, class_index])
            imagesScores.append([imgFile[0], PAScore, class_index])
    

    print("*********** Save the results ****************")
    file_name = args.scoreFile
    with open(os.path.join(args.results, file_name),'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)
