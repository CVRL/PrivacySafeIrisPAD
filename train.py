import os
import random
import argparse
import json
from Evaluation import evaluation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader import datasetLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32) 
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=True, default= '../TempData/Iris_Splits/test_train_split.csv', type=str)
parser.add_argument('-datasetPath', default= '/TempData/Iris_Splits', type=str) 
parser.add_argument('-outputPath', required=True, default= '/OutputPath/', type=str)
parser.add_argument('-method', default= 'DenseNet', type=str)
parser.add_argument('-nClasses', default= 2, type=int) 
parser.add_argument('-seed', default=0, type=int)


args = parser.parse_args()

# For reproducibilty
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda')

if not os.path.exists(args.outputPath):
    os.mkdir(args.outputPath)

# Creation of Log folder: used to save the trained model
log_path = os.path.join(args.outputPath, 'Logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)
     
# File for logging the training process
with open(os.path.join(log_path,'%s_params.json'%args.method), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}    

# Creation of result folder: used to save the performance of trained model on the test set
result_path = os.path.join(args.outputPath , 'Results')
if not os.path.exists(result_path):
    os.mkdir(result_path)
    
##################################################################
################ Definition of model architecture ################
##################################################################

if args.method == 'DenseNet':
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.nClasses)
    
elif args.method == 'resnet':
    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    
elif args.method == 'convnext':
    model = models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT')
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, args.nClasses)

elif args.method == 'vit':
    model = models.vit_b_16(pretrained=True)
    num_ftrs = model.heads[-1].in_features
    model.heads[-1] = nn.Linear(num_ftrs, args.nClasses)

elif args.method == 'inception':
    """ Inception v3 expects (299,299) sized images """
    #model = models.inception_v3(pretrained=True, aux_logits=False)
    #model = models.inception.inception_v3(weights='Inception_V3_Weights.DEFAULT')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)

# Move the model to GPU
model = model.to(device)
    
# Description of hyperparameters
if args.method == 'convnext':
    lr = 0.004
    solver = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)

else:
    lr = 0.005 
    solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)

criterion = nn.CrossEntropyLoss()

##################################################################
############### Dataloader for train and test data ###############
##################################################################
all_classes = ['real','fake']
class_assgn = {c: i for i, c in enumerate(all_classes)}

dataseta = datasetLoader(args.csvPath, args.datasetPath, args.method, train_test='train', c2i=class_assgn) 
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=2, pin_memory=True)
print('Built training dataset.')

dataset = datasetLoader(args.csvPath, args.datasetPath, args.method, train_test='test', c2i=dataseta.class_to_id)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=2, pin_memory=True)
print('Built test dataset.')
dataloader = {'train': dl, 'test':test}

#####################################################################################
#
############### Training of the model and logging ###################################
#
#####################################################################################
train_loss=[]
test_loss=[]
bestAccuracy=0
bestEpoch=0

for epoch in range(args.nEpochs):

    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        tloss = 0.
        acc = 0.
        tot = 0   # Total number of images (dataset size)
        c = 0
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        
        with torch.set_grad_enabled(train):
            for data, cls, imageName in dataloader[phase]:

                # Data and ground truth
                data = data.to(device)
                cls = cls.to(device)

                # Running model over data
                outputs = model(data)

                # Prediction of accuracy
                pred = torch.max(outputs, dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                loss = criterion(outputs, cls)

                # Optimization of weights for training data
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'test':
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)
                
                # Loss computation
                tloss += loss.item()
                c += 1

        # Logging of train and test results
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
            train_loss.append(tloss / c)

        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            if args.method != 'convnext':
                lr_sched.step(tloss / c)
            test_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy = accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path,args.method+'_best.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames= imgNames

    with open(os.path.join(log_path,args.method+'_log.json'), 'w') as out:
        json.dump(log, out)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(log_path, args.method+'_model_'+str(epoch)+'.pth'))


# Plotting of train and test loss
plt.figure(figsize=(20,8))
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
plt.xticks(np.arange(0, args.nEpochs, 5.0))
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(result_path, args.method+'_Loss.jpg'))


# Evaluation of test set utilizing the trained model
obvResult = evaluation()
errorIndex, predictScore, threshold = obvResult.get_result(args.method, testImgNames, testTrueLabels, testPredScores, result_path)




