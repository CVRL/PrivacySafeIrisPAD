import os
import tqdm
import numpy as np
import torch
import random
import torch.utils.data as data_utl
from PIL import Image, ImageFilter
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, method, train_test, random=True, c2i={}):

        self.split_file = split_file
        self.method = method
        self.root = root
        self.train_test = train_test
        self.random = random
        self.image_size = 229
        
        # Image pre-processing
        if self.train_test == 'test':
            """inception expects (299,299) sized images"""
            if self.method == 'inception': 
                self.image_size = 229
            else: 
                self.image_size = 224
                
        self.transform_test = transforms.Compose([
            transforms.Resize([self.image_size, self.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            
        # Class assignment
        self.class_to_id = c2i
        self.id_to_class = []
        self.assign_classes()
        
        # Data loading
        self.get_data()
        
    # Class assignment
    def assign_classes(self):
        for i in range(len(self.class_to_id.keys())):
            for k in self.class_to_id.keys():
                if self.class_to_id[k] == i:
                    self.id_to_class.append(k)
    
    # Data loading (Reading data from CSV file)
    def get_data(self):
        self.data = []
        print('Reading data from CSV file...', self.train_test)
        cid = 0
        with open(self.split_file, 'r') as f:
            for l in f.readlines():
                v = l.strip().split(',')
                if self.train_test == v[0]:
                    image_name = v[2].replace(v[2].split('.')[-1], 'png')
                    imagePath = self.root + image_name
                    c = v[1]
                    if c not in self.class_to_id:
                        self.class_to_id[c] = cid
                        self.id_to_class.append(c)
                        cid += 1
                    # Storing data with image path and class
                    if os.path.exists(imagePath):
                        self.data.append([imagePath, self.class_to_id[c]])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # get image path, image name, and class from data
        imagePath, cls = self.data[index]
        imageName = imagePath.split('/')[-1]
        path = imagePath
        
        #####################################################################################
        ############### Read the train data, do pre-processing and augmentation #############
        #####################################################################################
        if self.train_test == 'train':
            
            # Reading the image and convert it to gray image
            img = Image.fromarray(np.array(Image.open(path).convert('RGB'))[:, :, 0], 'L')
            
            # Resize the image
            if self.method == 'inception':  
                img = img.resize((229, 229), Image.BILINEAR)  
            else:
                img = img.resize((224, 224), Image.BILINEAR)
            
            ###########################################
            ########### Data augmentation #############
            ###########################################
            # 1) horizontal flip
            if random.random() < 0.5:    
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # 2) Affine transformation
            aug = iaa.Affine(scale=(0.8,1.25), translate_px={"x": (-15, 15), "y": (-15, 15)}, rotate=(-30, 30), mode='edge')
            img_np = aug(images = np.expand_dims(np.expand_dims(np.uint8(img), axis=0), axis=-1))
            img = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
            
            # 3) Sharpening, 4)blurring, 5)Gaussian noise, 6)contrast change, 7)enhance brightness
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])

                if random_choice == 1:
                    # 3) sharpening
                    random_degree = np.random.choice([1,2,3])
                    if random_degree == 1:
                        img = img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 2:
                        img= img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    elif random_degree == 3:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        img_np = aug(images = np.array(img))
                        img = Image.fromarray(img_np)

                elif random_choice == 2:
                    # 4) blurring
                    random_degree = np.random.choice([1,2,3,4])
                    if random_degree == 1:
                        aug = iaa.GaussianBlur(sigma=(2.0,5.0))#(0.1, 1.0))
                        img_np = aug(images = np.array(img))
                        img = Image.fromarray(img_np)
                    elif random_degree == 2:
                        aug = iaa.imgcorruptlike.MotionBlur(severity=2)#1)
                        img_np = aug(images = np.array(img))
                        img = Image.fromarray(img_np)
                    elif random_degree == 3:
                        aug = iaa.imgcorruptlike.GlassBlur(severity=2)#1)
                        img_np = aug(images = np.array(img))
                        img = Image.fromarray(img_np)
                    elif random_degree == 4:
                        aug = iaa.imgcorruptlike.DefocusBlur(severity=3)#1)
                        img_np = aug(images = np.array(img))
                        img = Image.fromarray(img_np)
                        
                elif random_choice == 3:
                    # 5) AdditiveLaplaceNoise
                    if random.random() < 0.5:
                        aug = iaa.AdditiveLaplaceNoise(scale=(0, 3))
                        img_np = aug(images = np.array(img))
                        img = Image.fromarray(img_np)
                
                
            if random.random() < 0.5:
                # 6) contrast change
                random_degree = np.random.choice([1,2,3,4,5])
                if random_degree == 1:
                    aug = iaa.GammaContrast((1.2, 1.8))
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.75))
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(10, 12), cutoff=(0.5, 0.75))
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.5, 1.4))
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)
                else:
                    # 7) Enhance Brightness
                    aug = iaa.pillike.EnhanceBrightness(factor=(0.8, 1.9))
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)


            if random.random() < 0.5:
                # 8) sequence of randomness
                random_degree = np.random.choice([1,2,3,4,5])
                if random_degree == 1:
                    aug = iaa.Sequential([
                          iaa.GammaContrast((0.5, 2.0)),
                          iaa.pillike.EnhanceBrightness(),
                          iaa.GaussianBlur(sigma=(0.1,1.0))
                          ])
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)
                elif random_degree == 2:
                    aug = iaa.Sequential([
                          iaa.LinearContrast((0.4, 1.6)),
                          iaa.pillike.EnhanceBrightness(),
                          iaa.imgcorruptlike.MotionBlur(severity=1)
                          ])
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)

                elif random_degree == 3:
                    aug = iaa.Sequential([
                          iaa.LogContrast(gain=(0.6, 1.4)),
                          iaa.pillike.EnhanceBrightness(),
                          iaa.imgcorruptlike.GlassBlur(severity=1)
                          ])
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)

                elif random_degree == 4:
                    aug = iaa.Sequential([
                          iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                          iaa.pillike.EnhanceBrightness(),
                          iaa.GaussianBlur(sigma=(0.1,2.0))
                          ])
                    img_np = aug(images = np.array(img))
                    img = Image.fromarray(img_np)

                else:
                     aug = iaa.Sequential([
                          iaa.pillike.EnhanceBrightness(factor=(0.1, 1.9)),
                          iaa.imgcorruptlike.DefocusBlur(severity=1)
                          ])
                     img_np = aug(images = np.array(img))
                     img = Image.fromarray(img_np)

                              
            # convert PIL to tensor
            pil_to_torch = ToTensor()
            torch_img = pil_to_torch(img)
            # Normalize the tensor
            tranform_img = transforms.Normalize(mean=[0.485],std=[0.229])(torch_img)

        elif self.train_test == 'test':
            # Reading of the image and apply transformation
            img = Image.open(path)
            tranform_img = self.transform_test(img)

        img.close()

        # Repeat NIR single channel thrice before feeding into the network
        tranform_img= tranform_img.repeat(3,1,1)

        return tranform_img[0:3,:,:], cls, imageName

#if __name__ == '__main__':

#    dataseta = datasetLoader('../TempData/Iris_OCT_Splits_Val/test_train_split.csv', 'PathToDatasetFolder', train_test='train')

#    for i in range(len(dataseta)):
#        print(len(dataseta.data))
