import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# 데이터셋 클래스 정의
class BoneAgeDataset_png(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = f"{self.image_dir}/{self.dataframe.iloc[idx, 0]}.png"
        image = Image.open(img_name).convert('L')  # Grayscale로 이미지 변환
        label = self.dataframe.iloc[idx, 1]
        gender = self.dataframe.iloc[idx, 2]  # Assuming gender is the third column
        gender = torch.tensor(0 if gender == 'True' else 1)  # Convert gender to tensor (0 for Male, 1 for Female)

        if self.transform:
            image = self.transform(image)

        return image, label, gender
    
# 데이터셋 클래스 정의
class BoneAgeDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        dicom_path = f"{self.image_dir}/{self.dataframe.iloc[idx, 0]}.dcm"

        image, _ = load_dicom_image(dicom_path)
        
        label = self.dataframe.iloc[idx, 1]
        gender = self.dataframe.iloc[idx, 2]  # Assuming gender is the third column
        
        '''
        #gender update only one
        print(gender)
        if gender:
            dicom.PatientSex = 'M'
            print('M')
        else:
            dicom.PatientSex = 'F'
            print('F')
        dicom.save_as(dicom_path)
        '''

        gender = torch.tensor(0 if gender else 1)   # Convert gender to tensor (0 for Male, 1 for Female)
        #gender = torch.tensor(0 if gender == 'True' else 1)   # Convert gender to tensor (0 for Male, 1 for Female)

        if self.transform:
            image = self.transform(image)

        return image, label, gender
    

def load_dicom_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    # Apply modality LUT (rescale slope/intercept) and VOI LUT (windowing)
    image = apply_modality_lut(dicom.pixel_array, dicom)
    image = apply_voi_lut(image, dicom)
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32)
    # Handle Photometric Interpretation
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.max(image) - image
    # Convert to PIL image
    image = Image.fromarray(image).convert('L')
    return image, dicom