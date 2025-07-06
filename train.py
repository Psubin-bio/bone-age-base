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


# 데이터셋 클래스 정의
class BoneAgeDataset(Dataset):
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
    
def plot_performance(training_losses, validation_losses, validation_maes):
    epochs = range(1, len(training_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, 'bo-', label='Training loss')
    plt.plot(epochs, validation_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_maes, 'go-', label='Validation MAE')
    plt.title('Validation Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale 이미지를 위한 정규화
])

# 데이터 로딩
print("----------------start data loading----------------")
train_df = pd.read_csv('../rsna-bone-age/boneage-training-dataset.csv')
test_df = pd.read_csv('../rsna-bone-age/boneage-test-dataset.csv')

train_dataset = BoneAgeDataset(train_df, '../rsna-bone-age/boneage-training-dataset/boneage-training-dataset', transform=transform)
test_dataset = BoneAgeDataset(test_df, '../rsna-bone-age/boneage-test-dataset/boneage-test-dataset', transform=transform)

# Calculate the sizes for training and validation sets (e.g., 80-20 split)
print("----------------data split (8:2) ----------------")
total_train = len(train_dataset)
len_train = int(0.8 * total_train)
len_val = total_train - len_train

# Split the dataset
train_ds, val_ds = random_split(train_dataset, [len_train, len_val])

# Data loaders for training and validation sets
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("----------------model setting----------------")
# 모델 설정
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.vgg_features = models.vgg16(pretrained=True)
        self.vgg_features.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg_features.classifier[6] = nn.Linear(4096, 512)
        
        self.gender_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc = nn.Linear(544, 1)  # 512 from VGG and 32 from gender_layer

    def forward(self, image, gender):
        image_features = self.vgg_features(image)
        gender_features = self.gender_layer(gender.float().unsqueeze(1))
        combined_features = torch.cat((image_features, gender_features), dim=1)
        output = self.fc(combined_features)
        return output
    
# 기기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)


# 손실 함수 및 최적화
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 학습 함수
def validate_and_save_results(model, loader, criterion, device, epoch, result_save=True):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    total_mae = 0.0
    count = 0
    results = []
    
    for inputs, labels, genders in loader:
        inputs, labels, genders = inputs.to(device), labels.to(device), genders.to(device)
        with torch.no_grad():
            outputs = model(inputs, genders)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            val_loss += loss.item() * inputs.size(0)
            total_mae += np.sum(np.abs(outputs.cpu().numpy() - labels.cpu().numpy()))
            count += inputs.size(0)
            for i in range(inputs.size(0)):
                results.append((loader.dataset.indices[count - inputs.size(0) + i], labels[i].item(), outputs[i].item()))
    
    average_loss = val_loss / count
    average_mae = total_mae / count
    
    if result_save == True:
        # Save the results to CSV
        df_results = pd.DataFrame(results, columns=['Index', 'Actual Age', 'Predicted Age'])
        df_results.to_csv(f'./output/validation_results_epoch_{epoch}.csv', index=False)
    
    return average_loss, average_mae


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    training_losses = []
    validation_losses = []
    validation_maes = []
    best_loss = float('inf')  # Initialize best loss for model saving
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels, genders in train_loader:
            inputs, labels, genders = inputs.to(device), labels.to(device), genders.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, genders)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        training_losses.append(train_loss)
        
        # Validate after each epoch
        val_loss, val_mae = validate_and_save_results(model, val_loader, criterion, device, epoch)
        validation_losses.append(val_loss)
        validation_maes.append(val_mae)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation MAE: {val_mae}')
        
        # Save model if improvement
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './output/bone_age_model.pth')
            print(f'Model saved with loss: {val_loss}')
    
    return training_losses, validation_losses, validation_maes

# 기기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 실행
num_epochs = 100
print("----------------start training----------------")
training_losses, validation_losses, validation_maes = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
plot_performance(training_losses, validation_losses, validation_maes)