import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# Assuming model setup as defined in your model script
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
        gender_features = self.gender_layer(gender.unsqueeze(1))
        combined_features = torch.cat((image_features, gender_features), dim=1)
        output = self.fc(combined_features)
        return output

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)

# Load the model weights
model.load_state_dict(torch.load('./output/bone_age_model.pth'))
model.eval()  # Set the model to evaluation mode

# Image transformation for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization for grayscale
])

def inference(image_path, gender, model, transform):
    image = Image.open(image_path).convert('L')  # Load image and convert to grayscale
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Send to device
    
    gender_tensor = torch.tensor(0 if gender == 'M' else 1).float().unsqueeze(0).to(device)  # Convert gender to tensor
    
    with torch.no_grad():  # No need to track gradients
        output = model(image, gender_tensor)
        predicted_age = output.item()  # Get the predicted age
    
    return predicted_age

# Load test dataset CSV
test_df = pd.read_csv('./rsna-bone-age/boneage-test-dataset.csv')

# Directory containing the test images
test_directory = './rsna-bone-age/boneage-test-dataset/boneage-test-dataset'

# Iterate over each image in the test directory
results = []
for filename in os.listdir(test_directory):
    if filename.endswith(".png"):
        file_id = os.path.splitext(filename)[0]  # Remove file extension
        file_path = os.path.join(test_directory, filename)
        # Assuming gender information is available in the test dataset CSV
        gender = test_df.loc[test_df['Case ID'].astype(str) == file_id, 'Sex'].values[0]
        predicted_age = inference(file_path, gender, model, transform)
        results.append((filename, predicted_age))
        print(f"Image: {filename}, Predicted Age (month): {predicted_age}")

# Optionally, save the results to a CSV file
df_results = pd.DataFrame(results, columns=['Image', 'Predicted Age (month)'])
df_results.to_csv('./output/inference_results.csv', index=False)