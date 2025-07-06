import torch
from torchvision import transforms
import os
import pandas as pd
from model_multimodal import MultiModalModel
from dataset import load_dicom_image


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)

# Load the model weights
model.load_state_dict(torch.load('../checkpoint/bone_age_model_dcm.pth'))
model.eval()  # Set the model to evaluation mode

# Image transformation for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization for grayscale
])


def inference(dicom_path, model, transform):
    image, dicom = load_dicom_image(dicom_path)
    gender = dicom.PatientSex  # Extract gender from DICOM header
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Send to device
    
    gender_tensor = torch.tensor(0 if gender == 'M' else 1).float().unsqueeze(0).to(device)  # Convert gender to tensor
    
    with torch.no_grad():  # No need to track gradients
        output = model(image, gender_tensor)
        predicted_age = output.item()  # Get the predicted age
    
    return predicted_age


# Directory containing the test DICOM files
test_directory = '../smc-bone-age/boneage_inference'

# Iterate over each DICOM file in the test directory
results = []
for filename in os.listdir(test_directory):
    if filename.endswith(".dcm"):
        file_path = os.path.join(test_directory, filename)
        predicted_age = inference(file_path, model, transform)
        results.append((filename, predicted_age))
        print(f"Image: {filename}, Predicted Age (month): {predicted_age}")

# Optionally, save the results to a CSV file
df_results = pd.DataFrame(results, columns=['Image', 'Predicted Age (month)'])
df_results.to_csv('../output_smc/inference_results.csv', index=False)