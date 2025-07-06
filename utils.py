import os
import matplotlib.pyplot as plt




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

    plt.savefig('../checkpoint/plots/training_mae_swin.png')
    plt.show()



def rename_dicom_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):
            # Split the filename and get the name part before the first underscore
            parts = filename.split('_')
            if len(parts) > 1:
                new_name = parts[0] + ".dcm"
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed {filename} to {new_name}")

# Directory containing the DICOM files
#dicom_directory = './path/to/dicom/files'

# Rename the files
#rename_dicom_files(dicom_directory)
