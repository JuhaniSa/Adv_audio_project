import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

# Check if CUDA (NVIDIA's parallel computing platform) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    # Get the number of available GPU devices
    num_gpu = torch.cuda.device_count()
    
    # Print information about the GPU devices
    for i in range(num_gpu):
        print(f"Using GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. PyTorch is running on CPU.")
    

class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.25)

        # Remove the GRU layer
        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adjust the fully connected layer input size
        self.fc = nn.Linear(64, num_classes)

    def _forward_features(self, x):
        x = self.dropout1(self.batch_norm1(F.relu(self.pool1(self.conv1(x)))))
        x = self.dropout2(self.batch_norm2(F.relu(self.pool2(self.conv2(x)))))
        x = self.dropout3(self.batch_norm3(F.relu(self.pool3(self.conv3(x)))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        # Apply global average pooling to reduce each feature map to a single value
        x = self.global_avg_pool(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Custom dataset class
class MelSpectrogramDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data, self.labels = self.load_data()

    def load_data(self):
        data = []
        labels = []

        for filename in os.listdir(self.data_path):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.data_path, filename)
                with open(filepath, 'rb') as f:
                    data_dict = pickle.load(f)
                    mel_spectrogram = data_dict['mel_spectrogram']
                    label = data_dict['label']
                    data.append(mel_spectrogram)
                    labels.append(label)

        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.tensor(self.labels[index])

# Function to train the model
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.dataset)
    accuracy = correct / total
    return train_loss, accuracy

# Function to validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / total
    return val_loss, accuracy

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# Get the current script directory
current_script_directory = os.path.dirname(os.path.realpath(__file__))
output_directory = os.path.join(current_script_directory, "output_mel_spectrograms/")
metadata_file = os.path.join(current_script_directory, "Data", "meta.csv")
num_classes = 10
input_shape = (128, 216)

# Load mel-spectrogram data and labels
dataset = MelSpectrogramDataset(output_directory)
print("Shape of dataset.data:", dataset.data.shape)
print("Shape of dataset.data:", dataset.data.shape)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(dataset.labels)
X_train, X_temp, y_train, y_temp = train_test_split(dataset.data, encoded_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_expanded = np.expand_dims(X_train, axis=1)  # This adds the channel dimension
X_val_expanded = np.expand_dims(X_val, axis=1)
X_test_expanded = np.expand_dims(X_test, axis=1)

# Convert the numpy arrays to PyTorch tensors
X_train_tensor = torch.Tensor(X_train_expanded)
X_val_tensor = torch.Tensor(X_val_expanded)
X_test_tensor = torch.Tensor(X_test_expanded)

print("Shape of X_train_tensor:", X_train_tensor.shape)
print("Shape of X_val_tensor:", X_val_tensor.shape)
print("Shape of X_test_tensor:", X_test_tensor.shape)

# Create DataLoader instances
train_loader = DataLoader(list(zip(X_train_tensor, y_train)), batch_size=32, shuffle=True)
val_loader = DataLoader(list(zip(X_val_tensor, y_val)), batch_size=32, shuffle=False)
test_loader = DataLoader(list(zip(X_test_tensor, y_test)), batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
input_shape = (128, 216)

# Correct the model initialization
model = SimpleCNN(input_shape=(1, 128, 216), num_classes=num_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


# Training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
    scheduler.step()
    val_loss, val_accuracy = validate_model(model, val_loader, criterion)  # Get both loss and accuracy
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)  # Correctly append validation loss
    val_accuracies.append(val_accuracy)  # Correctly append validation accuracy
    train_accuracies.append(train_accuracy)  # Append training accuracy
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test the model
test_accuracy = test_model(model, test_loader, criterion)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model and label encoder
torch.save(model.state_dict(), 'pytorch_model.pth')
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

# Visualize confusion matrix and save the figure
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    else:
        plt.show()

# Plot confusion matrix
y_true = y_test
model.eval()
with torch.no_grad():
    y_pred = []
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())

# Specify the directory to save the figures
save_directory = os.path.dirname(os.path.realpath(__file__))

plot_confusion_matrix(y_true, y_pred, classes=label_encoder.classes_, save_path=save_directory)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_directory, 'training_validation_loss.png'))

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_directory, 'training_validation_accuracy.png'))
    
    
# Calculate precision, recall, f1-score, and class-wise accuracy
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
class_accuracy = accuracy_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Class-wise Accuracy: {class_accuracy:.4f}")