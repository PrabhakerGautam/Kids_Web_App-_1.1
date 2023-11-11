import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
#from torchvision import datasets
from PIL import Image
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

# Streamlit app
st.set_page_config(page_title="2.Tain", layout='centered',page_icon='./images/brain.png')


          
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Ensure the image has 3 channels (RGB)
    img = img.convert('RGB')
    
    # Resize the image to the model's input size (28x28)
    img = img.resize((28, 28))
    
    # Convert the image to a PyTorch tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)
    
    return img_tensor

# Load the generated dataset
def load_dataset(dataset_folder):
    images = []
    labels = []
    for label in os.listdir(dataset_folder):
        label_folder = os.path.join(dataset_folder, label)
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            img_tensor = load_and_preprocess_image(image_path)
            images.append(img_tensor)
            labels.append(int(label))
    return (torch.stack(images), torch.tensor(labels))

# Load the dataset
dataset_folder = "dataset"
images, labels = load_dataset(dataset_folder)

# Define your batch size
batch_size = 64

# Create a model instance
model = SimpleCNN()

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



st.markdown(""" 
            # **Let's Teach and Train Together!**

#### 🚀 Welcome to the "Train" section, where the real magic happens!


**Your computer friend is like a clever robot, and you're the mentor**.

**More epochs mean more learning for our model. It's like giving it more time to become super smart! 🚀🧠 You can adjust the number of epochs on the sidebar.** 

**So, feel free to try different numbers of epochs and see the magic happen in our learning adventure! ✨📚**


###### Let's embark on the training adventure and see your model grow before your eyes! 🎩📝🤖

            
            """)

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# User input for the number of epochs
num_epochs = st.sidebar.number_input("Number of epochs:", min_value=1, value=10, step=1)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
if st.button("**Train Model**"):
    st.info("Training the model...")

    # Define your training dataset and data loader
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / total
            progress_bar = st.progress(0)
            for percent_complete in range(0, 101, 10):
                progress_bar.progress(percent_complete)
                time.sleep(0.2) 
            st.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
    if accuracy<50:
        st.warning("Draw, teach, and boost accuracy above 50%. Your art makes your computer friend smarter!")
    st.success("Trained model saved ")
    st.write("#### Next Step, Model Testing")
    st.markdown("[Click here to begin the Testing process](/Test/)")
#saving the model
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)
model_path = os.path.join(model_folder, "trained_model.pth")
torch.save(model.state_dict(), model_path)

