import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os


          
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

# Streamlit app
st.set_page_config(page_title="Kids' Digit Classifier", layout='centered',page_icon='./images/object.png')

#st.title("Kids' Digit Classifier")

#st.write("Welcome to the *Train* section, where the real magic happens! This is where your computer model learns from the dataset you created in the *Create Dataset* section. Just like a teacher instructs students, you'll guide your computer model to recognize your drawings. Watch as your model learns and gets smarter with each training session. It's like magic, but with computers! Ready to see your model grow? Let's begin the training adventure! 🚀🤖")

st.markdown(""" 
            # **Let's Teach and Train Together!**

🚀 Welcome to the "Train" section, where the real magic happens! Just like a teacher instructs students, here you'll guide your computer model to recognize your drawings. Your computer friend is like a clever robot, and you're the mentor.

🧠 **Watch It Learn**: As you train your model, it gets smarter with every lesson. Imagine your computer buddy growing wiser with each session. It's like having a digital pet that's eager to learn from you.

📚 **Your Art, Its Knowledge**: Remember those colorful numbers you drew in the "Create Dataset" section? They become the lessons your model studies. It's like sharing your creations with a digital friend.

🌟 **Magical Progress**: Watch as your computer model transforms. It's like magic, but it's really the power of your creativity and your computer working together.
  
 🤖 **epoch** is like a special round of practice for your computer friend. It's when your friend looks at all the drawings you've made and learns from them. Think of it as one big step in training your computer to be super smart!" 📚🚀

Are you ready to teach your computer friend some tricks? Let's embark on the training adventure and see your model grow before your eyes! 🎩📝🤖

            
            """)

# User input for the number of epochs
num_epochs = st.sidebar.number_input("Number of Epochs:", min_value=1, value=5, step=1)

if st.button("**Train Model**"):
    st.info("Training the model...")

    # Define your training dataset and data loader
    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        with st.empty():
            progress_text = st.empty()
            progress_bar = st.progress(0)

            progress_text.text(f"Training Epoch {epoch + 1}")

            for i, (image_batch, label_batch) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(image_batch)
                loss = criterion(outputs, label_batch)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                # Update the progress bar
                progress = (i + 1) / len(train_loader)
                progress_bar.progress(progress)

            st.success(f"Training complete for Epoch {epoch + 1}")
