import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from sklearn.model_selection import train_test_split
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os
import time
st.set_page_config(page_title="3.Test", layout='wide', page_icon='./images/target.png')

#st.title("Test")

st.markdown("""
           
            # **Time for a Fun Challenge!**

##### 🤗 Welcome to the "Predict" section, where the real excitement begins! It's time to see how well your computer friend has learned from your colorful creations.

##### 🖌️ **Your Turn to Challenge**: In this area, you get to be the quizmaster. Draw numbers and challenge your computer friend to guess them correctly. It's like a friendly showdown between you and your digital buddy.

##### Ready to play, challenge, and learn together? Let's start the exciting journey in this playful world of numbers and imagination! 🚀🎉🎨

"""
)

drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
stroke_color = st.sidebar.color_picker("Stroke color: ")
bg_color = st.sidebar.color_picker("Background color: ", "#eee")


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

    
    
# Create a unique filename for the saved image
image_filename = "saved_image.png"
image_filepath = os.path.join("test", image_filename)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=400,
    drawing_mode=drawing_mode,
    key="canvas"
)

if st.button("**Predict**"):
    with st.spinner("Predicting..."):
        try:
            canvas_img = Image.fromarray(canvas_result.image_data)
            canvas_img.save(image_filepath, "PNG")
            #st.success(f"Drawing saved as '{image_filename}'")
            progress_bar = st.progress(0)
            for percent_complete in range(0, 101, 10):
                progress_bar.progress(percent_complete)
                time.sleep(0.2)
        except Exception as e:
            st.error(f"Error: {str(e)}")

uploaded_image = f"./test/{image_filename}"



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

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define your batch size
batch_size = 64

# Create a model instance
model = SimpleCNN()

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
import time
# Training loop
epochs = 10
for epoch in range(epochs):
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
     
    #st.write(f'Epoch {epoch+1}/{epochs}, Loss: {val_loss:.4f}, Accuracy: {accuracy*100:.2f}%')





if uploaded_image is not None:
    #st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    img_tensor = load_and_preprocess_image(uploaded_image)
    img_tensor = img_tensor.unsqueeze(0)

    prediction = model(img_tensor)
    predicted_digit = torch.argmax(prediction, dim=1).item()

    st.write(f" #### Predicted Digit: {predicted_digit}")


st.markdown("""🌟 🌟 **It's Not Always About Being Perfect!** 🌟

Remember, your computer friend is like a clever detective trying its best to guess your drawings. Sometimes it'll get it right, and sometimes it might need a little help. It's all part of the adventure! Just enjoy the process, and let's see how close your computer buddy can get. 🕵️‍♂️🎨🤖

"""
)