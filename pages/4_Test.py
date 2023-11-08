import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os

st.set_page_config(page_title="Home", layout='wide', page_icon='./images/object.png')

#st.title("Test")

st.markdown("""
           
            # **Time for a Fun Challenge!**

🤗 Welcome to the "Test" section, where the real excitement begins! It's time to see how well your computer friend has learned from your colorful creations.

🖌️ **Your Turn to Challenge**: In this area, you get to be the quizmaster. Draw numbers and challenge your computer friend to guess them correctly. It's like a friendly showdown between you and your digital buddy.

🤖 **Let's Play and Learn**: The more you test, the smarter your computer model becomes. It's like playing a game and learning at the same time. Each drawing you test is a step closer to mastering this exciting adventure.

🎯 **Guess and Giggle**: Can your computer friend guess your drawings? Let's find out! It's not just a test; it's a fun adventure full of surprises.

Ready to play, challenge, and learn together? Let's start the exciting journey in this playful world of numbers and imagination! 🚀🎉🎨

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
    key="canvas",
)

if st.button("**Predict**"):
    with st.spinner("Saving..."):
        try:
            canvas_img = Image.fromarray(canvas_result.image_data)
            canvas_img.save(image_filepath, "PNG")
            st.success(f"Drawing saved as '{image_filename}'")
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

# Define your batch size
batch_size = 64

# Create a model instance
model = SimpleCNN()

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    img_tensor = load_and_preprocess_image(uploaded_image)
    img_tensor = img_tensor.unsqueeze(0)

    prediction = model(img_tensor)
    predicted_digit = torch.argmax(prediction, dim=1).item()

    st.write(f"Predicted Digit: {predicted_digit}")

#st.sidebar.write("Train a simple model for digit classification.")

st.markdown("""🌟 🌟 **It's Not Always About Being Perfect!** 🌟

Remember, your computer friend is like a clever detective trying its best to guess your drawings. Sometimes it'll get it right, and sometimes it might need a little help. It's all part of the adventure! Just enjoy the process, and let's see how close your computer buddy can get. 🕵️‍♂️🎨🤖

"""
)