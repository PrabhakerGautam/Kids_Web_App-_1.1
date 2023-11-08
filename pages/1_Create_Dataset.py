import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import time


st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/object.png')
# Create a Streamlit app
st.markdown(" # **Let's Generate DataSet for Computer Model** ")


            
st.write("   Welcome to the *Create Dataset* section, where you become the teacher and artist! Here, you can adjust your brush stroke on the left sidebar and start drawing numbers. Each drawing you create becomes a part of your very own dataset. These drawings will help teach your computer model to recognize different numbers. So, start drawing and let's create an amazing dataset together! 🎨📚")



# Specify canvas parameters in the application
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
stroke_color = st.sidebar.color_picker("Stroke color: ")
bg_color = st.sidebar.color_picker("Background color: ", "#eee")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Create a folder named 'dataset' if it does not exist
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

# Get user input for the label (0 to 9)
user_input_label = st.number_input("Enter a label (0 to 9):", min_value=0, max_value=9, step=1)

# Create the specified label folder if it does not exist
label_folder = os.path.join(dataset_folder, str(user_input_label))
os.makedirs(label_folder, exist_ok=True)

# Create a button to save the drawing with the specified label
if st.button("**Save Drawing**"):
    with st.spinner("Saving..."):
        try:
            # Get the Pillow image from the canvas component
            canvas_img = Image.fromarray(canvas_result.image_data)

            # Generate a unique filename
            filename = f"image_{len(os.listdir(label_folder))}.png"

            # Save the image to the specified label folder
            file_path = os.path.join(label_folder, filename)
            canvas_img.save(file_path, "PNG")

            st.success(f"Drawing saved as '{file_path}' with label '{user_input_label}'")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Adjust the path to match the saved images


# Content
st.markdown("""
  
- For Training [Click here for App](/Train/)  
  



           
            """)

