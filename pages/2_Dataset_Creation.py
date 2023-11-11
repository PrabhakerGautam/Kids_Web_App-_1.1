import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import time


st.set_page_config(page_title="1.Dataset Creation",
                   layout='wide',
                   page_icon='./images/palette.png')
# Create a Streamlit app
st.markdown(" # **Welcome to dataset creation !** ")


            
st.markdown("""
  #### Let's teach your computer to recognize different numbers together! 🎨
          
- ##### 🖌️📚   Adjust your brush,
- ##### draw numbers, and  
- ##### create a unique dataset by saving the drawings.
""")


# Get user input for the label (0 to 9)
st.write(" #### 1. Enter a digit (0 to 9):")
user_input_label = st.number_input( "",min_value=0, max_value=9, step=1)
st.markdown(" ### 2. Now, draw the digit that you have entered above (0 to 9):")


# Specify canvas parameters in the application
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Adjust your brush",1,25,10) 
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

# Create the specified label folder if it does not exist
label_folder = os.path.join(dataset_folder, str(user_input_label))
os.makedirs(label_folder, exist_ok=True)

st.write(" ### 3. Finally,  Save that's all! for Now")

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
st.write("#### Next Step, Model Training")
st.markdown("[Click here to begin the training process](/Train/)")


   
 



 