import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

st.set_page_config(page_title="Home", layout='wide', page_icon='./images/object.png')

st.title("Let's Generate DataSet for Computer Model")
st.write("Try to save at least 5 images for your particular digit")

drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color: ")
bg_color = st.sidebar.color_picker("Background color: ", "#eee")

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

dataset_folder = "test"  # Change to your desired folder name
os.makedirs(dataset_folder, exist_ok=True)

user_input_label = st.number_input("Enter a label (0 to 9):", min_value=0, max_value=9, step=1)

label_folder = os.path.join(dataset_folder, str(user_input_label))
os.makedirs(label_folder, exist_ok=True)

if st.button("Save Drawing"):
    with st.spinner("Saving..."):
        try:
            canvas_img = Image.fromarray(canvas_result.image_data)
            filename = f"image_{len(os.listdir(label_folder))}.png"
            file_path = os.path.join(label_folder, filename)
            canvas_img.save(file_path, "PNG")
            st.success(f"Drawing saved as '{file_path}' with label '{user_input_label}'")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Provide a link to the training app
st.markdown("[Click here to go to the training app](/Train/)")
