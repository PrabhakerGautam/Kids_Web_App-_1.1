# streamlit_app.py
import streamlit as st
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="About",
                   layout='wide',
                   page_icon='./images/about.png')
def about_section():
    st.subheader("About the App")
    st.write(
        "#### Welcome to the **AI Doodle Recognizer**! 🎨✨ "
        "\n\n"
        "#### This app allows you to draw a digit, and the AI model will predict what digit you've drawn."
        "\n\n"
        "#### You can use the drawing canvas to create your digit masterpiece. After drawing, click the **Predict** button to see the model's prediction."
        "\n\n"
        "#### Feel free to provide feedback on the prediction accuracy in the text box below the prediction. Your input helps improve the model!"
    )
    st.markdown(
        "[GitHub Repository](https://github.com/PrabhakerGautam/Kids_Web_App-_1.1) | "
        "[Report an Issue](https://github.com/PrabhakerGautam/Kids_Web_App-_1.1/issues)"
    )
    st.write("---")



# Include this function call at the beginning of your app
about_section()
