# streamlit_app.py
import streamlit as st

# App title
st.title("Deep Learning Models Dashboard")

# Hide sidebar
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Create a container for the buttons
button_container = st.container()

# Display buttons for different models
def navigate_to_page(page_name):
    st.experimental_set_query_params(page=page_name)

button_container.button("Text Classification", on_click=lambda: navigate_to_page("text_classification"), use_container_width=True)
button_container.button("Drug Classification", on_click=lambda: navigate_to_page("pill_classification"), use_container_width=True)
button_container.button("Insurance Cost Prediction", on_click=lambda: navigate_to_page("insurance_prediction"), use_container_width=True)
button_container.button("Image Segmentation", on_click=lambda: navigate_to_page("image_segmentation"), use_container_width=True)
button_container.button("Plot based Title Generation", on_click=lambda: navigate_to_page("netflix_title_generator"), use_container_width=True)

# Get the current page from query params
query_params = st.experimental_get_query_params()
page = query_params.get("page", [""])[0]

# Render the selected page
if page == "text_classification":
    exec(open("pages/text_classification.py").read())
elif page == "pill_classification":
    exec(open("pages/pill_classification.py").read())
elif page == "insurance_prediction":
    exec(open("pages/insurance_prediction.py").read())
elif page == "image_segmentation":
    exec(open("pages/image_segmentation.py").read())
elif page == "netflix_title_generator":
    exec(open("pages/netflix_title_generator.py").read())