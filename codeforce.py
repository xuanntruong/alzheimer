import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import streamlit as st
import gdown


# Specify the Google Drive file ID
gdrive_file_id = '1u7PMM6CG3WmWWQ5VWUzIuYgM3ObJgHzH'  # Replace with your actual file ID
gdrive_model_url = f'https://drive.google.com/uc?id={gdrive_file_id}'

# Download the model if it doesn't exist
model_path = 'model.h5'
gdown.download(gdrive_model_url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Sidebar Navigation with Icons
st.sidebar.title("")
options = st.sidebar.radio("", [
    "🏠 Model",       # Main model page with home icon
    "📜 Certificate", # Certificate page with scroll icon
    "❓ FAQ",         # FAQ page with question mark icon
    "📞 Contact",     # Contact page with phone icon
    "ℹ️ About",       # About page with info icon
    "💎 Pricing"      # New Pricing page with diamond icon
])

# Main Page - Model for Alzheimer's Detection
if options == "🏠 Model":
    st.title("Alzheimer's Disease Detection with Explainable AI")
    st.write("Upload an MRI scan to visualize the saliency map.")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        # Load the image and preprocess
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((128, 128))
        image_array = np.array(image)
        image_array = cv2.GaussianBlur(image_array, (5, 5), 0)  # Apply Gaussian blur (noise cancellation)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize
        img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

        # Predict class
        predictions = model(img_tensor)
        predicted_class = np.argmax(predictions[0])
        prediction_percentages = predictions[0] * 100
        stages = ['Moderate Alzheimer', 'Mild Alzheimer', 'Non Alzheimer', 'Very mild Alzheimer']
        
        # Compute saliency map
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            class_output = predictions[0, predicted_class]

        grads = tape.gradient(class_output, img_tensor)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)  # Maximum over the color channels
        saliency = saliency[0].numpy()

        # Overlay saliency on the original image
        image = Image.open(uploaded_file).convert('RGB')
        saliency = np.uint8(255 * saliency)
        saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2RGB)
        saliency = Image.fromarray(saliency).resize(image.size)
        saliency = np.array(saliency)
        alpha = 0.4
        superimposed_img = saliency * alpha + np.array(image)
        saliency_image = Image.fromarray(np.uint8(superimposed_img))

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Model Prediction")
            for i, stage in enumerate(stages):
                st.write(f"{stage}: {prediction_percentages[i]:.2f}%")
        with col2:
            st.write("### Saliency Map")
            st.image(saliency_image, caption='Saliency Map Overlay', use_column_width=True)
            st.write("Saliency map overlay completed.")

# Certificate Page
elif options == "📜 Certificate":
    st.title("Certificate of Completion")
    st.write("This page is for generating and displaying user certificates upon completing the diagnostic test.")
    st.write("To be developed: certificate generation functionality.")

# FAQ Page
elif options == "❓ FAQ":
    st.title("Frequently Asked Questions")
    st.write("Find answers to the most common questions below:")

    st.write("**Q1: What is Alzheimer's disease?**")
    st.write("Alzheimer's disease is a progressive neurological disorder that causes memory loss and cognitive decline.")
    
    st.write("**Q2: How does this model work?**")
    st.write("Our AI model analyzes MRI scans to detect the potential stage of Alzheimer's disease using deep learning.")

    st.write("**Q3: What is a saliency map?**")
    st.write("A saliency map highlights the regions of the MRI scan that contribute most to the model's prediction, providing explainability.")

# Contact Page
elif options == "📞 Contact":
    st.title("Contact Us")
    st.write("Feel free to reach out for any inquiries or support.")
    st.write("**Email:** levuxuantruong2008@gmail.com")
    st.write("**Phone:** +84898183708")
    st.write("**Address:** 235 Nguyen Van Cu, ward 4, district 5, Ho Chi Minh city, Vietnam")

# About Page
elif options == "ℹ️ About":
    st.title("About This Project")
    st.write("This project aims to use AI to assist in detecting the stages of Alzheimer's disease from MRI scans.")
    st.write("It is designed to help healthcare professionals with an explainable AI model, which uses a saliency map to show the model's attention on MRI scans.")
    st.write("Developed by: Alzheimer-detecting startup")

# Pricing Page
elif options == "💎 Pricing":
    st.title("Premium Membership Pricing")
    st.write("Choose a plan that suits your needs and unlock exclusive features with our premium membership.")
    
    # Define pricing plans
    st.write("### Membership Tiers")
    st.write("#### 🥈 Basic Plan - Free")
    st.write("- Access to basic diagnostic features")
    st.write("- Limited report generation")
    
    st.write("#### 🥇 Premium Plan - $9.99/month")
    st.write("- Access to advanced diagnostic features")
    st.write("- Unlimited report generation")
    st.write("- Access to priority customer support")
    st.write("- Additional features such as custom report exports")
    st.write("- Direct access to AI model explainability tools")

    # Registration button for premium users
    if st.button("Register for Premium"):
        st.write("Thank you for choosing the Premium Plan! Our team will contact you shortly for further assistance.")
