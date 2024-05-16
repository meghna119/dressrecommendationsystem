import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Define the RGB values for each skin tone class (MST)
skin_tone_rgb_values = {
    1: [246, 237, 228],  
    2: [243, 231, 219],   
    3: [247, 234, 208],  
    4: [234, 218, 186],  
    5: [215, 189, 150],  
    6: [160, 126, 86],   
    7: [130, 92, 67],    
    8: [96, 65, 52],   
    9: [58, 49, 42],    
    10: [41, 36, 32]    
}

bright_colors = {
    1: [(255,0,255), (255,247,0), (0,245,255), (255,153,51), (255,0,144), (0,255,0), (220,20,60), (0,255,255), (255,105,180), (218,165,32)],
    2: [(255,153,51), (255,0,144), (0,255,0), (255,0,0), (0,245,255), (255,0,255), (220,20,60), (0,255,255), (255,105,180), (218,165,32)],
    3: [(255,191,0), (255,0,255), (0,128,128), (220,20,60), (0,255,255), (255,153,51), (255,105,180), (0,255,0), (255,0,144), (218,165,32)],
    4: [(220,20,60), (218,165,32), (15,82,186), (255,0,255), (0,255,0), (255,153,51), (255,0,144), (0,128,128), (0,255,255), (255,105,180)],
    5: [(255,99,71), (238,130,238), (127,255,0), (255,0,255), (220,20,60), (0,255,255), (255,153,51), (0,255,0), (255,105,180), (15,82,186)],
    6: [(215,0,64), (255,219,88), (64,224,208), (255,153,51), (255,0,255), (220,20,60), (0,255,255), (0,255,0), (255,105,180), (218,165,32)],
    7: [(255,36,0), (218,165,32), (51,161,201), (255,0,255), (255,153,51), (220,20,60), (0,255,255), (0,255,0), (255,105,180), (15,82,186)],
    8: [(227,36,36), (244,196,48), (80,200,120), (255,0,255), (255,153,51), (220,20,60), (0,255,255), (0,255,0), (255,105,180), (15,82,186)],
    9: [(255,69,0), (255,105,180), (15,82,186), (255,0,255), (255,153,51), (220,20,60), (0,255,255), (0,255,0), (255,0,144), (218,165,32)],
    10: [(255,153,51), (255,0,255), (65,105,225), (220,20,60), (0,255,255), (0,255,0), (255,105,180), (15,82,186), (255,0,144), (218,165,32)]
}

soft_colors = {
    1: [(255,211,211), (204,204,255), (180,255,180), (255,218,185), (230,230,250), (160,255,175), (255,228,225), (176,224,230), (201,160,220), (188,223,153)],
    2: [(255,218,185), (230,230,250), (160,255,175), (255,211,211), (180,255,180), (255,228,225), (176,224,230), (201,160,220), (224,176,255), (188,223,153)],
    3: [(255,215,180), (224,176,255), (188,223,153), (255,218,185), (230,230,250), (255,228,225), (180,255,180), (201,160,220), (176,224,230), (181,166,117)],
    4: [(255,228,225), (185,121,119), (176,224,230), (255,218,185), (230,230,250), (224,176,255), (180,255,180), (201,160,220), (188,223,153), (181,166,117)],
    5: [(255,127,80), (200,162,235), (245,255,250), (255,218,185), (255,228,225), (176,224,230), (224,176,255), (201,160,220), (188,223,153), (181,166,117)],
    6: [(210,105,30), (218,112,214), (188,223,153), (255,218,185), (255,228,225), (230,230,250), (245,255,250), (224,176,255), (176,224,230), (181,166,117)],
    7: [(226,114,91), (201,160,220), (160,185,128), (255,218,185), (255,228,225), (230,230,250), (245,255,250), (224,176,255), (176,224,230), (181,166,117)],
    8: [(226,114,91), (201,160,220), (160,185,128), (255,218,185), (255,228,225), (230,230,250), (245,255,250), (224,176,255), (176,224,230), (181,166,117)],
    9: [(184,115,51), (221,160,221), (181,166,117), (255,218,185), (255,228,225), (230,230,250), (245,255,250), (224,176,255), (201,160,220), (188,223,153)],
    10: [(210,105,30), (153,102,204), (160,185,128), (255,218,185), (255,228,225), (230,230,250), (245,255,250), (224,176,255), (201,160,220), (181,166,117)]
}

def detect_skin_tone(image):
    # Convert the image to 2D array
    image_array = np.array(image)
    X = image_array.reshape(-1, 3)

    # Perform K-Means clustering on the entire image
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    skin_tone_clusters = kmeans.cluster_centers_

    # Find the closest skin tone cluster to the image pixels
    closest_skin_tone = pairwise_distances_argmin_min(skin_tone_clusters, [skin_tone_rgb_values[i] for i in skin_tone_rgb_values])[0][0]

    return closest_skin_tone   

# Streamlit GUI
st.title("Color Analysis")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    detected_skin_tone = detect_skin_tone(image)
    st.write("Detected skin tone class:", detected_skin_tone)
    
    # Display the color palettes for the detected skin tone
    st.subheader("Color Palettes for Detected Skin Tone")
    col1, col2 = st.columns(2)

    # Display the bright color palette in the first column
    with col1:
        st.subheader("Bright Color Palette")
        color_palette_bright = bright_colors.get(detected_skin_tone, [])
        for color in color_palette_bright:
            st.write(f'<div style="background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: 50px; height: 50px; display: inline-block;"></div>', unsafe_allow_html=True)

    # Display the soft color palette in the second column
    with col2:
        st.subheader("Soft Color Palette")
        color_palette_soft = soft_colors.get(detected_skin_tone, [])
        for color in color_palette_soft:
            st.write(f'<div style="background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: 50px; height: 50px; display: inline-block;"></div>', unsafe_allow_html=True)

