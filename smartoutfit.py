import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
import pandas as pd
from PIL import Image

# Set up the Streamlit app with custom theme
st.set_page_config(layout="wide", page_title="✨ Smart Outfit Recommender ✨")

# Custom CSS styling with cream background and dark green/pink color scheme
st.markdown("""
<style>
    .stApp {
        background-color: pink;
    }
    body {
        background-color: pink;
    }
    figure > figcaption {
        color: black;
        font-weight: bold;
    }
    div
    {
        color:black;        
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: black;
    }
    /* Upload button background and text */
    [data-testid="stFileUploader"] > div:first-child {
        background-color: #FFFDD0; /* Light cream color */
        color: black; /* Black text inside upload button */
        border-radius: 10px;
        padding: 10px;
    }
    :root {
        --light-pink: #FFE4E1;
        --soft-pink: #FFB6C1;
        --hot-pink: #FF69B4;
        --white: #FFFFFF;
    }
    body {
        background-color: var(--light-pink);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--hot-pink) !important;
        border-bottom: 2px solid var(--soft-pink);
        padding-bottom: 8px;
    }
    .stButton>button {
        background-color: var(--hot-pink) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        background-color: var(--soft-pink) !important;
    }
    .stImage img {
        border-radius: 12px;
        width: 200px;
        height: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    [data-testid="stFileUploader"] {
        background-color: var(--white);
        border: 2px dashed var(--hot-pink);
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.header('✨ Smart Outfit Recommender TRY IT ✨')

# Load data
@st.cache_data
def load_data():
    image_features = pkl.load(open('Images_features.pkl', 'rb'))
    filenames = pkl.load(open('filenames.pkl', 'rb'))
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    
    metadata = pd.read_csv('styles.csv', dtype={'id': str}, on_bad_lines='warn')
    metadata['image_path'] = 'images/' + metadata['id'] + '.jpg'
    
    return image_features, file_ids, metadata

image_features, file_ids, metadata = load_data()
id_to_meta = {row['id']: row for _, row in metadata.iterrows()}

# Initialize model
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tf.keras.models.Sequential([model, GlobalMaxPool2D()])

model = load_model()

# Initialize NearestNeighbors
@st.cache_resource
def init_neighbors(features):
    neighbors = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean')
    neighbors.fit(features)
    return neighbors

neighbors = init_neighbors(image_features)

def get_image_path(item_id):
    return f"images/{item_id}.jpg"

def extract_features_from_images(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_expand_dim = np.expand_dims(img_array, axis=0)
        img_preprocess = preprocess_input(img_expand_dim)
        result = model.predict(img_preprocess).flatten()
        return result / norm(result)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_recommendations(uploaded_id, uploaded_meta):
    # Find index of uploaded item
    try:
        idx = file_ids.index(uploaded_id)
        input_features = image_features[idx]
    except ValueError:
        upload_path = get_image_path(uploaded_id)
        if not os.path.exists(upload_path):
            st.error("Uploaded image not found")
            return {}
        input_features = extract_features_from_images(upload_path, model)
        if input_features is None:
            return {}

    # Get similar items
    distances, indices = neighbors.kneighbors([input_features])
    similar_ids = [file_ids[i] for i in indices[0][1:]]
    
    # Determine item type
    item_type = uploaded_meta['articleType'].lower()
    
    recommendations = {}
    
    # For Topwear (T-shirts, shirts, etc.)
    if any(t in item_type for t in ['t-shirt', 'shirt', 'top', 'blouse', 'sweater']):
        # Recommend bottoms
        bottoms = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(b in id_to_meta[id]['articleType'].lower() 
               for b in ['jeans', 'trousers', 'pants', 'shorts', 'skirt'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if bottoms:
            recommendations["Recommended Bottoms"] = bottoms[:6]
        
        # Recommend footwear
        footwear = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(f in id_to_meta[id]['articleType'].lower() 
               for f in ['shoes', 'footwear', 'sandals', 'sneakers'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if footwear:
            recommendations["Recommended Footwear"] = footwear[:4]
    
    # For Bottomwear (Jeans, trousers, etc.)
    elif any(b in item_type for b in ['jeans', 'trousers', 'pants', 'shorts', 'skirt']):
        # Recommend tops
        tops = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(t in id_to_meta[id]['articleType'].lower() 
               for t in ['t-shirt', 'shirt', 'top', 'blouse', 'sweater'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if tops:
            recommendations["Recommended Tops"] = tops[:8]
        
        # Recommend footwear
        footwear = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(f in id_to_meta[id]['articleType'].lower() 
               for f in ['shoes', 'footwear', 'sandals', 'sneakers'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if footwear:
            recommendations["Recommended Footwear"] = footwear[:4]
    
    # For Footwear (Shoes, sandals, etc.)
    elif any(f in item_type for f in ['shoes', 'footwear', 'sandals', 'sneakers']):
        # Recommend bottoms
        bottoms = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(b in id_to_meta[id]['articleType'].lower() 
               for b in ['jeans', 'trousers', 'pants', 'shorts', 'skirt'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if bottoms:
            recommendations["Recommended Bottoms"] = bottoms[:6]
        
        # Recommend tops
        tops = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(t in id_to_meta[id]['articleType'].lower() 
               for t in ['t-shirt', 'shirt', 'top', 'blouse', 'sweater'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if tops:
            recommendations["Recommended Tops"] = tops[:6]
    
    # Common recommendations for all types
    # Outerwear
    outerwear = [
        (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
        for id in similar_ids
        if id in id_to_meta and 
        any(o in id_to_meta[id]['articleType'].lower() 
           for o in ['jacket', 'coat', 'blazer', 'cardigan'])
        and id_to_meta[id]['gender'] == uploaded_meta['gender']
    ]
    if outerwear:
        recommendations["Recommended Outerwear"] = outerwear[:3]
    
    # Accessories
    accessories = [
        (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
        for id in similar_ids
        if id in id_to_meta and 
        any(a in id_to_meta[id]['articleType'].lower() 
           for a in ['belt', 'bag', 'hat', 'scarf'])
        and id_to_meta[id]['gender'] == uploaded_meta['gender']
    ]
    if accessories:
        recommendations["Recommended Accessories"] = accessories[:3]
    
    return recommendations

# Main UI
st.subheader("Upload Your Fashion Item")
upload_file = st.file_uploader("Choose any clothing item", type=['jpg', 'jpeg', 'png'])

if upload_file is not None:
    uploaded_id = os.path.splitext(upload_file.name)[0]
    image_path = get_image_path(uploaded_id)
    
    # Save uploaded file if not exists
    if not os.path.exists(image_path):
        os.makedirs('images', exist_ok=True)
        with open(image_path, 'wb') as f:
            f.write(upload_file.getbuffer())
    
    # Display uploaded item
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(upload_file, width=300, caption="Your Item")
    
    # Get metadata
    uploaded_meta = id_to_meta.get(uploaded_id)
    if uploaded_meta is None:
        st.warning("No metadata found for this item. Showing similar items.")
        features = extract_features_from_images(image_path, model)
        if features is not None:
            distances, indices = neighbors.kneighbors([features])
            st.subheader("Similar Items")
            cols = st.columns(4)
            for i, col in zip(indices[0][1:5], cols):
                item_id = file_ids[i]
                if item_id in id_to_meta:
                    with col:
                        try:
                            img = Image.open(get_image_path(item_id))
                            meta = id_to_meta[item_id]
                            st.image(img, caption=meta['articleType'], use_container_width=True)
                        except:
                            st.error(f"Couldn't load image {item_id}")
    else:
        with col2:
            st.subheader("Item Details")
            st.write(f"**Type:** {uploaded_meta['articleType']}")
            st.write(f"**Color:** {uploaded_meta.get('baseColour', 'Unknown')}")
            st.write(f"**Gender:** {uploaded_meta.get('gender', 'Unknown')}")
            st.write(f"**Style:** {uploaded_meta.get('productDisplayName', 'Unknown')}")
        
        # Get recommendations
        recommendations = get_recommendations(uploaded_id, uploaded_meta)
        
        if not recommendations:
            st.warning("No recommendations found for this item type")
        else:
            st.subheader("Complete Outfit Suggestions")
            
            # Display smart combinations based on input type
            input_type = uploaded_meta['articleType'].lower()
            
            if any(t in input_type for t in ['t-shirt', 'shirt', 'top', 'blouse']):
                # For tops, show top + bottom + shoes combo
                if "Recommended Bottoms" in recommendations and "Recommended Footwear" in recommendations:
                    st.write("### Suggested Outfit Combination")
                    cols = st.columns(3)
                    with cols[0]:
                        st.image(upload_file, caption=f"Your {uploaded_meta['articleType']}", use_container_width=True)
                    with cols[1]:
                        if len(recommendations["Recommended Bottoms"]) > 0:
                            item_id, item_type, color = recommendations["Recommended Bottoms"][0]
                            st.image(get_image_path(item_id), caption=f"Bottom: {item_type}", use_container_width=True)
                    with cols[2]:
                        if len(recommendations["Recommended Footwear"]) > 0:
                            item_id, item_type, color = recommendations["Recommended Footwear"][0]
                            st.image(get_image_path(item_id), caption=f"Shoes: {item_type}", use_container_width=True)
            
            elif any(b in input_type for b in ['jeans', 'trousers', 'pants', 'skirt']):
                # For bottoms, show bottom + top + shoes combo
                if "Recommended Tops" in recommendations and "Recommended Footwear" in recommendations:
                    st.write("### Suggested Outfit Combination")
                    cols = st.columns(3)
                    with cols[0]:
                        if len(recommendations["Recommended Tops"]) > 0:
                            item_id, item_type, color = recommendations["Recommended Tops"][0]
                            st.image(get_image_path(item_id), caption=f"Top: {item_type}", use_container_width=True)
                    with cols[1]:
                        st.image(upload_file, caption=f"Your {uploaded_meta['articleType']}", use_container_width=True)
                    with cols[2]:
                        if len(recommendations["Recommended Footwear"]) > 0:
                            item_id, item_type, color = recommendations["Recommended Footwear"][0]
                            st.image(get_image_path(item_id), caption=f"Shoes: {item_type}", use_container_width=True)
            
            # Display all recommendations by category
            for section, items in recommendations.items():
                st.write(f"### {section}")
                cols = st.columns(len(items))
                for col, (item_id, item_type, color) in zip(cols, items):
                    with col:
                        try:
                            st.image(get_image_path(item_id), 
                                    caption=f"{item_type}\nColor: {color}", 
                                    use_container_width=True)
                        except:
                            st.error(f"Couldn't load image {item_id}")

# Footer
st.markdown("""
<div class="footer">
    ✨ Smart Outfit Recommender✨
</div>
""", unsafe_allow_html=True)