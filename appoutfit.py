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
import csv

# Set up the Streamlit app
st.set_page_config(layout="wide")
st.header('✨ Fashion Outfit Recommendation System ✨')

# Load image features and filenames
@st.cache_data
def load_image_data():
    try:
        image_features = pkl.load(open('Images_features.pkl', 'rb'))
        filenames = pkl.load(open('filenames.pkl', 'rb'))
        file_ids = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
        return image_features, file_ids
    except Exception as e:
        st.error(f"Error loading image data: {str(e)}")
        return None, None

image_features, file_ids = load_image_data()

# Load metadata
@st.cache_data
def load_metadata():
    try:
        metadata = pd.read_csv('styles.csv', dtype={'id': str}, on_bad_lines='warn', encoding='utf-8')
        metadata['image_path'] = 'images/' + metadata['id'] + '.jpg'
        return metadata
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return pd.DataFrame()

metadata = load_metadata()
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

def get_outfit_recommendations(uploaded_id, uploaded_meta):
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

    # Define what to recommend based on input type
    input_type = uploaded_meta['articleType'].lower()
    
    recommendations = {}
    
    # For T-shirts/Tops, recommend bottoms and footwear
    if any(t in input_type for t in ['t-shirt', 'top', 'shirt']):
        # Bottoms (jeans, trousers, shorts, etc.)
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
        
        # Footwear (shoes, sandals, etc.)
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
            
        # Outerwear (jackets, coats, etc.)
        outerwear = [
            (id, id_to_meta[id]['articleType'], id_to_meta[id]['baseColour'])
            for id in similar_ids
            if id in id_to_meta and 
            any(o in id_to_meta[id]['articleType'].lower() 
               for o in ['jacket', 'coat', 'blazer', 'sweater'])
            and id_to_meta[id]['gender'] == uploaded_meta['gender']
        ]
        if outerwear:
            recommendations["Recommended Outerwear"] = outerwear[:4]
    
    return recommendations

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

# Main UI
upload_file = st.file_uploader("Upload your t-shirt/top image", type=['jpg', 'jpeg', 'png'])

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
        st.image(upload_file, width=300)
    
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
                            caption = f"{meta['articleType']}\n{meta.get('baseColour', '')}"
                            st.image(img, caption=caption, use_column_width=True)
                        except:
                            st.error(f"Couldn't load image {item_id}")
    else:
        with col2:
            st.subheader("Item Details")
            st.write(f"Type: {uploaded_meta['articleType']}")
            st.write(f"Color: {uploaded_meta.get('baseColour', 'Unknown')}")
            st.write(f"Gender: {uploaded_meta.get('gender', 'Unknown')}")
            st.write(f"Style: {uploaded_meta.get('productDisplayName', 'Unknown')}")
        
        # Get recommendations
        recommendations = get_outfit_recommendations(uploaded_id, uploaded_meta)
        
        if not recommendations:
            st.warning("No recommendations found for this item type")
        else:
            st.subheader("Outfit Recommendations")
            for section, items in recommendations.items():
                st.write(f"### {section}")
                cols = st.columns(len(items))
                for col, (item_id, item_type, color) in zip(cols, items):
                    with col:
                        try:
                            img = Image.open(get_image_path(item_id))
                            st.image(img, caption=f"{item_type}\n{color}", use_column_width=True)
                        except:
                            st.error(f"Couldn't load image {item_id}")

# Add some styling
st.markdown("""
<style>
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .stImage img:hover {
        transform: scale(1.03);
    }
    [data-testid="stFileUploader"] {
        width: 80%;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)""""""