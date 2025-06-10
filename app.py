# active_learning_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuration
st.set_page_config(
    page_title="Active Learning Demo",
    page_icon="🌱",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

X, y, feature_names, target_names = load_data()

# Session State Initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'labeled_idx' not in st.session_state:
    # Initialize with at least one sample per class
    st.session_state.labeled_idx = []
    st.session_state.labeled_y = []
    st.session_state.unlabeled_idx = list(range(len(X)))
    
    # Add one sample of each class to start
    for class_idx in range(len(target_names)):
        sample_idx = np.where(y == class_idx)[0][0]
        st.session_state.labeled_idx.append(sample_idx)
        st.session_state.labeled_y.append(class_idx)
        st.session_state.unlabeled_idx.remove(sample_idx)
        
if 'history' not in st.session_state:
    st.session_state.history = {
        'accuracy': [],
        'labels_added': []
    }

# Model Training
def train_model():
    if len(np.unique(st.session_state.labeled_y)) < 2:
        return None
        
    model = LogisticRegression(max_iter=1000)
    # model = RandomForestClassifier()  # Alternative more robust model
    model.fit(X[st.session_state.labeled_idx], 
             np.array(st.session_state.labeled_y))
    return model

# Active Learning Sampling
def get_sample_to_label():
    if not st.session_state.unlabeled_idx:
        return None
        
    if st.session_state.model:
        probs = st.session_state.model.predict_proba(X[st.session_state.unlabeled_idx])
        uncertainty = 1 - np.max(probs, axis=1)
        sample_pos = np.argmax(uncertainty)
    else:
        sample_pos = np.random.randint(0, len(st.session_state.unlabeled_idx))
        
    return st.session_state.unlabeled_idx[sample_pos]

# UI Layout
st.title("🌻 Interactive Active Learning")
st.write("Teach a model by labeling the most uncertain Iris samples")

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()
    
    st.metric("Labeled Samples", len(st.session_state.labeled_idx))
    st.metric("Unlabeled Samples", len(st.session_state.unlabeled_idx))
    
    if len(np.unique(st.session_state.labeled_y)) < 2:
        st.warning("Need samples from at least 2 classes")

# Main Interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Sample")
    current_sample_idx = get_sample_to_label()
    
    if current_sample_idx is not None:
        sample = X[current_sample_idx]
        df_sample = pd.DataFrame([sample], 
                               columns=feature_names,
                               index=["Value"])
        st.dataframe(df_sample.style.format("{:.2f}"))
        
        label = st.radio("Select class:", 
                        options=target_names,
                        key=f"label_{current_sample_idx}")
        
        if st.button("Submit Label"):
            # Update data pools
            st.session_state.labeled_idx.append(current_sample_idx)
            st.session_state.labeled_y.append(np.where(target_names == label)[0][0])
            st.session_state.unlabeled_idx.remove(current_sample_idx)
            
            # Retrain and record accuracy
            st.session_state.model = train_model()
            if st.session_state.model:
                acc = accuracy_score(y, st.session_state.model.predict(X))
                st.session_state.history['accuracy'].append(acc)
                st.session_state.history['labels_added'].append(len(st.session_state.labeled_idx))
            
            st.rerun()
    else:
        st.success("All samples labeled!")
        st.balloons()

with col2:
    st.subheader("Learning Progress")
    if len(st.session_state.history['accuracy']) > 0:
        chart_data = pd.DataFrame({
            'Accuracy': st.session_state.history['accuracy'],
            'Labels Added': st.session_state.history['labels_added']
        })
        st.line_chart(chart_data.set_index('Labels Added'))
    
    if st.session_state.model:
        st.write(f"Current Accuracy: {st.session_state.history['accuracy'][-1]:.1%}")
    else:
        st.info("Label samples from at least 2 classes to train model")

# Debug Info (can be removed)
with st.expander("Debug Info"):
    st.write("Labeled indices:", st.session_state.labeled_idx)
    st.write("Current labels:", st.session_state.labeled_y)
    st.write("Unique classes:", np.unique(st.session_state.labeled_y))
