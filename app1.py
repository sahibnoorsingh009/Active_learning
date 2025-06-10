# robust_active_learning.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(
    page_title="Robust Active Learning",
    page_icon="🛡️",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

X, y, feature_names, target_names = load_data()

# Active Learning Strategies
def get_sample(strategy, model, unlabeled_indices, X_pool):
    if strategy == "Random":
        return np.random.choice(unlabeled_indices)
    
    probs = model.predict_proba(X_pool[unlabeled_indices])
    
    if strategy == "Uncertainty":
        uncertainty = 1 - np.max(probs, axis=1)
        return unlabeled_indices[np.argmax(uncertainty)]
    
    elif strategy == "Margin":
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return unlabeled_indices[np.argmin(margins)]
    
    elif strategy == "Entropy":
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return unlabeled_indices[np.argmax(entropy)]

# Initialize Session State
if 'experiment' not in st.session_state:
    st.session_state.experiment = {
        'strategies': ["Random", "Uncertainty", "Margin", "Entropy"],
        'results': {s: {'accuracy': [], 'labels': []} for s in ["Random", "Uncertainty", "Margin", "Entropy"]},
        'current_strategy': "Uncertainty",
        'model': None,
        'labeled_idx': [],
        'labeled_y': [],
        'unlabeled_idx': list(range(len(X))),
        'min_classes': 2  # Minimum classes required for training
    }

# UI Controls
st.sidebar.header("Controls")
strategy = st.sidebar.selectbox(
    "Strategy",
    st.session_state.experiment['strategies'],
    index=st.session_state.experiment['strategies'].index(
        st.session_state.experiment['current_strategy']
    )
)

if st.sidebar.button("Initialize Experiment"):
    # Start with at least one sample per class
    st.session_state.experiment['labeled_idx'] = []
    st.session_state.experiment['labeled_y'] = []
    
    for class_idx in range(len(target_names)):
        sample_idx = np.where(y == class_idx)[0][0]
        st.session_state.experiment['labeled_idx'].append(sample_idx)
        st.session_state.experiment['labeled_y'].append(class_idx)
    
    st.session_state.experiment['unlabeled_idx'] = [
        i for i in range(len(X)) 
        if i not in st.session_state.experiment['labeled_idx']
    ]
    st.session_state.experiment['current_strategy'] = strategy
    st.session_state.experiment['model'] = LogisticRegression(max_iter=1000)
    st.rerun()

# Main Experiment Logic
if st.session_state.experiment['model']:
    # Check for minimum classes
    unique_classes = np.unique(st.session_state.experiment['labeled_y'])
    
    if len(unique_classes) >= st.session_state.experiment['min_classes']:
        try:
            # Train model
            st.session_state.experiment['model'].fit(
                X[st.session_state.experiment['labeled_idx']],
                st.session_state.experiment['labeled_y']
            )
            
            # Record performance
            acc = accuracy_score(y, st.session_state.experiment['model'].predict(X))
            st.session_state.experiment['results'][strategy]['accuracy'].append(acc)
            st.session_state.experiment['results'][strategy]['labels'].append(
                len(st.session_state.experiment['labeled_idx'])
            )
            
            # Get next sample
            if st.session_state.experiment['unlabeled_idx']:
                new_idx = get_sample(
                    strategy,
                    st.session_state.experiment['model'],
                    st.session_state.experiment['unlabeled_idx'],
                    X
                )
                st.session_state.experiment['labeled_idx'].append(new_idx)
                st.session_state.experiment['labeled_y'].append(y[new_idx])
                st.session_state.experiment['unlabeled_idx'].remove(new_idx)
                st.rerun()
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
    else:
        st.warning(f"Need at least {st.session_state.experiment['min_classes']} classes to train")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
for strategy_name, data in st.session_state.experiment['results'].items():
    if data['accuracy']:
        ax.plot(data['labels'], data['accuracy'], 
                label=strategy_name, marker='o')

ax.set_xlabel("Number of Labeled Samples")
ax.set_ylabel("Accuracy")
ax.set_title("Active Learning Strategy Comparison")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Status Display
st.subheader("Experiment Status")
col1, col2 = st.columns(2)
with col1:
    st.metric("Labeled Samples", len(st.session_state.experiment['labeled_idx']))
    st.metric("Unique Classes", len(np.unique(st.session_state.experiment['labeled_y'])))
    
with col2:
    if st.session_state.experiment['results'][strategy]['accuracy']:
        st.metric(
            "Current Accuracy", 
            f"{st.session_state.experiment['results'][strategy]['accuracy'][-1]:.1%}"
        )
    else:
        st.metric("Current Accuracy", "N/A")

# Data Inspection
with st.expander("Debug Info"):
    st.write("Labeled indices:", st.session_state.experiment['labeled_idx'])
    st.write("Current labels:", st.session_state.experiment['labeled_y'])
    st.write("Unique classes:", np.unique(st.session_state.experiment['labeled_y']))