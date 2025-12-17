import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Iris Classification with ANN",
    page_icon="🌸",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model('iris_ann_model.h5')
    scaler = joblib.load('scaler.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    return model, scaler, metadata, history

try:
    model, scaler, metadata, history = load_model_and_scaler()
    model_loaded = True
except:
    model_loaded = False

# Header
st.title("🌸 Iris Flower Classification")
st.markdown("### Artificial Neural Network Classifier")
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model files not found! Please run `train_ann_model.py` first to train the model.")
    st.stop()

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.markdown(f"**Test Accuracy:** {metadata['test_accuracy']:.4f}")
st.sidebar.markdown(f"**Test Loss:** {metadata['test_loss']:.4f}")

st.sidebar.markdown("### 🎛️ Hyperparameters")
for param, value in metadata['hyperparameters'].items():
    st.sidebar.markdown(f"**{param.replace('_', ' ').title()}:** {value}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Model Performance", "ℹ️ About"])

# Tab 1: Prediction
with tab1:
    st.header("Make a Prediction")
    st.markdown("Enter the flower measurements below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.8,
            step=0.1
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.0,
            step=0.1
        )
    
    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=4.3,
            step=0.1
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=1.3,
            step=0.1
        )
    
    # Predict button
    if st.button("🔍 Classify Flower", type="primary"):
        # Prepare input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### **{metadata['target_names'][predicted_class].upper()}**")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        with col2:
            # Create probability chart
            fig = go.Figure(data=[
                go.Bar(
                    x=metadata['target_names'],
                    y=prediction[0] * 100,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Iris Species",
                yaxis_title="Probability (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display input values
        st.markdown("### 📋 Input Values")
        input_df = pd.DataFrame({
            'Feature': metadata['feature_names'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_df, use_container_width=True)

# Tab 2: Model Performance
with tab2:
    st.header("Model Training Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy plot
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=history['accuracy'],
            mode='lines',
            name='Training Accuracy',
            line=dict(color='#4ECDC4', width=2)
        ))
        fig_acc.add_trace(go.Scatter(
            y=history['val_accuracy'],
            mode='lines',
            name='Validation Accuracy',
            line=dict(color='#FF6B6B', width=2)
        ))
        fig_acc.update_layout(
            title="Model Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            height=400
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Loss plot
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#4ECDC4', width=2)
        ))
        fig_loss.add_trace(go.Scatter(
            y=history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#FF6B6B', width=2)
        ))
        fig_loss.update_layout(
            title="Model Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Final metrics
    st.markdown("### 📊 Final Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Final Training Accuracy", f"{history['accuracy'][-1]:.4f}")
    with metric_col2:
        st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.4f}")
    with metric_col3:
        st.metric("Final Training Loss", f"{history['loss'][-1]:.4f}")
    with metric_col4:
        st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")

# Tab 3: About
with tab3:
    st.header("About This Project")
    
    st.markdown("""
    ### 🌸 Iris Dataset
    The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers 
    from three different species: Setosa, Versicolor, and Virginica. Each sample has four features:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
    
    ### 🧠 Model Architecture
    This project uses an Artificial Neural Network (ANN) with the following architecture:
    - **Input Layer:** 4 neurons (one for each feature)
    - **Hidden Layer 1:** 16 neurons with ReLU activation
    - **Hidden Layer 2:** 8 neurons with ReLU activation
    - **Output Layer:** 3 neurons with Softmax activation (one for each class)
    
    ### ⚙️ Training Configuration
    The model was trained using the following hyperparameters:
    """)
    
    config_df = pd.DataFrame({
        'Hyperparameter': ['Learning Rate', 'Batch Size', 'Epochs', 'Loss Function', 'Optimizer'],
        'Value': [
            metadata['hyperparameters']['learning_rate'],
            metadata['hyperparameters']['batch_size'],
            metadata['hyperparameters']['epochs'],
            metadata['hyperparameters']['loss_function'],
            metadata['hyperparameters']['optimizer']
        ]
    })
    st.table(config_df)
    
    st.markdown("""
    ### 🚀 Technology Stack
    - **TensorFlow/Keras:** Deep learning framework
    - **Scikit-learn:** Data preprocessing and evaluation
    - **Streamlit:** Web application framework
    - **Plotly:** Interactive visualizations
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with ❤️ using Streamlit and TensorFlow</div>",
    unsafe_allow_html=True
)