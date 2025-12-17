# **🌸 Iris Classification with ANN**

A complete machine learning project implementing an Artificial Neural Network (ANN) for classifying Iris flowers, with a beautiful Streamlit frontend interface.

## **📋 Project Overview**

This project demonstrates:

* Training an ANN classifier on the Iris dataset  
* Building an interactive web application with Streamlit  
* Deploying the model through GitHub

## **🎯 Model Specifications**

### **Hyperparameters**

* **Learning Rate:** 0.01  
* **Batch Size:** 16  
* **Epochs:** 50  
* **Loss Function:** Mean Squared Error (Sparse Categorical Crossentropy)  
* **Optimizer:** Stochastic Gradient Descent (SGD)

### **Architecture**

* Input Layer: 4 neurons  
* Hidden Layer 1: 16 neurons (ReLU)  
* Hidden Layer 2: 8 neurons (ReLU)  
* Output Layer: 3 neurons (Softmax)

## **🚀 Getting Started**

### **Prerequisites**

* Python 3.8 or higher  
* pip package manager

### **Installation**

1. **Clone the repository**

git clone https://github.com/YOUR\_USERNAME/iris-ann-classifier.git  
cd iris-ann-classifier

2. **Create a virtual environment** (recommended)

python \-m venv venv

\# On Windows  
venv\\Scripts\\activate

\# On macOS/Linux  
source venv/bin/activate

3. **Install dependencies**

pip install \-r requirements.txt

### **Training the Model**

Run the training script to create the model:

python train\_ann\_model.py

This will generate:

* `iris_ann_model.h5` \- Trained model  
* `scaler.pkl` \- Feature scaler  
* `training_history.json` \- Training metrics  
* `model_metadata.json` \- Model information

### **Running the Streamlit App**

Launch the web application:

streamlit run app.py

The app will open in your browser at `http://localhost:8501`

## **📁 Project Structure**

iris-ann-classifier/  
├── train\_ann\_model.py      \# Model training script  
├── app.py                   \# Streamlit application  
├── requirements.txt         \# Python dependencies  
├── README.md               \# Project documentation  
├── iris\_ann\_model.h5       \# Trained model (generated)  
├── scaler.pkl              \# Feature scaler (generated)  
├── training\_history.json   \# Training metrics (generated)  
└── model\_metadata.json     \# Model info (generated)

## **🌐 Deploying to GitHub**

### **Step 1: Initialize Git Repository**

git init  
git add .  
git commit \-m "Initial commit: ANN Iris classifier with Streamlit"

### **Step 2: Create GitHub Repository**

1. Go to [GitHub](https://github.com/) and create a new repository  
2. Name it `iris-ann-classifier`  
3. Don't initialize with README (we already have one)

### **Step 3: Push to GitHub**

git remote add origin https://github.com/YOUR\_USERNAME/iris-ann-classifier.git  
git branch \-M main  
git push \-u origin main

## **☁️ Deploying to Streamlit Cloud**

### **Option 1: Streamlit Community Cloud (Free)**

1. Go to [share.streamlit.io](https://share.streamlit.io/)  
2. Sign in with GitHub  
3. Click "New app"  
4. Select your repository: `YOUR_USERNAME/iris-ann-classifier`  
5. Set main file path: `app.py`  
6. Click "Deploy"

### **Important Notes for Deployment:**

* Make sure all generated files (`*.h5`, `*.pkl`, `*.json`) are committed to your repository  
* The app will automatically train the model if files are missing  
* Consider using Git LFS for large model files if needed

### **Option 2: Deploy Locally with Port Forwarding**

Use ngrok or similar tools to expose your local app:

\# Install ngrok  
\# Then run:  
ngrok http 8501

## **🎨 Features**

### **1\. Interactive Prediction Interface**

* Adjust flower measurements using sliders  
* Real-time classification with confidence scores  
* Visual probability distribution

### **2\. Model Performance Dashboard**

* Training and validation accuracy curves  
* Loss visualization over epochs  
* Final performance metrics

### **3\. Project Information**

* Dataset details  
* Model architecture visualization  
* Hyperparameter configuration

## **📊 Dataset Information**

The Iris dataset contains:

* **150 samples** from 3 species  
* **4 features** per sample  
* **Species:** Setosa, Versicolor, Virginica

Features:

1. Sepal Length (cm)  
2. Sepal Width (cm)  
3. Petal Length (cm)  
4. Petal Width (cm)

## **🔧 Customization**

### **Modify Hyperparameters**

Edit `train_ann_model.py` to change:

LEARNING\_RATE \= 0.01  
BATCH\_SIZE \= 16  
EPOCHS \= 50

### **Change Model Architecture**

Modify the model definition in `train_ann_model.py`:

model \= keras.Sequential(\[  
    layers.Dense(16, activation='relu'),  
    layers.Dense(8, activation='relu'),  
    layers.Dense(3, activation='softmax')  
\])

## **📈 Model Performance**

Expected results:

* **Training Accuracy:** \~98-99%  
* **Test Accuracy:** \~95-97%  
* **Training Time:** \<1 minute on CPU

## **🐛 Troubleshooting**

### **Model files not found**

\# Re-run training  
python train\_ann\_model.py

### **Port already in use**

\# Use different port  
streamlit run app.py \--server.port 8502

### **TensorFlow installation issues**

\# Try CPU-only version  
pip install tensorflow-cpu==2.15.0

## **📝 License**

This project is open source and available under the MIT License.

## **🤝 Contributing**

Contributions are welcome\! Please feel free to submit a Pull Request.

## **📧 Contact**

For questions or feedback, please open an issue on GitHub.

## **🙏 Acknowledgments**

* UCI Machine Learning Repository for the Iris dataset  
* TensorFlow and Keras teams  
* Streamlit community

---

**Built with ❤️ using TensorFlow, Streamlit, and Python**

