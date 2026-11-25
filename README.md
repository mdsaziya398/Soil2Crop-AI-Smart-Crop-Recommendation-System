# 🌱 Soil2Crop AI – Smart Crop Recommendation System

### From Soil Insights to Smarter Harvests.

## 📌 Project Overview
Soil2Crop AI is a Machine Learning-based Smart Crop Recommendation System designed to identify suitable crops using agricultural data.

The project implements multiple Machine Learning algorithms, evaluates their performance, and automatically selects the best-performing model based on the weighted F1 score.

## 🎯 Objectives
- Develop an intelligent crop recommendation system.
- Apply different Machine Learning classification algorithms.
- Compare model performance using Accuracy and F1 Score.
- Identify and save the best-performing model.
- Support data-driven agricultural decisions.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Machine Learning

## 🤖 Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

## ⚙️ Project Workflow
1. Load preprocessed training and testing datasets.
2. Define multiple Machine Learning models.
3. Train each model using training data.
4. Evaluate predictions using Accuracy and Weighted F1 Score.
5. Compare the performance of all models.
6. Select the best-performing model.
7. Save the trained model using Joblib.

## 📂 Project Structure
```text
Soil2Crop-AI/
│
├── dataset/
├── processed/
├── train_model.py
├── crop_recommendation_model.pkl
├── requirements.txt
└── README.md
```

## 🚀 Installation and Execution

**Step 1: Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/Soil2Crop-AI.git
```

**Step 2: Navigate to the project folder**
```bash
cd Soil2Crop-AI
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run the training script**
```bash
python train_model.py
```

Make sure the required preprocessed CSV files are available inside the `processed/` directory before running the script.

## 📊 Model Evaluation
The project compares models using:
- Accuracy Score
- Weighted F1 Score
- Classification Performance

The model with the highest weighted F1 score is selected and saved.

## 🌟 Expected Outcome
An intelligent crop recommendation model that evaluates agricultural data and identifies the best-performing Machine Learning algorithm for crop classification.

## 👩‍💻 Author
**Saziya**

B.Tech – Computer Science and Engineering (IoT)

## 🔮 Future Enhancements
- Develop a Streamlit-based user interface.
- Integrate real-time weather information.
- Add soil parameter visualization.
- Provide crop-specific farming suggestions.
- Deploy the application online.

---
**Soil2Crop AI – Understand Your Soil. Choose Your Crop. Grow Smarter.** 🌱
