# **Car Purchase Amount Predictions Using ANNs**  

**Car Purchase Amount Predictions Using ANNs** is a machine learning project that leverages **Artificial Neural Networks (ANNs)** to predict how much an individual is likely to spend on a car based on demographic and financial attributes. The model is trained using **supervised learning techniques** and evaluated to determine its accuracy.  

---

## **Table of Contents**  

1. [Project Overview](#project-overview)
2. [Video Demo](#Video-Demo)  
3. [Motivation and Purpose](#motivation-and-purpose)  
4. [Problem Statement and Objectives](#problem-statement-and-objectives)  
5. [Dataset Description](#dataset-description)  
6. [Model Architecture](#model-architecture)  
7. [Installation and Usage](#installation-and-usage)  
8. [Evaluation and Results](#evaluation-and-results)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## **Project Overview**  

### **Introduction**  
This project explores how **Artificial Neural Networks (ANNs)** can be used to predict the amount an individual is likely to spend on a car based on key financial and demographic factors. Using a structured dataset, the model learns patterns and relationships between input features such as **age, salary, net worth, and credit card debt** to make precise predictions.  

---

## **Video Demo**  

https://github.com/user-attachments/assets/f28bd6df-c229-4966-b7ab-eba2124d13ad

---

## **Motivation and Purpose**  
Predicting consumer spending behavior is a crucial application of machine learning, particularly in industries such as **automobile sales, finance, and marketing**. This project was developed to:  

- Gain hands-on experience with **Artificial Neural Networks (ANNs)** in real-world predictive analytics.  
- Understand how **financial and demographic data** influence car purchase behavior.  
- Experiment with different **ANN architectures** and **hyperparameters** for improved model accuracy.  

---

## **Problem Statement and Objectives**  

### **Problem Statement:**  
Many factors influence how much a person is willing to spend on a car, including their **income, debt, and net worth**. However, manually assessing these relationships can be challenging. The objective of this project is to use **machine learning** to automate and optimize **car purchase predictions** based on relevant financial and personal factors.  

### **Objectives:**  
✔️ Train an **Artificial Neural Network (ANN)** to predict **car purchase amounts** based on user data.  
✔️ Experiment with different **network architectures, activation functions, and optimizers**.  
✔️ Evaluate **model performance** using loss curves and validation metrics.  
✔️ Deploy the trained model to make **real-time predictions**.  

---

## **Dataset Description**  
The dataset used for this project includes the following key features:  

| Feature | Description |  
|---------|------------|  
| **Gender** | Binary (0 = Female, 1 = Male) |  
| **Age** | Age of the individual |  
| **Annual Salary** | Yearly income in USD |  
| **Credit Card Debt** | Total credit card debt in USD |  
| **Net Worth** | Total financial net worth in USD |  
| **Car Purchase Amount (Target)** | Amount the individual is expected to spend on a car |  

📌 **Data Preprocessing Steps:**  
- **Normalization**: Scaled numerical values to a common range for better ANN performance.  
- **Handling Missing Values**: Checked and filled any missing data appropriately.  
- **Feature Engineering**: Selected the most relevant predictors.  

---

## **Model Architecture**  
The **ANN model** was built using **TensorFlow/Keras** and consists of the following layers:  

| Layer Type | Activation Function | Purpose |  
|------------|---------------------|---------|  
| **Input Layer** | - | Accepts input features |  
| **Hidden Layer 1** | ReLU | Captures nonlinear relationships |  
| **Hidden Layer 2** | ReLU | Deepens feature learning |  
| **Output Layer** | Linear | Predicts car purchase amount |  

📌 **Hyperparameters Used:**  
- **Optimizer**: Adam  
- **Loss Function**: Mean Squared Error (MSE)  
- **Batch Size**: 32  
- **Epochs**: 50  

---

## **Installation and Usage**  

### **🔧 Prerequisites**  
Before running the project, install the necessary dependencies:  

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras
```

### **📌 Running the Model**  

#### **Clone the Repository:**  
```bash
git clone https://github.com/yourusername/CarPurchaseANN.git
cd CarPurchaseANN
```

#### **Run the Jupyter Notebook:**  
```bash
jupyter notebook Car_Purchase_ANN.ipynb
```

#### **Train the Model and Evaluate Predictions**  

---

## **Evaluation and Results**  

### **📊 Model Performance**  
The **ANN model** was trained using **supervised learning** and evaluated using **loss curves**:  

- **Training vs. Validation Loss:**  
  - The loss function showed **steady convergence** over multiple epochs.  
  - **Overfitting** was minimized using **regularization techniques**.  

📌 **Visualization:**  
- The **loss progression graph** shows how well the model learns over time.  
- The model provides **reasonably accurate predictions** on unseen test data.  

### **📈 Sample Prediction**  

Given the input:  
```python
X_Testing = np.array([[1, 50, 50000, 10985, 629312]])
```

The model predicts:  
```bash
Expected Car Purchase Amount: $30,500.00
```

---

## **Contributing**  

If you'd like to contribute to **Car Purchase Amount Predictions Using ANNs**, feel free to **fork the repository** and submit a **pull request**. Contributions are always welcome!  

### **Guidelines:**  
✔️ **Follow Best Practices**: Ensure the code is clean and well-documented.  
✔️ **Testing**: Validate model performance before submitting any changes.  
✔️ **Feature Additions**: If suggesting enhancements, provide a detailed explanation.  

---

## **License**  

This project is licensed under the **MIT License** – see the `LICENSE` file for details.  

---

## **📌 Summary**  
🚀 This project applies **Artificial Neural Networks (ANNs)** to predict car purchase amounts based on key **financial and demographic** features. By leveraging **deep learning**, it provides an **automated approach** to analyzing **consumer spending behavior** in the automobile industry.  
