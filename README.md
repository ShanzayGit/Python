## Overview
# ML Algorithms from Scratch

This repository contains implementations of fundamental Machine Learning algorithms built from scratch using Python. The goal of this project is to strengthen core ML concepts by avoiding high-level libraries like Scikit-learn for algorithm implementation.


## 📌 Project Overview

This project focuses on implementing core ML algorithms manually to understand their internal working, mathematical intuition, and step-by-step computation.

It is designed for learning, academic practice and portfolio building.


## 📁 Project Structure

```
ML_Algorithms_From_Scratch/
├── data/                         # Datasets for experiments
│   ├── clustering_data.csv
│   ├── regression_data.csv
│   ├── real_estate_price_size.csv
│   └── revenue_customers.xlsx
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── Linear_Regression.ipynb
│   ├── KMeans_Clustering.ipynb
│   └── Classification.ipynb
│
├── src/                          # Source code implementations
│   ├── __init__.py               # Marks src as a package
│   ├── regression.py             # Linear regression from scratch
│   ├── kmeans.py                 # K-means clustering algorithm
│   └── classification.py         # Classification algorithms
│
├── README.md                     # Project overview and usage
└── requirements.txt              # Dependencies (numpy, pandas, matplotlib, etc.)

```




## 🚀 Implemented Algorithms

### 1. Linear Regression
- Simple linear regression from scratch
- Gradient-based optimization (if applicable)
- Cost function implementation

### 2. K-Means Clustering
- Distance-based clustering
- Centroid initialization
- Iterative optimization until convergence

### 3. Classification Models (optional/extendable)
- Basic classification logic
- Decision boundary intuition



## 🧠 Key Concepts Covered

- Vectorized operations using NumPy
- Euclidean distance computation
- Iterative optimization techniques
- Unsupervised learning (K-Means)
- Supervised learning (Regression / Classification basics)



## 🛠️ Tech Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebooks



## 📚 Learning Purpose

This project was built to:
- Understand ML algorithms at a low level
- Improve coding skills in algorithm design
- Prepare for internships and technical interviews
- Strengthen mathematical intuition behind ML



## ▶️ How to Run

1. Clone or download the repository
2. Open in VS Code or Jupyter Notebook
3. Run notebooks inside `notebooks/` folder
4. Or import modules from `src/`

Example:

python
from src.kMeans import kmeans



## 📌 Future Improvements

* Add data preprocessing module
* Add visualization for clustering and regression
* Convert into pip-installable package
* Add more algorithms (SVM, Logistic Regression, PCA)


## 👤 Author

Student focused on Machine Learning, Data Structures, and AI development.

## ⭐ Note

This project is purely educational and built from scratch to reinforce core ML understanding without using high-level ML libraries.


