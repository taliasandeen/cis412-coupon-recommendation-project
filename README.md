# 🚗 In-Vehicle Coupon Recommendation – CIS 412 Final Project

This repository contains the full deployment package for our CIS 412 final project, including:

- A **Streamlit web application** that predicts whether a driver will accept an in-vehicle coupon  
- The **dataset** used to train the model  
- Our **Jupyter notebook** with exploratory analysis and model development  
- Our **presentation slide deck**

The Streamlit app allows a user (professor) to plug in scenario details—destination, passenger type, weather, time of day, coupon category, driving distance, and more—and receive both a **prediction (accept/reject)** and **probability score**.  
This satisfies the project deployment requirement and provides a fully interactive demo.

---

## 🚀 Live Streamlit App (Deepnote)
You can run the interactive model here:

🔗 **Streamlit App:** https://deepnote.com/streamlit-apps/1dff3afe-12ec-4c0e-a908-2a0e21c47d5a?utm_content=f167eb1c-f8c3-46e3-8ad0-eec822609986) 

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `app.py` | The Streamlit application that loads the model, handles user inputs, and produces predictions |
| `in-vehicle-coupon-recommendation.csv` | Original dataset used for model training |
| `requirements.txt` | Dependency list required to run the Streamlit app |
| `Final Combined Draft-1.ipynb` | Notebook containing EDA, preprocessing, and model development |
| `CIS412 Final Project-3.pptx` | Final project presentation slide deck |

---

## 🧠 Project Overview

The objective of this project was to analyze factors that influence drivers’ decisions to accept in-vehicle coupons and use classification models to predict acceptance.

We:

1. Performed data cleaning and exploratory analysis  
2. Tested several machine learning models  
3. Selected and deployed the final model using Streamlit  
4. Built an interactive app for input-based predictions  

The result is a fully working deployment where any instructor or user can modify parameters and instantly see model outputs.

---

## 🛠️ How to Run the Streamlit App Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cis412-coupon-recommendation-project
cd cis412-coupon-recommendation-project
