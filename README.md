# Customer Churn Prediction

## Overview

Customer Churn Prediction is an end-to-end Machine Learning application that predicts whether a customer is likely to leave a bank based on customer information. The project uses a Deep Learning model built with TensorFlow/Keras and provides an interactive Streamlit web application for real-time predictions.

---

## Features

- Customer churn prediction using Deep Learning
- Interactive Streamlit web application
- Data preprocessing with Label Encoder, One-Hot Encoder, and StandardScaler
- Real-time prediction interface
- Clean and user-friendly UI

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

## Project Structure

```
customer-churn-prediction/
│
├── app.py
├── experiment.ipynb
├── model.h5
├── label_encoder_gender.pkl
├── one_hot_encoder_geography.pkl
├── standard_scaler.pkl
├── requirements.txt
├── README.md
└── images/
    ├── home.jpeg
    ├── input.jpeg
    └── prediction.jpeg
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/radhayadav84808-crypto/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Application Screenshots

### Home Page

![Home Page](images/home.jpeg)

### Input Form

![Input Form](images/input.jpeg)

### Prediction Result

![Prediction Result](images/prediction.jpeg)

---

## Model Workflow

1. Load customer information.
2. Preprocess the input using trained encoders and scaler.
3. Feed the processed data into the TensorFlow model.
4. Predict customer churn probability.
5. Display the prediction instantly using Streamlit.

---

## Machine Learning Pipeline

- Data Collection
- Data Cleaning
- Feature Engineering
- Data Encoding
- Feature Scaling
- Deep Learning Model Training
- Model Evaluation
- Streamlit Deployment

---

## Future Improvements

- Improve prediction accuracy
- Deploy on Streamlit Community Cloud
- Add user authentication
- Store prediction history
- Support CSV batch predictions
- Improve UI design

---

## Author

**Radha Kumari Yadav**

GitHub:
https://github.com/radhayadav84808-crypto

---

## License

This project is developed for educational and learning purposes.
