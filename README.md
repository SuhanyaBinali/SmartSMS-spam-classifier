# 🛡️ SmartSMS — AI-Powered SMS Spam Detection

SmartSMS is a machine learning-based web application that classifies SMS messages as either **spam** or **legitimate (ham)**.

The application uses **Natural Language Processing (NLP)** techniques to preprocess SMS text, **TF-IDF** for feature extraction, and a **Linear Support Vector Machine (Linear SVM)** to perform the final classification.

Users can enter an SMS message through the Streamlit web interface and receive an instant classification result along with the model's decision score.


## 📌 Project Overview

SmartSMS is a machine learning-based SMS spam detection system.

The application accepts an SMS message from the user, processes the text using Natural Language Processing techniques, transforms the message into numerical features using a trained TF-IDF vectorizer, and uses a trained machine learning model to classify the message as either:

- 🟢 **Ham** – legitimate SMS
- 🔴 **Spam** – unwanted or potentially harmful SMS

The trained model and TF-IDF vectorizer are stored in the `models/` directory and are used by the Streamlit application for real-time prediction.




## 🌐 Live Demo

🔗 **[Launch SmartSMS Spam Detector](https://smartsms-spam-classifier.streamlit.app)**


## ✨ Features

- 📩 **SMS Message Classification** — Classifies messages as spam or legitimate (ham).
- 🤖 **Machine Learning Prediction** — Uses a trained Linear SVM model for classification.
- 🧹 **Text Preprocessing** — Cleans and prepares SMS messages before prediction.
- 📊 **TF-IDF Feature Extraction** — Converts text messages into numerical features for the machine learning model.
- 📈 **Model Decision Score** — Displays the Linear SVM decision score to provide additional insight into the prediction.
- 🔢 **Message Statistics** — Shows the number of characters and words in the entered message.
- ⚡ **Interactive Web Interface** — Provides a simple Streamlit interface for real-time message analysis.
- 📱 **Example Messages** — Provides sample spam and legitimate messages for quick testing.
- 📊 **Model Performance Metrics** — Displays the model's accuracy, precision, recall, and F1 score.

## 🖥️ Application Interface

SmartSMS provides an interactive Streamlit web interface for entering SMS messages, testing example messages, viewing model performance, and receiving spam or legitimate (ham) predictions.

### 🏠 SmartSMS Home Interface

The main interface introduces the SmartSMS application and provides an input area where users can enter an SMS message and analyze it.

![SmartSMS Home Interface](screenshots/Interface%201.png)

### 🧪 Example Messages and Model Performance

The interface provides example messages for quick testing and displays the performance metrics of the trained machine learning model.

![Example Messages and Model Performance](screenshots/Interface%202.png)

### ⚙️ How SmartSMS Works

This section explains the SmartSMS workflow and shows the remaining part of the application's interface.

![How SmartSMS Works](screenshots/Interface%203.png)

### 🚨 Spam Message Detection

When a spam message is analyzed, SmartSMS displays the prediction result along with the model's decision score.

![Spam Message Detection](screenshots/Interface%204.png)

### ✅ Legitimate Message Detection

When a legitimate (ham) message is analyzed, SmartSMS displays the legitimate-message prediction together with the model's decision score.

![Legitimate Message Detection](screenshots/Interface%205.png)


### 🧠 Machine Learning Approach

SmartSMS follows a simple Natural Language Processing and machine learning pipeline:

```text
SMS Message
    ↓
Text Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
Linear SVM Classifier
    ↓
Spam / Legitimate Prediction
```

## 📊 Model Performance

The trained Linear SVM model achieved the following performance on the evaluation dataset:

| Metric | Score |
|---|---:|
| **Accuracy** | **97.97%** |
| **Precision** | **97.41%** |
| **Recall** | **86.26%** |
| **F1 Score** | **91.50%** |

These metrics provide an overview of how well the model distinguishes between spam and legitimate SMS messages.

- **Accuracy** — Overall percentage of correctly classified messages.
- **Precision** — How many messages predicted as spam were actually spam.
- **Recall** — How many actual spam messages were correctly detected.
- **F1 Score** — Harmonic mean of precision and recall.


## 🤖 Model Information

SmartSMS uses a Natural Language Processing and machine learning pipeline to classify SMS messages as spam or legitimate.

### Model Pipeline

```text
SMS Message
     ↓
Text Preprocessing
     ↓
TF-IDF Feature Extraction
     ↓
Linear SVM Classifier
     ↓
Spam / Legitimate Prediction
```


## 🛠️ Technologies Used

### Programming Language
- **Python**

### Machine Learning & NLP
- **Scikit-learn**
- **TF-IDF Vectorization**
- **Linear Support Vector Machine (SVM)**

### Data Processing
- **Pandas**
- **NumPy**
- **NLTK**

### Model Persistence
- **Joblib**

### Web Application
- **Streamlit**

### Development Environment
- **Jupyter Notebook**
- **Visual Studio Code**
- **Python Virtual Environment (`venv`)**


## 📁 Project Structure

```text
SmartSMS-spam-classifier/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── Spam_Classification_original.ipynb
│   └── Spam_Classification_v2.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 📂 Folder and File Description

- **`data/`** — Contains the SMS dataset used for the project.
- **`models/`** — Contains the trained Linear SVM model and TF-IDF vectorizer.
- **`notebooks/`** — Contains the original course notebook and the developed version of the notebook.
- **`src/`** — Contains the reusable Python modules for preprocessing, model training, and prediction.
- **`app.py`** — Streamlit application that provides the web interface.
- **`requirements.txt`** — Lists the Python packages required to run the project.
- **`.gitignore`** — Specifies files and folders that should not be tracked by Git.
- **`README.md`** — Project documentation.




## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SuhanyaBinali/SmartSMS-spam-classifier.git

```


## 🔮 Future Improvements

Possible improvements include:

- Improve model performance with additional training data.
- Compare multiple machine learning algorithms.
- Add confidence/probability visualization.
- Improve the user interface.
- Add prediction history.
- Continuously update the training dataset with newer spam patterns.



## 👩‍💻 Author

**Suhanya Binali Wanniarachchi**

SmartSMS – AI-Powered SMS Spam Detectiongit add README.md




## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.











