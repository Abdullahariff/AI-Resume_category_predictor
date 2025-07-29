AI Resume Category Predictor
About the Project
This project provides an AI-powered solution to classify resumes into job categories and analyze their strength. It uses machine learning for text extraction, preprocessing, TF-IDF vectorization, and classification with a LinearSVC model. A Streamlit web application offers an intuitive user interface for uploading resumes and receiving instant feedback.

Features
Resume Upload: Supports PDF and TXT formats.

Category Prediction: Classifies resumes into 25 predefined job categories.

Strength Analysis: Offers overall assessment and actionable improvement tips.

User-Friendly Interface: Built with Streamlit for easy interaction.

Getting Started
To run this application locally:

Prerequisites
Python 3.9+

pip

Installation
Clone the repository:

git clone https://github.com/Abdullahariff/AI-Resume_category_predictor.git
cd AI-Resume_category_predictor

Install dependencies:
Create a requirements.txt file with:

streamlit
joblib
PyPDF2
nltk
spacy
scikit-learn
pandas
numpy
json

Then run:

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm

Running the Application
streamlit run app.py

This will open the Streamlit application in your web browser.

Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

License
Distributed under the MIT License.

Contact
Abdullah Arif - arifabdullah619@gmail.com
Project Link: https://github.com/Abdullahariff/AI-Resume_category_predictor
