# Spam Mail Classifier

This project uses a machine learning model to detect spam emails using a CSV dataset and a Colab notebook.


## Features

- Data preprocessing (cleaning and tokenization)
- Feature extraction (CountVectorizer)
- Model training (Multinomial Naive Bayes)
- Evaluation (accuracy, precision, recall)
- Supports further improvements (e.g., deep learning, word embeddings)

## Technologies Used

- Python
- Google Colab
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (for visualization)

## Dataset Info

The `spam_data.csv` file contains SMS messages labeled as `spam` or `ham` (not spam). It’s used for training and evaluating the classifier.

## How to Run

1. Open the notebook `spam_classifier.ipynb` in Google Colab.
2. Upload the `spam_data.csv` file when prompted (or mount Google Drive).
3. Run all cells to train and evaluate the model.

##  Results
 
 Class-wise performance:
 
- Class 1 (Spam):
  - Precision: 96%
  - Recall: 95%
  - F1-score: 95%
- Class 0 (Non-spam):
  - Precision: 99%
  - Recall: 99%
  - F1-score: 99%

The model performs very well overall, with high accuracy and balanced precision and recall across both classes.

##  Future Improvements

- Use deep learning models (LSTM, BERT)
- Add support for email inputs instead of SMS

##  Author

  Dhenuka Dudde
  (https://github.com/Dhenuka45)
