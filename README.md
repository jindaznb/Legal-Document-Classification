
## Project Title: Legal Document Classification

### Overview

This project focuses on classifying legal documents based on their content using Natural Language Processing (NLP) techniques. The aim is to train a model that can accurately predict the posture of legal documents using a dataset of judicial opinions.

### Table of Contents

- [Installation](#installation)
- [Data Loading](#data-loading)
- [Data Processing](#data-processing)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Recommendations](#recommendations)
- [Contributing](#contributing)

### Installation

To set up the project environment, follow these steps:

1.  **Download the Zip file:**
   - Click on "Download ZIP." 
   - Extract the contents of the ZIP file to your desired directory.

    
2. **Set up a virtual environment (optional but recommended):**
        
    `` python -m venv venv source venv/bin/activate  # On Windows use `venv\Scripts\activate` ``
    
3. **Install required packages:** You can install the necessary packages using `pip`. Run the following command:
    

    
    `pip install -r requirements.txt`
    
    Create a `requirements.txt` file with the following content:
    

    
    `json pandas matplotlib scikit-learn`
    

### Data Loading

1. **Prepare your data file:** Ensure that your data is formatted as JSON and stored in a text file named `TRDataChallenge2023.txt`. Each line should represent a separate JSON object containing the necessary fields.
    
2. **Load the data:** The script will read each line of the file and parse it into a Python dictionary, extracting relevant text and labels for further analysis.
    

### Data Processing

1. **Document Processing:**
    
    - The script counts the number of documents, postures, and paragraphs.
    - It aggregates text from all paragraphs in each document into a single string for model input.
2. **Label Frequency Analysis:**
    
    - The distribution of labels (postures) is calculated and visualized.
    - Thresholds are established to evaluate the frequency of labels, helping to determine which classes are sufficiently represented in the dataset.
3. **Filtering Low-Frequency Labels:**
    
    - Labels with occurrences below a specified threshold (e.g., 20) are removed to improve model performance and reduce noise.

### Data Visualization

- The script utilizes Matplotlib to plot a table that summarizes the distribution of labels across various thresholds, illustrating how many classes are retained based on their frequency.

### Model Training and Evaluation

1. **Train-Test Split:**
    
    - The dataset is split into training and testing sets using an 80-20 ratio while stratifying by labels.
2. **Text Vectorization:**
    
    - The `TfidfVectorizer` is used to convert text data into numerical format suitable for model training.
3. **Model Training:**
    
    - A Logistic Regression model is instantiated and fitted to the training data.
    - Predictions are made on the test set to evaluate the model’s performance.
4. **Performance Metrics:**
    
    - The script calculates and prints the model's accuracy and classification report, including precision, recall, and F1-score for each label.

### Future Work

For future development, it is recommended to explore the potential of leveraging Legal BERT, a model specifically designed for legal text processing. The `legal_bert.py` script includes skeleton codes, and the training pipeline has already been completed. Although time limitations, it presents an opportunity for subsequent projects. Utilizing Legal BERT may enhance classification accuracy and better handle the complexities of legal language.