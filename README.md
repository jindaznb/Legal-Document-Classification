
## Project Title: Legal Document Classification

### Overview

This project focuses on classifying legal documents based on their content using Natural Language Processing (NLP) techniques. The aim is to train a model that can accurately predict the posture of legal documents using a dataset of judicial opinions.

### Table of Contents

- [Installation](#installation)
- [Data Loading](#data-loading)
- [Data Processing](#data-processing)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Evaluation of Results](#Evaluation-of-Results)
- [Future Work](#Future-Work)

### Installation

To set up the project environment, follow these steps:

1.  **Download the Zip file:**
   - Click on "Download ZIP." 
   - Extract the contents of the ZIP file to your desired directory.

    
2. **Set up a virtual environment (optional but recommended):**
        
    ` python -m venv venv source venv/bin/activate `
    
3. **Install required packages:** You can install the necessary packages using `pip`. Run the following command:
    

    
    `pip install -r requirements.txt`
    
    Create a `requirements.txt` file with the following content:
    

    
    `json pandas matplotlib scikit-learn`
    

### File Denotebook `submission.ipynb`ions

 
- **`submission.ipynb`**: Jupyter Notebook detailing the code used for generating final submission results. It includes code, visualizations, and analysis findings on the dataset, showcasing model performance.
    
- **`tr-report.pdf`**: The report file along with submission.ipynb for instructions, and usage.
    
- **`README.md`**: Documentation file providing a comprehensive guide to the repository, its setup, usage instructions, and explanations of each component. This file is the entry point for onboarding new users.
    
- **`requirements.txt`**: Lists required Python packages to ensure the code runs consistently across different environments. It includes essential libraries such as `scikit-learn`, `pandas`, and others used for data preprocessing, model training, and evaluation.

- **`TRDataChallenge2023.txt`**: The primary dataset file in plain text format. This file contains legal documents with structured text that serves as input for model training and evaluation. It is integral to the project.
    
- **`TRDataChallenge2023.zip`**: Compressed archive containing the original dataset. This is a backup format of `TRDataChallenge2023.txt` and may be used to extract the dataset when required.

- **`legal-bert.ipynb`**: Jupyter Notebook exploring potential of using Legal BERT, a specialized model for handling legal text data. This notebook demonstrates various applications and results for Legal BERT in classifying legal text.

### Data Loading

1. **Prepare your data file:** Ensure that your data is formatted as JSON and stored in a text file named `TRDataChallenge2023.txt`. Each line should represent a separate JSON object containing the necessary fields.
    
2. **Load the data:** The notebook `submission.ipynb` will read each line of the file and parse it into a Python dictionary, extracting relevant text and labels for further analysis.
    

### Data Processing

1. **Document Processing:**
    
    - The notebook `submission.ipynb` counts the number of documents, postures, and paragraphs.
    - It aggregates text from all paragraphs in each document into a single string for model input.
2. **Label Frequency Analysis:**
    
    - The distribution of labels (postures) is calculated and visualized.
    - Thresholds are established to evaluate the frequency of labels, helping to determine which classes are sufficiently represented in the dataset.
3. **Filtering Low-Frequency Labels:**
    
    - Labels with occurrences below a specified threshold (e.g., 20) are removed to improve model performance and reduce noise.

### Data Visualization

- The notebook `submission.ipynb` utilizes Matplotlib to plot a table that summarizes the distribution of labels across various thresholds, illustrating how many classes are retained based on their frequency.

### Model Training and Evaluation

1. **Train-Test Split:**
    
    - The dataset is split into training and testing sets using an 80-20 ratio while stratifying by labels.
2. **Text Vectorization:**
    
    - The `TfidfVectorizer` is used to convert text data into numerical format suitable for model training.
3. **Model Training:**
    
    - A Logistic Regression model is instantiated and fitted to the training data.
    - Predictions are made on the test set to evaluate the model’s performance.
4. **Performance Metrics:**
    
    - The notebook `submission.ipynb` calculates and prints the model's accuracy and classification report, including precision, recall, and F1-score for each label.

### Evaluation of Results


- **Confusion Matrix (Top 10 Classes)**: Shows a heatmap of true vs. predicted labels, limited to the top 10 classes and an "Other" category for all remaining classes.
    
- **Classification Report Summary (Top 10 Classes)**: A bar chart of precision, recall, and F1-scores for each of the top 10 classes.
    
- **F1 Score for Top 10 Classes**: Highlights the model’s F1-score for each top-performing class, which indicates accuracy in balancing precision and recall.
    
- **Sample Inference Instances (Top 10 Classes)**: Displays a table of individual predictions, limited to examples from the top 10 classes to show specific instances of correct and incorrect predictions.
    
- **Label Distribution for Top 10 Classes**: A distribution chart of the true labels in the test set, focusing on the top 10 classes. This aids in verifying that the top 10 classes are well-represented and identifying any remaining class imbalance.

### Future Work

For future development, it is recommended to explore the potential of leveraging Legal BERT, a model specifically designed for legal text processing. The `legal_bert.ipynb` notebook `submission.ipynb` includes skeleton codes, and the training pipeline has already been completed. Although time limitations, it presents an opportunity for subsequent projects. Utilizing Legal BERT may enhance classification accuracy and better handle the complexities of legal language.
