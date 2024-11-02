
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Define file paths
file_path = 'TRDataChallenge2023.txt'
train_file = 'train_data.txt'
val_file = 'val_data.txt'


POSTURE_MAPPING = {label: idx for idx, label in enumerate(unique_labels)}
NUM_CLASSES = len(unique_labels)

# Initialize LegalBERT model and tokenizer
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)

# Step 1: Load and Split the Data
def load_data(file_path):
    documents = []
    with open(file_path, 'r') as file:
        for line in file:
            document = json.loads(line.strip())
            text = " ".join([p for section in document.get("sections", []) for p in section.get("paragraphs", [])])
            postures = document.get("postures", [])
            documents.append({"text": text, "postures": postures})
    return documents

# Split data into training and validation sets (80% train, 20% validation)
data = load_data(file_path)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 2: Custom Dataset Class for BERT Fine-Tuning
class LegalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def posture_to_label(self, postures):
        # Convert postures to a multi-label vector
        label = [0] * NUM_CLASSES
        for posture in postures:
            if posture in POSTURE_MAPPING:
                label[POSTURE_MAPPING[posture]] = 1
        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        encoding = self.tokenizer(
            sample["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        label = self.posture_to_label(sample["postures"])
        encoding["labels"] = torch.tensor(label, dtype=torch.float)
        return {key: val.squeeze() for key, val in encoding.items()}

# Load train and validation datasets
train_dataset = LegalDataset(train_data, tokenizer=tokenizer)
val_dataset = LegalDataset(val_data, tokenizer=tokenizer)


# Empty GPU cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# Step 3: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Step 4: Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 5: Train and Evaluate the Model
trainer.train()
# Step 6: Save the Trained Model
model.save_pretrained("./legalbert_procedural_postures")
tokenizer.save_pretrained("./legalbert_procedural_postures")

# Evaluate the model after training
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

