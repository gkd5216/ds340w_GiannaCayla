from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load only one folder to avoid schema mismatch
dataset = load_dataset(
    "parquet",
    data_files="/Users/giannadelorenzo/.cache/huggingface/hub/datasets--CS2CD--CS2CD.Counter-Strike_2_Cheat_Detection/snapshots/44e5129654508b22802a050a45bcdbb44b103d87/with_cheater_present/0.parquet"
)

print(dataset)

# For demo: pick one column as "text" and add a dummy label column
# (Adjust based on what columns exist in your parquet files)
def preprocess_data(example):
    # Example: treat 'active_weapon_name' as text input
    text = str(example.get("active_weapon_name", "unknown"))
    # Example: use 'is_alive' (True/False) as label if it exists
    label = int(example.get("is_alive", 0))
    return {"text": text, "label": label}

dataset = dataset["train"].map(preprocess_data)

# Keep only 2000 rows for demo training
dataset = dataset.shuffle(seed=42).select(range(2000))

# Split manually (train=80%, test=20%)
train_test = dataset.train_test_split(test_size=0.2)

# Tokenizer + Model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

encoded = train_test.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training Arguments
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    num_train_epochs=1,   # keep it small for demo
    per_device_train_batch_size=8,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
)

# Train + Evaluate
trainer.train()
results = trainer.evaluate()
print("Final Evaluation:", results)

