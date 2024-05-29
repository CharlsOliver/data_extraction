import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

# 1. Preparación del entorno y carga de datos
# ------------------------------------------------------------
data_dir = 'output'  # Ruta a los archivos etiquetados

# Leer los archivos etiquetados y convertirlos en un DataFrame
data = []
for file in os.listdir(data_dir):
    if file.endswith('.ann'):
        with open(os.path.join(data_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                word, tag = line.strip().split()
                data.append((word, tag))

df = pd.DataFrame(data, columns=['word', 'tag'])

# Dividir los datos en entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convertir los DataFrames a datasets de Hugging Face
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 2. Tokenización y alineación de etiquetas
# ------------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['word'], truncation=True, is_split_into_words=True)
    labels = []
    for i in range(len(examples['word'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        if word_ids is None:
            print(f"Warning: No word_ids for index {i}. Skipping this example.")
            continue
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(examples['tag'][i])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenización de los datasets
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["__index_level_0__"])
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["__index_level_0__"])

# Verificar tokenización correcta
print(train_dataset[0])

# 3. Definición y entrenamiento del modelo BERT
# ------------------------------------------------------------
# Definir el modelo
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(df['tag'].unique()))

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Definir el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Entrenar el modelo
trainer.train()

# 4. Evaluación y uso del modelo entrenado
# ------------------------------------------------------------
# Evaluación del modelo
trainer.evaluate()

# Función para realizar predicciones
def predict(texts):
    tokenized_inputs = tokenizer(texts, truncation=True, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])
    tags = [df['tag'].unique()[pred] for pred in predictions[0].numpy() if pred != -100]
    return list(zip(tokens, tags))

# Ejemplo de predicción
texts = [["This", "is", "a", "test", "sentence", "."]]  # Reemplaza con tus textos de prueba
print(predict(texts))
