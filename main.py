from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
import string

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import re

# Загрузим необходимые ресурсы для лемматизации
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv('train.csv')
df['text'] = df['text'].astype(str)
df['encoded_cat'] = df['language'].astype('category').cat.codes

# Инициализируем лемматизатор
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Убираем лишние пробелы
    text = re.sub(r'\s{2,}', ' ', text)

    # Лемматизируем
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]

    tokens = [token for token in tokens if len(token) > 1]

    return tokens


def prcoess_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    '''
    Функция обрабатывает входной текст (переводит в нижний регистр, леммитизирует, 
    удаляет лишние символы и т.д.)

    Args:
        df (pd.DataFrame): Датафрейм с входными данными.
        column_name (str): Наименование колонки с заголовками.

    Returns:
        pd.DataFrame
    '''

    df[column_name] = df[column_name].str.lower()
    df['processed_text'] = df[column_name].apply(preprocess_text)
    df['processed_text'] = df['processed_text'].apply(lambda x: " ".join(x))

    return df


df = prcoess_text(df=df.copy(), column_name='text')
df.head(5)

texts = df['processed_text'].to_list()
classes = df['encoded_cat'].to_list()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts,
                                                                    classes,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    shuffle=True)

train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts,
                                                                      train_labels,
                                                                      test_size=0.01,
                                                                      random_state=0,
                                                                      shuffle=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

num_labels = df['encoded_cat'].unique().shape[0]

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_dataset.shuffle(1000).batch(16),
          epochs=2,
          batch_size=16,
          validation_data=val_dataset.shuffle(1000).batch(16),
          callbacks=[early_stopping]
          )

print(model.summary())

save_directory = "models"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)



def predict_category(text):
    predict_input = tokenizer.encode(text,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="tf")
    output = model(predict_input)[0]
    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    return prediction_value


y_pred = []

for text_ in test_texts:
    y_pred.append(predict_category(text_))



confusion = confusion_matrix(test_labels, y_pred)
cols_for_matrix = df.sort_values('encoded_cat')['language'].unique().tolist()

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
xticklabels=cols_for_matrix, yticklabels=cols_for_matrix)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(test_labels, y_pred))
