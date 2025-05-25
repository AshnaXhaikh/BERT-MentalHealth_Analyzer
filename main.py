import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import re
import nltk
from nltk.corpus import stopwords

# download nltk stopwords
# nltk.download('stopwords')

# Force CPU if you don't have GPU
device = torch.device('cpu')

model = AutoModelForSequenceClassification.from_pretrained(
        'mental_health_model',
        use_safetensors=True,
        device_map='cpu',
        torch_dtype=torch.float32
    ).to(device)


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert_tokenizer')
label_encoder = pickle.load(open('label_encoder.pkl','rb'))


stopwords = set(stopwords.words('english'))
def clean_statement(input_text):
  
    if not isinstance(input_text, str):
        return ""
    # remove urls
    text = re.sub(f"http[s]?://\S+", "",input_text)

    #remove markdown-style links
    text = re.sub(r"\[.*?\]\(.*?\)", "", input_text)

    #remove extra whitespaces
    text = re.sub(f"\s+", " ", input_text).strip()

    words = text.split()
    # remove stopwords and punctuation
    words = [word for word in words if word.lower() not in stopwords and word]
    cleaned_text = " ".join(words)
    return cleaned_text

def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]


st.cache_resource.clear()  # In your Streamlit app
# UI app
st.title('Bert Sentimnent Analysis')

input_text = st.text_input("Enter text here for analyzing....")

if st.button("detect"):
    predicted_class = detect_anxiety(input_text)
    st.write("Predicted Status :", predicted_class)