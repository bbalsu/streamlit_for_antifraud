import io
import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import torch
import _pickle as cPickle
from transformers import AutoTokenizer, AutoModel
# from sklearn.preprocessing import MinMaxScaler
# import tqdm
#streamlit run C:\Users\Acer\PycharmProjects\streamlead_web2\main.py

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()

@st.cache(allow_output_mutation=True)
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    return model, tokenizer

def load_svc():
    with open('clf.pkl', 'rb') as fid:
        svc = cPickle.load(fid)
    return svc

def preprocess_image(img, reader, model):
    result = reader.readtext(img)
    rare_text = [result[i][- 2] for i in range(len(result))]
    text = str.lower(" ".join(rare_text))
    content_item_embeddings = []
    with torch.no_grad():
        sentence_embeddings = embed_bert_cls(text, model, tokenizer)
        content_item_embeddings.append(sentence_embeddings)
    return sentence_embeddings

def predict(emb):
    svc = load_svc()
    pred = svc.predict(emb)
    return pred

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания (формат jpg/jpeg/png)')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        img = Image.open(io.BytesIO(image_data))
        return img
        # if img.format == 'JPG' or img.format == 'JPEG':
        #     return Image.open(io.BytesIO(image_data))
        # # elif img.format == 'PNG':
        # #     im = Image.open(io.BytesIO(image_data))
        # #     bg = Image.new("RGB", im.size, (255, 255, 255))
        # #     bg.paste(im, im)
        # #     bg.save("colors.jpg")
        # #     return Image.open(io.BytesIO('colors.jpg'))
        # else:
        #     st.write('Загрузите файл в предложенном формате')
        #     return None
        # x = Image.open(io.BytesIO(image_data))
        # return x.convert('RGB')
        # im = Image.open(io.BytesIO(image_data))
        # bg = Image.new("RGB", im.size, (255, 255, 255))
        # bg.paste(im, im)
        # bg.save("colors.jpg")
        # return Image.open('colors.jpg')
        # return Image.open(io.BytesIO(image_data))
    else:
        return None

def print_predictions(pred):
    if pred[0] == 0:
        st.write('не мошенник')
    else:
        st.write('мошенник')

svc = load_svc()
bert_model, tokenizer = load_bert_model()
reader = easyocr.Reader(['ru', 'en'])

st.title('Антифрод: классификация мошеннических отзывов')
img = load_image()
result = st.button('Распознать изображение')
if result:
    if img.format == 'JPEG' or img.format == 'JPG':
        emb = preprocess_image(img, reader, bert_model)
        pred = predict(emb)
        st.write('**Результаты распознавания:**')
        print_predictions(pred)
    else:
        st.write('**Загрузите изображение в предложенном формате**')

# st.title('Выявление мошеннических изображений')
# img = load_image()
