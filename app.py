import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, UploadFile
import io
import uvicorn


MODEL = tf.keras.models.load_model('vegetable_model/')

app = FastAPI()

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict')
async def predic(file: UploadFile) -> str:
    upload = await file.read()
    content = io.BytesIO(upload)


    img = image.load_img(content, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = MODEL.predict(images)
    predicted = np.argmax(classes)

    if predicted == 0:
        return "This is bean"
    elif predicted == 1:
         return "This is Bitter Gourd"
    elif predicted == 2:
        return "This is Bottle Gourd"
    elif predicted == 3:
        return "This is Brinjal"
    elif predicted == 4:
        return "This is Broccoli"
    elif predicted == 5:
        return "This is Cabbage"
    elif predicted == 6:
        return "This is Capsicum"
    elif predicted == 7:
        return "This is Carrot"
    elif predicted == 8:
        return "This is Cauliflower"
    elif predicted == 9:
        return "This is Papaya"
    elif predicted == 10:
        return "This is Cucumber"
    elif predicted == 11:
        return "This is Potato"
    elif predicted == 12:
        return "This is Pumpkins"
    elif predicted == 13:
        return "This is Radish"
    elif predicted == 14:
        return "This is Tomato"
    else:
        return "Sorry i don't know"

    # 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. = Bean
    # 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. = Bitter Gourd
    # 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. = Bottle Gourd
    # 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. = Brinjal
    # 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. = Broccoli
    # 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. = Cabbage
    # 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. = Capsicum
    # 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. = Carrot
    # 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. = Cauliflower
    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. = Papaya
    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. = Potato
    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. = Cucumber
    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. = Pumpkin
    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. = Radish
    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. = Tomato
