from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
app = FastAPI()

# Load the Keras model
model_path = os.path.join(os.getcwd(), 'keras_model.h5')



def preprocess_image(img):
    img = img.resize((224, 224))  
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_class(img):
    model = load_model(model_path)
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    return prediction

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    prediction = predict_class(img)
    prediction = prediction.tolist()[0]
    class_names = ["Blight disease on grape leaves", "Powdery mildew on grapes"]
    result = {"prediction": class_names[np.argmax(prediction)], "probabilities": prediction}
    return result

# Allow CORS (Cross-Origin Resource Sharing) for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/predict/")
async def options_predict():
    return {"methods": ["POST"]}