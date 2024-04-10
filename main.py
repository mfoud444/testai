from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the Keras model
model = load_model('keras_model.h5')  # Replace 'your_model.h5' with the path to your .h5 file

# Function to preprocess the input image
def preprocess_image(img):
    img = img.resize((224, 224))  # Assuming input size of 224x224
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255  # Normalization
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to predict the class of an image
def predict_class(img):
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    return prediction

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    prediction = predict_class(img)
    
    # Assuming your model output is a list of probabilities for each class
    # You may need to modify this based on your model's output
    prediction = prediction.tolist()[0]
    
    # Assuming you have two classes: Blight disease and Powdery mildew
    # Modify this based on your actual class names
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

# Handle OPTIONS requests
@app.options("/predict/")
async def options_predict():
    return {"methods": ["POST"]}