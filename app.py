from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI()

# Load model
model = mlflow.sklearn.load_model("runs:/0714b98aa85d4164bdcf6ab24520da0f/model")

@app.post("/predict")
def predict_sentiment(text: str):
    prediction = model.predict([text])
    return {"sentiment": prediction[0]}