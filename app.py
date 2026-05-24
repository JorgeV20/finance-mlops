from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI()

# Load model
model = mlflow.sklearn.load_model("runs:/3841d465650e4932830adacd486c7899/model")

@app.post("/predict")
def predict_sentiment(text: str):
    prediction = model.predict([text])
    return {"sentiment": prediction[0]}