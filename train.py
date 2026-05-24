import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Create synthetic Financial Dataset
def load_data():
    data = {
        "text": [
            "Profits exceeded expectations this quarter.",
            "The company is facing bankruptcy due to high debt.",
            "The stock market remained completely flat today.",
            "Revenue grew by 20% year over year.",
            "CEO resigns amidst massive accounting scandal.",
            "Interest rates were left unchanged by the central bank.",
            "Dividends have been cut to save cash.",
            "New product launch drives record sales.",
            "Inflation data matches analyst predictions exactly.",
            "Supply chain issues cause heavy manufacturing delays."
        ],
        "sentiment": [
            "Positive", "Negative", "Neutral", "Positive", "Negative", 
            "Neutral", "Negative", "Positive", "Neutral", "Negative"
        ]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

    # Create 'mlruns' directory to store data locally
    mlflow.set_experiment("Financial_Sentiment_Model")

    # Start the MLflow run
    with mlflow.start_run():
        print("Training model...")
        
        n_estimators = 100
        max_depth = 5
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Model trained! Accuracy: {acc:.2f}")

        # Log the hyperparameters used to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log the performance metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Save the actual model artifact so FastAPI can load it later
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Get the unique Run ID (need it for FastAPI)
        run_id = mlflow.active_run().info.run_id
        print(f"\nSUCCESS! Model saved to MLflow.")
        print(f"Your Run ID is: {run_id}")
        print("Use this path in the FastAPI app: runs:/{}/model".format(run_id))