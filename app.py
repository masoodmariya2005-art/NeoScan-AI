from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and train the model
data = pd.read_csv("cancer_data.csv")
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_cancer(sample):
    prediction = model.predict([sample])[0]
    return encoder.inverse_transform([prediction])[0]

# API routes
@app.route('/')
def home():
    return "Neo Scan AI Backend Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        prediction = predict_cancer(data)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
