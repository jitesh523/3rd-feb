import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Drop any missing values
df.dropna(inplace=True)

# Extract queries (X) and intent labels (y)
X = df["instruction"]  # User queries
y = df["intent"]       # Intent classification

# Convert categorical labels (intent) into numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert text into numerical feature vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Save the label encoder and vectorizer for later use
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Step 2: Data Preprocessing Completed!")

# 🔽🔽🔽 PASTE THE TRAINING CODE BELOW THIS 🔽🔽🔽

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "chatbot_model.pkl")

# Evaluate model performance
accuracy = model.score(X_test, y_test)
print(f"Step 3: Model Training Completed! Accuracy: {accuracy:.2f}")
