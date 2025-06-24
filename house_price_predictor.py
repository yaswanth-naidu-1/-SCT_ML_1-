import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("house_prices.csv")  # CSV must have required columns

# Select input features and target
X = data[["SquareFootage", "Bedrooms", "Bathrooms"]]
y = data["Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Show sample predictions
print("\nSample Predictions:")
for i in range(min(5, len(predictions))):
    print(f"Predicted: ₹{predictions[i]:,.2f} | Actual: ₹{y_test.values[i]:,.2f}")


# Evaluate model
mse = mean_squared_error(y_test, predictions)
print(f"\nMean Squared Error: {mse:.2f}")