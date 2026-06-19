import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Historical sales data
data = {
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [100, 120, 140, 160, 180, 200]
}

df = pd.DataFrame(data)

# Prepare data
X = df[['Month']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future_months = pd.DataFrame({'Month': [7, 8, 9]})
predictions = model.predict(future_months)

print("Future Sales Predictions:")
for month, pred in zip(future_months['Month'], predictions):
    print(f"Month {month}: {pred:.2f}")

# Visualization
plt.scatter(df['Month'], df['Sales'])
plt.plot(df['Month'], model.predict(X))
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Prediction")
plt.show()
