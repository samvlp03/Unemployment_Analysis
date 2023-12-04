import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the unemployment dataset
dataset_path = 'UE_Data.csv'  
unemployment_data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset preview:")
print(unemployment_data.head())

# Basic statistics and information about the dataset
print("Dataset summary:")
print(unemployment_data.info())
print(unemployment_data.describe())

plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=unemployment_data)
plt.title('Unemployment Rate in India during COVID-19')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Model Preparation
# Assuming 'Date' is a datetime object, convert it to a numerical feature for regression
unemployment_data['Date'] = pd.to_datetime(unemployment_data['Date'], format='%d-%m-%Y')
unemployment_data['NumericalDate'] = pd.to_numeric(unemployment_data['Date'])

# Display the first few rows of the updated dataset
print("Updated Dataset preview:")
print(unemployment_data.head())

# Drop rows with missing values in the target variable
unemployment_data = unemployment_data.dropna(subset=['Estimated Unemployment Rate (%)'])

# Split the dataset into features and target variable
features = unemployment_data['NumericalDate']
target = unemployment_data['Estimated Unemployment Rate (%)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Drop any remaining NaN values in the target variable
X_train = X_train.dropna()
y_train = y_train.dropna()
X_test = X_test.dropna()
y_test = y_test.dropna()

# Reshape the features to a 2D array
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the model's performance
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Numerical Date')
plt.ylabel('Unemployment Rate')
plt.title('Linear Regression Model for Unemployment Analysis')
plt.show()

print("Unemployment analysis and modeling completed.")
