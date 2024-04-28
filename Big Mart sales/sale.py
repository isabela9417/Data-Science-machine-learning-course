# Importing necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Calculate RMSE for train set



# Load the train and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Explore the structure and summary statistics of the train dataset
print(train_data.head())
print(train_data.info())
print(train_data.describe())


# Pre-processing the data
# Handling missing values
imputer = SimpleImputer(strategy='mean')
train_data['Item_Weight'] = imputer.fit_transform(train_data[['Item_Weight']])
test_data['Item_Weight'] = imputer.transform(test_data[['Item_Weight']])

train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0], inplace=True)
test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0], inplace=True)

# Normalizing the variables
scaler = StandardScaler()
train_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']] = scaler.fit_transform(train_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']])
test_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']] = scaler.transform(test_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']])

# Define features and target variable
X = train_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']]
y = train_data['Item_Outlet_Sales']

# Splitting the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
val_rmse = mean_squared_error(y_val, val_predictions, squared=False)

print("Train RMSE:", train_rmse)
print("Validation RMSE:", val_rmse)

# Pre-processing the test set
test_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']] = scaler.transform(test_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']])

# Generating predictions for the test set
test_predictions = model.predict(test_data[['Item_Weight', 'Item_Visibility', 'Item_MRP']])

# Saving predictions to a CSV file
submission = pd.DataFrame({'Item_Identifier': test_data['Item_Identifier'], 'Outlet_Identifier': test_data['Outlet_Identifier'], 'Item_Outlet_Sales': test_predictions})
submission.to_csv('sales_predictions.csv', index=False)
