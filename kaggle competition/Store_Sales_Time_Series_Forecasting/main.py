'''
/kaggle/input/store-sales-time-series-forecasting/oil.csv
/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv
/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv
/kaggle/input/store-sales-time-series-forecasting/stores.csv
/kaggle/input/store-sales-time-series-forecasting/train.csv
/kaggle/input/store-sales-time-series-forecasting/test.csv
/kaggle/input/store-sales-time-series-forecasting/transactions.csv
'''

import pandas as pd
import numpy as np

# Load datasets
oil = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv', parse_dates=['date'])
holidays_events = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv', parse_dates=['date'])
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv', parse_dates=['date'])
transactions = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv', parse_dates=['date'])

print("Datasets loaded successfully.")
print(f"Oil dataset shape: {oil.shape}")
print(f"Holidays Events dataset shape: {holidays_events.shape}")
print(f"Stores dataset shape: {stores.shape}")
print(f"Train dataset shape: {train.shape}")
print(f"Test dataset shape: {test.shape}")
print(f"Transactions dataset shape: {transactions.shape}")

# Display first few rows of each dataset
print("\nOil dataset sample:")
print(oil.head())
print("\nHolidays Events dataset sample:")
print(holidays_events.head())
print("\nStores dataset sample:")
print(stores.head())
print("\nTrain dataset sample:")
print(train.head())
print("\nTest dataset sample:")
print(test.head())
print("\nTransactions dataset sample:")
print(transactions.head())

# Check for missing values
print("\nMissing values in each dataset:")
print(f"Oil dataset missing values:\n{oil.isnull().sum()}")
print(f"Holidays Events dataset missing values:\n{holidays_events.isnull().sum()}")
print(f"Stores dataset missing values:\n{stores.isnull().sum()}")
print(f"Train dataset missing values:\n{train.isnull().sum()}")
print(f"Test dataset missing values:\n{test.isnull().sum()}")
print(f"Transactions dataset missing values:\n{transactions.isnull().sum()}")




    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
# Example: Simple model training on the train dataset
# For demonstration, let's predict 'sales' using a simple RandomForestRegressor
# Note: In a real scenario, more feature engineering and preprocessing would be required
# Prepare data
X = train.drop(columns=['sales', 'date'])
y = train['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# create pipeline


pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])




import tensorflow as tf

try:
    # Detect TPU, return appropriate distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default to CPU/GPU strategy if TPU is not detected
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)


with strategy.scope():
    # Define your Keras model here
    model = tf.keras.Sequential([
        # ... your layers ...
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['sparse_categorical_accuracy']
    )










pipeline.fit(X_train, y_train,epochs=EPOCHS, 
    batch_size=BATCH_SIZE)
# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error on test set: {mse}")
print(accuracy_score(y_test, np.round(y_pred)))