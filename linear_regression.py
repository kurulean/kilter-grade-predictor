import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

conn = sqlite3.connect("kilter_data.sqlite")

# load data from the SQL query into a DataFrame
df = pd.read_sql_query(
    "SELECT * FROM kilter_train", conn
)

# Target: what we are predicting/measuring
y = df["difficulty_numeric"]

# Features: info we want to pass into the model
features = [
    "angle",
    "num_holds",
    "diameter"
]

# x is the variable holding all the info to make predictions
x = df[features]

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state=42 
)

# initialize linear regression model
model = LinearRegression()

# train model using the trainig data
model.fit(x_train, y_train)

# test the model on climbs the model did not train on
predictions = model.predict(x_test)

# measure how wrong it was
mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
