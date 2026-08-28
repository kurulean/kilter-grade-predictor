"""
Kilter Grade Predictor - ML Model

Predict a climb's V-grade from its hold layout on the original Kilter Board.
Currently only working on the 7x10 layout for more simplicity.
First iteration will be a XGBoost regression model. The second iteration will be a either a GNN or CNN model.

"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import sqlite3
import pandas as pd

conn = sqlite3.connect("kilter_data.sqlite")

df = pd.read_sql_query(
    "SELECT * FROM kilter_train",
    conn
)

conn.close()

features = [
    "angle",
    "num_holds",
    "diameter",
    "frames_count",
    "frames_pace",
    "has_start",
    "has_finish",
]

X = df[features]
y = df["difficulty_numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(df["frames_xy"].iloc[0])
print(df["frames"].iloc[0])