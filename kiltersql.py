"""
Kilter Grade Predictor - ML Model

Predict a climb's V-grade from its hold layout on the original Kilter Board.
Currently only working on the 7x10 layout for more simplicity.
First iteration will be a XGBoost regression model. The second iteration will be a either a GNN or CNN model.

"""

import sqlite3
import pandas as pd


# connect to sqlite database
conn = sqlite3.connect("kilter_data.sqlite")

# load the training table into a pandas data frame
df = pd.read_sql_query(
    "SELECT * FROM kilter_train",
    conn
)

# close the database connection
conn.close()


# inspect the data
print(df.head())
print(df.shape)
print(df.columns)
