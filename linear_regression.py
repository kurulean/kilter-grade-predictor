import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# create a connection to sqlite database
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


# calculate the absolute error of each v-grade
errors = abs(y_test - predictions)

grades = df.loc[y_test.index, "boulder_grade"]

results = pd.DataFrame({
    "grade": grades,
    "error": errors
})

grade_mae = results.groupby("grade")["error"].mean()

print(grade_mae)


# create a table showing the numeric difficulty for each v-grade
grade_mapping = (
    df[["difficulty_numeric", "boulder_grade"]]
    .dropna()
    .groupby("boulder_grade")["difficulty_numeric"]
    .median()
)

# convert each predicted difficulty into the nearest V-grade
def numeric_to_grade(value):
    closest_grade = (grade_mapping - value).abs().idxmin()
    return closest_grade

predicted_grades = [numeric_to_grade(value) for value in predictions]
actual_grades = df.loc[y_test.index, "boulder_grade"]

results = pd.DataFrame({
    "actual_grade": actual_grades,
    "predicted_grade": predicted_grades
})

print(results)

# exact grade accuracy
correct = (results["actual_grade"] == results["predicted_grade"]).mean()

print("Exact accuracy:", correct)


# convert grades like "6c/V5" -> 5
actual = (
    results["actual_grade"]
    .str.split("/")
    .str[-1]
    .str.replace("V", "")
    .astype(int)
)

predicted = (
    results["predicted_grade"]
    .str.split("/")
    .str[-1]
    .str.replace("V", "")
    .astype(int)
)


# Accuracy within one V-grade
within_one = (abs(actual - predicted) <= 1).mean()

print("Within 1 grade:", within_one)
