import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

data = pd.read_csv("/Users/gnwuha/Documents/AI:ML/projects/RentSmart/AB_NYC.csv")

X = data.drop('price', axis=1)
y = data['price']

numerical = X.select_dtypes(include=['number']).columns
categorical = X.select_dtypes(include=["object"]).columns

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])

cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoding', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('numerical', num_pipeline, numerical),
                                               ('categorical', cat_pipeline, categorical)])

tree_model = Pipeline([('preprocessing', preprocessor),
                       ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_model.fit(X_train, y_train)

st.title("RentSmart Price Predictor")
st.write("Input features to predict rent price:")

input_data = {}

# Numerical inputs
for col in numerical:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    input_data[col] = st.number_input(f"{col} (numeric)", min_value=min_val, max_value=max_val, value=mean_val)

# Categorical inputs
for col in categorical:
    options = X[col].dropna().unique().tolist()
    options = sorted(options)
    input_data[col] = st.selectbox(f"{col} (categorical)", options)

# Convert input to DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict Price"):
    prediction = tree_model.predict(input_df)[0]
    st.success(f"Predicted Rent Price: ${prediction:,.2f}")