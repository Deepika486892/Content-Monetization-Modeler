import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Sidebar Navigation
# -------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Multiple Linear Regression",
        "Random Forest Regression",
        "Elastic Net Regression",
        "Gradient Boosting Regression"
    ]
)

@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/user/Desktop/VS exc/VS Code/youtube_ad_revenue_dataset.csv")
    data = data.drop(["video_id", "date"], axis=1)

    # Fill missing values
    data["likes"].fillna(data["likes"].median(), inplace=True)
    data["comments"].fillna(data["comments"].mode()[0], inplace=True)
    data["watch_time_minutes"] = data["watch_time_minutes"].ffill()

    # Encode categorical columns
    le = LabelEncoder()
    data["country_encoded"] = le.fit_transform(data["country"])
    data["category_encoded"] = le.fit_transform(data["category"])
    data["device_encoded"] = le.fit_transform(data["device"])
    data.drop(columns=["country", "category", "device"], inplace=True)

    return data

data = load_data()

# -------------------------------
# Linear Regression
# -------------------------------
if page == "Linear Regression":
    st.title("Linear Regression Model Evaluation")

    X = data.drop(columns=["ad_revenue_usd"])
    y = data["ad_revenue_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write({"MAE": mae, "RMSE": rmse, "R²": r2*100})

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color="green")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig)

    # Table
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(results_df.head(20))

# -------------------------------
# Ridge Regression
# -------------------------------
elif page == "Ridge Regression":
    st.title("Ridge Regression Model Evaluation")

    X = data.drop(columns=["ad_revenue_usd"])
    y = data["ad_revenue_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Ridge Regression
    ridge_model = Ridge(alpha=0.01) 
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_r2 = r2_score(y_test, ridge_pred)
    

    st.subheader("Evaluation Metrics")
    st.write({"MAE": ridge_mae, "RMSE": ridge_rmse, "R²": ridge_r2*100})



    fig, ax = plt.subplots()
    ax.scatter(y_test, ridge_pred, color="blue", alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig)

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": ridge_pred})
    st.dataframe(results_df.head(20))

# -------------------------------
# Lasso Regression
# -------------------------------
elif page == "Lasso Regression":
    st.title("Lasso Regression Model Evaluation")

    X = data.drop(columns=["ad_revenue_usd"])
    y = data["ad_revenue_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lasso_model = Lasso(alpha=0.1, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)

    mae = mean_absolute_error(y_test, lasso_pred)
    mse = mean_squared_error(y_test, lasso_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, lasso_pred)

    st.subheader("Evaluation Metrics")
    st.write({"MAE": mae, "RMSE": rmse, "R²": r2*100})

    fig, ax = plt.subplots()
    ax.scatter(y_test, lasso_pred, color="purple", alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig)

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": lasso_pred})
    st.dataframe(results_df.head(20))

# -------------------------------
# Multiple Linear Regression
# -------------------------------
elif page == "Multiple Linear Regression":
    st.title("Multiple Linear Regression Model Evaluation")

    X = data[["comments", "watch_time_minutes", "likes"]]
    y = data["ad_revenue_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Evaluation Metrics")
    st.write({"R²": r2_score(y_test, y_pred)*100, "MSE": mean_squared_error(y_test, y_pred)})

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="blue", alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    st.pyplot(fig)

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(results_df.head(20))

# -------------------------------
# Random Forest Regression
# -------------------------------
elif page == "Random Forest Regression":
    st.title("Random Forest Regression Model Evaluation")

    X = data.drop(columns=["ad_revenue_usd"])
    y = data["ad_revenue_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write({"MAE": mae, "RMSE": rmse, "R²": r2*100})

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="orange", alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig)

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(results_df.head(20))

# -------------------------------
# Elastic Net Regression
# -------------------------------
elif page == "Elastic Net Regression":
    st.title("Elastic Net Regression Model Evaluation")

    X = data.drop(columns=["ad_revenue_usd"])
    y = data["ad_revenue_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=42)
    elastic_model.fit(X_train, y_train)
    y_pred = elastic_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write({"MAE": mae, "RMSE": rmse, "R²": r2*100})

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="blue", alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig)

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(results_df.head(20))

# -------------------------------
# Gradient Boosting Regression
# -------------------------------
elif page == "Gradient Boosting Regression":
    st.title("Gradient Boosting Regression Model Evaluation")

    # -------------------------------
    # Features and Target
    # -------------------------------
    X = data.drop(columns=["ad_revenue_usd"])
    y = data["ad_revenue_usd"]

    # -------------------------------
    # Train/Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Scaling
    # -------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # Gradient Boosting Model
    # -------------------------------
    gbr = GradientBoostingRegressor(
        n_estimators=50, max_depth=10, random_state=42
    )
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write({
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Squared Error (RMSE)": mse,
        "R² Score": r2*100
    })

    # -------------------------------
    # Visualization: Actual vs Predicted
    # -------------------------------
    st.subheader("Actual vs Predicted Revenue")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted vs Actual")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Fit")
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Gradient Boosting Regression: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Actual vs Predicted Table
    # -------------------------------
    results_df = pd.DataFrame({
        "Actual Revenue": y_test,
        "Predicted Revenue": y_pred
    })
    results_df["Difference"] = results_df["Actual Revenue"] - results_df["Predicted Revenue"]
    results_df = results_df.reset_index(drop=True)

    st.subheader("Actual vs Predicted Table (first 20 rows)")
    st.dataframe(results_df.head(20))

# -------------------------------
# Dark Mode Style
# -------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: black; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)
