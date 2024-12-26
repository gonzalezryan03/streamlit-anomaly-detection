import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# ---------------------------------------
# STEP 1: Generate a Synthetic Dataset
# ---------------------------------------
def generate_synthetic_data(num_points=200, num_anomalies=10, random_state=42):
    """
    Generates synthetic 2D data with some random anomalies.
    
    :param num_points: Number of normal data points
    :param num_anomalies: Number of anomalies
    :param random_state: Random seed for reproducibility
    :return: A pandas DataFrame with columns ['feature1', 'feature2']
    """
    np.random.seed(random_state)
    
    # Generate normal data around (0,0)
    normal_data = 0.5 * np.random.randn(num_points, 2)
    
    # Generate some 'outliers' / anomalies far from the center
    anomalies = 5 * np.random.randn(num_anomalies, 2)
    anomalies += np.array([5, -5])  # Shift them away from center

    # Stack and create DataFrame
    data = np.vstack((normal_data, anomalies))
    df = pd.DataFrame(data, columns=["feature1", "feature2"])

    # Shuffle the DataFrame so anomalies aren’t all at the bottom
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

# ---------------------------------------
# STEP 2: Train an Isolation Forest
# ---------------------------------------
def train_isolation_forest(data, n_estimators=100, max_samples='auto', contamination=0.05, random_state=42):
    """
    Trains an Isolation Forest for anomaly detection.
    
    :param data: pandas DataFrame with columns ["feature1", "feature2"]
    :param n_estimators: Number of base estimators in the ensemble
    :param max_samples: Number of samples to draw from data
    :param contamination: The proportion of outliers in the data
    :param random_state: Random seed
    :return: Trained model, predictions, and anomaly scores
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(data[["feature1", "feature2"]])
    
    # Predict: -1 for anomaly, 1 for normal
    preds = model.predict(data[["feature1", "feature2"]])
    anomaly_scores = model.decision_function(data[["feature1", "feature2"]])
    
    return model, preds, anomaly_scores

# ---------------------------------------
# STEP 3: Streamlit Dashboard
# ---------------------------------------
def main():
    st.title("Anomaly Detection Dashboard")
    
    # Sidebar configuration
    st.sidebar.header("Model Settings")
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 300, 100, 10)
    contamination = st.sidebar.slider("Contamination (Outlier Proportion)", 0.01, 0.20, 0.05, 0.01)
    max_samples = st.sidebar.selectbox("Max Samples", ["auto", 50, 100, 200])
    
    # Generate synthetic data
    df = generate_synthetic_data(num_points=200, num_anomalies=10)
    
    # Train the Isolation Forest
    model, preds, scores = train_isolation_forest(
        df,
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination
    )
    
    # Append predictions and scores to the DataFrame
    df["prediction"] = preds
    df["anomaly_score"] = scores
    
    # Labeling: -1 = anomaly, 1 = normal
    df["label"] = df["prediction"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
    
    # Show data table
    st.subheader("Data Overview")
    st.write(df.head(15))  # Show first 15 rows
    
    # Plot the data
    st.subheader("Anomaly Visualization")
    fig = px.scatter(
        df,
        x="feature1",
        y="feature2",
        color="label",
        symbol="label",
        title="Detected Anomalies vs Normal Points"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    st.subheader("Summary")
    total_points = len(df)
    total_anomalies = df["label"].value_counts()["Anomaly"]
    st.write(f"**Total Points:** {total_points}")
    st.write(f"**Total Anomalies Detected:** {total_anomalies}")

if __name__ == "__main__":
    main()