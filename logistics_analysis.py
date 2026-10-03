"""
Logistics Data Analysis – YuvaIntern Internship Task
Intern: Iftiqаr Ahmad Khatana

This script demonstrates the strategic analysis workflow described
in the internship report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error


DATA_FILE = "data/logistics_data.csv"


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()

    date_columns = ["order_date", "promised_date", "actual_date"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["delivery_delay_days"] = (
        df["actual_date"] - df["promised_date"]
    ).dt.days

    df["cost_per_km"] = (
        df["transport_cost"]
        / df["distance_km"].replace(0, np.nan)
    )

    df["on_time"] = df["delivery_delay_days"] <= 0
    return df


def calculate_kpis(df):
    on_time_rate = df["on_time"].mean() * 100
    avg_delivery_time = df["delivery_time_hours"].mean()
    avg_transport_cost = df["transport_cost"].mean()

    fulfillment_rate = (
        df["fulfilled"].mean() * 100
        if "fulfilled" in df.columns
        else np.nan
    )

    vehicle_utilization = (
        df["order_quantity"] / df["vehicle_capacity"]
    ).mean() * 100

    return {
        "On-time delivery rate (%)": on_time_rate,
        "Average delivery time (hours)": avg_delivery_time,
        "Average transport cost": avg_transport_cost,
        "Order fulfillment rate (%)": fulfillment_rate,
        "Average vehicle utilization (%)": vehicle_utilization,
    }


def run_regression(df):
    features = ["distance_km", "order_quantity", "vehicle_capacity"]
    target = "delivery_time_hours"

    model_data = df[features + [target]].dropna()

    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, mae


def run_clustering(df):
    features = ["distance_km", "order_quantity", "transport_cost"]
    cluster_data = df[features].dropna().copy()

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data["cluster"] = model.fit_predict(cluster_data)

    return cluster_data


def make_visualization(df):
    counts = df["on_time"].value_counts()
    counts.index = ["Late", "On Time"][:len(counts)]

    counts.plot(kind="bar", title="Delivery Performance")
    plt.xlabel("Delivery Status")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig("delivery_performance.png", dpi=150)
    plt.show()


def main():
    df = load_and_prepare_data(DATA_FILE)

    print("\n=== LOGISTICS KPI RESULTS ===")
    kpis = calculate_kpis(df)
    for name, value in kpis.items():
        if pd.isna(value):
            print(f"{name}: N/A")
        else:
            print(f"{name}: {value:.2f}")

    print("\n=== REGRESSION ===")
    _, mae = run_regression(df)
    print(f"Mean Absolute Error: {mae:.2f} hours")

    print("\n=== CLUSTERING ===")
    clusters = run_clustering(df)
    print(clusters.groupby("cluster").mean(numeric_only=True))

    make_visualization(df)
    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
