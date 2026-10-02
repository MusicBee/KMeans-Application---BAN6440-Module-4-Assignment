import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4"

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs"
)

KMEANS_DIR = os.path.join(
    OUTPUT_DIR,
    "kmeans_clustering"
)

ANALYSIS_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_50_station_analysis_2020_2024.csv"
)

STATION_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_station_summary_2020_2024.csv"
)

os.makedirs(
    KMEANS_DIR,
    exist_ok=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def save_plot(filename):
    filepath = os.path.join(
        KMEANS_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {filepath}")


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\n" + "=" * 80)
    print("LOADING WEATHER DATA")
    print("=" * 80)

    if not os.path.exists(ANALYSIS_FILE):

        raise FileNotFoundError(
            f"Analysis file not found:\n{ANALYSIS_FILE}"
        )

    if not os.path.exists(STATION_SUMMARY_FILE):

        raise FileNotFoundError(
            f"Station summary file not found:\n"
            f"{STATION_SUMMARY_FILE}"
        )

    analysis = pd.read_csv(
        ANALYSIS_FILE
    )

    station_summary = pd.read_csv(
        STATION_SUMMARY_FILE
    )

    analysis["date"] = pd.to_datetime(
        analysis["date"],
        errors="coerce"
    )

    print(
        f"Daily records       : {len(analysis):,}"
    )

    print(
        f"Stations             : "
        f"{analysis['station_id'].nunique()}"
    )

    return analysis, station_summary


# =========================================================
# CREATE CLUSTER FEATURES
# =========================================================

def create_cluster_features(
    analysis,
    station_summary
):

    print("\n" + "=" * 80)
    print("CREATING STATION-LEVEL CLUSTER FEATURES")
    print("=" * 80)

    # -----------------------------------------------------
    # Calculate station-level variability
    # -----------------------------------------------------

    variability = (
        analysis
        .groupby(
            [
                "station_id",
                "station_name"
            ],
            as_index=False
        )
        .agg(
            tmax_std=("tmax", "std"),
            tmin_std=("tmin", "std"),
            prcp_std=("prcp", "std")
        )
    )

    # -----------------------------------------------------
    # Calculate TMAX/TMIN correlation by station
    # -----------------------------------------------------

    correlations = []

    for station_id, group in analysis.groupby(
        "station_id"
    ):

        valid = group[
            [
                "tmax",
                "tmin"
            ]
        ].dropna()

        if len(valid) >= 2:

            correlation = (
                valid["tmax"]
                .corr(valid["tmin"])
            )

        else:

            correlation = np.nan

        station_name = (
            group["station_name"]
            .iloc[0]
        )

        correlations.append(
            [
                station_id,
                station_name,
                correlation
            ]
        )

    correlations = pd.DataFrame(
        correlations,
        columns=[
            "station_id",
            "station_name",
            "tmax_tmin_correlation"
        ]
    )

    # -----------------------------------------------------
    # Merge station summary with variability
    # -----------------------------------------------------

    features = station_summary.merge(
        variability,
        on=[
            "station_id",
            "station_name"
        ],
        how="left"
    )

    # -----------------------------------------------------
    # Merge TMAX/TMIN correlations
    # -----------------------------------------------------

    features = features.merge(
        correlations,
        on=[
            "station_id",
            "station_name"
        ],
        how="left"
    )

    # -----------------------------------------------------
    # Calculate temperature range
    # -----------------------------------------------------

    features["temperature_range"] = (
        features["tmax_mean"]
        - features["tmin_mean"]
    )

    # -----------------------------------------------------
    # Define clustering features
    # -----------------------------------------------------

    cluster_features = [
        "tmax_mean",
        "tmin_mean",
        "temperature_range",
        "prcp_total",
        "tmax_std",
        "tmin_std",
        "prcp_std",
        "tmax_tmin_correlation"
    ]

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in cluster_features
        if column not in features.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing clustering columns:\n"
            + "\n".join(missing_columns)
        )

    # -----------------------------------------------------
    # Select final clustering dataset
    # -----------------------------------------------------

    cluster_data = features[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude"
        ]
        + cluster_features
    ].copy()

    # -----------------------------------------------------
    # Check missing values
    # -----------------------------------------------------

    print("\nMissing values before cleaning:")

    print(
        cluster_data[
            cluster_features
        ]
        .isna()
        .sum()
        .to_string()
    )

    # -----------------------------------------------------
    # Fill missing values with feature median
    # -----------------------------------------------------

    for column in cluster_features:

        cluster_data[column] = (
            cluster_data[column]
            .fillna(
                cluster_data[column].median()
            )
        )

    # -----------------------------------------------------
    # Display feature information
    # -----------------------------------------------------

    print("\nClustering features:")

    for feature in cluster_features:

        print(
            f"  - {feature}"
        )

    print(
        f"\nStations available for clustering: "
        f"{len(cluster_data)}"
    )

    # -----------------------------------------------------
    # Save feature dataset
    # -----------------------------------------------------

    cluster_data.to_csv(
        os.path.join(
            KMEANS_DIR,
            "station_cluster_features.csv"
        ),
        index=False
    )

    print(
        "\nSaved station cluster features:"
    )

    print(
        os.path.join(
            KMEANS_DIR,
            "station_cluster_features.csv"
        )
    )

    return cluster_data, cluster_features

# =========================================================
# STANDARDIZE DATA
# =========================================================

def standardize_features(
    cluster_data,
    cluster_features
):

    print("\n" + "=" * 80)
    print("STANDARDIZING CLUSTER FEATURES")
    print("=" * 80)

    scaler = StandardScaler()

    X = scaler.fit_transform(
        cluster_data[
            cluster_features
        ]
    )

    scaled_data = pd.DataFrame(
        X,
        columns=cluster_features
    )

    scaled_data.to_csv(
        os.path.join(
            KMEANS_DIR,
            "standardized_cluster_features.csv"
        ),
        index=False
    )

    print(
        f"Features standardized: "
        f"{X.shape[1]}"
    )

    print(
        f"Stations standardized: "
        f"{X.shape[0]}"
    )

    return X, scaler


# =========================================================
# FIND OPTIMAL K
# =========================================================

def evaluate_k_values(X):

    print("\n" + "=" * 80)
    print("EVALUATING NUMBER OF CLUSTERS")
    print("=" * 80)

    results = []

    # K should be smaller than the number of stations

    k_values = range(
        2,
        11
    )

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X
        )

        inertia = model.inertia_

        silhouette = silhouette_score(
            X,
            labels
        )

        results.append(
            [
                k,
                inertia,
                silhouette
            ]
        )

        print(
            f"K={k:2d} | "
            f"Inertia={inertia:10.2f} | "
            f"Silhouette={silhouette:.4f}"
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "k",
            "inertia",
            "silhouette_score"
        ]
    )

    results_df.to_csv(
        os.path.join(
            KMEANS_DIR,
            "k_evaluation.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # Select K based on highest silhouette score
    # -----------------------------------------------------

    best_row = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    best_k = int(
        best_row["k"]
    )

    print(
        f"\nSelected K based on "
        f"highest silhouette score: {best_k}"
    )

    print(
        f"Best silhouette score: "
        f"{best_row['silhouette_score']:.4f}"
    )

    # -----------------------------------------------------
    # Elbow plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results_df["k"],
        results_df["inertia"],
        marker="o"
    )

    plt.title(
        "K-Means Elbow Analysis"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Within-Cluster Sum of Squares (Inertia)"
    )

    plt.xticks(
        results_df["k"]
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "01_elbow_analysis.png"
    )

    # -----------------------------------------------------
    # Silhouette plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results_df["k"],
        results_df["silhouette_score"],
        marker="o"
    )

    plt.title(
        "Silhouette Score by Number of Clusters"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.xticks(
        results_df["k"]
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "02_silhouette_analysis.png"
    )

    return best_k, results_df


# =========================================================
# RUN K-MEANS
# =========================================================

def run_kmeans(
    cluster_data,
    X,
    best_k
):

    print("\n" + "=" * 80)
    print("RUNNING K-MEANS CLUSTERING")
    print("=" * 80)

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(
        X
    )

    result = cluster_data.copy()

    result["cluster"] = labels

    # -----------------------------------------------------
    # Make cluster numbering easier to interpret
    # -----------------------------------------------------

    result["cluster"] = (
        result["cluster"]
        .astype(int)
    )

    result.to_csv(
        os.path.join(
            KMEANS_DIR,
            "station_clusters.csv"
        ),
        index=False
    )

    print(
        f"\nK-Means completed with "
        f"{best_k} clusters."
    )

    print("\nStations per cluster:")

    print(
        result["cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return model, result


# =========================================================
# CLUSTER PROFILE
# =========================================================

def create_cluster_profile(
    result,
    cluster_features
):

    print("\n" + "=" * 80)
    print("CREATING CLUSTER PROFILES")
    print("=" * 80)

    profile = (
        result
        .groupby("cluster")
        .agg(
            station_count=(
                "station_id",
                "count"
            ),
            latitude_mean=(
                "latitude",
                "mean"
            ),
            longitude_mean=(
                "longitude",
                "mean"
            ),
            tmax_mean=(
                "tmax_mean",
                "mean"
            ),
            tmin_mean=(
                "tmin_mean",
                "mean"
            ),
            temperature_range=(
                "temperature_range",
                "mean"
            ),
            prcp_total=(
                "prcp_total",
                "mean"
            ),
            tmax_std=(
                "tmax_std",
                "mean"
            ),
            tmin_std=(
                "tmin_std",
                "mean"
            ),
            prcp_std=(
                "prcp_std",
                "mean"
            ),
            tmax_tmin_correlation=(
                "tmax_tmin_correlation",
                "mean"
            )
        )
        .reset_index()
    )

    profile.to_csv(
        os.path.join(
            KMEANS_DIR,
            "cluster_profiles.csv"
        ),
        index=False
    )

    print(
        profile.to_string(
            index=False
        )
    )

    return profile


# =========================================================
# CREATE CLUSTER CHARACTERISTICS
# =========================================================

def create_cluster_interpretation(
    result,
    profile
):

    print("\n" + "=" * 80)
    print("CLUSTER INTERPRETATION")
    print("=" * 80)

    interpretation = []

    overall_tmax = result[
        "tmax_mean"
    ].mean()

    overall_tmin = result[
        "tmin_mean"
    ].mean()

    overall_prcp = result[
        "prcp_total"
    ].mean()

    overall_range = result[
        "temperature_range"
    ].mean()

    for _, row in profile.iterrows():

        cluster = int(
            row["cluster"]
        )

        # -------------------------------------------------
        # Temperature classification
        # -------------------------------------------------

        if row["tmax_mean"] >= overall_tmax * 1.05:

            temperature_type = "Warmer"

        elif row["tmax_mean"] <= overall_tmax * 0.95:

            temperature_type = "Cooler"

        else:

            temperature_type = "Moderate"

        # -------------------------------------------------
        # Precipitation classification
        # -------------------------------------------------

        if row["prcp_total"] >= overall_prcp * 1.25:

            precipitation_type = "Wet"

        elif row["prcp_total"] <= overall_prcp * 0.75:

            precipitation_type = "Dry"

        else:

            precipitation_type = "Moderate precipitation"

        # -------------------------------------------------
        # Temperature range
        # -------------------------------------------------

        if row["temperature_range"] >= overall_range * 1.10:

            range_type = "Large temperature range"

        elif row["temperature_range"] <= overall_range * 0.90:

            range_type = "Small temperature range"

        else:

            range_type = "Moderate temperature range"

        description = (
            f"{temperature_type}; "
            f"{precipitation_type}; "
            f"{range_type}"
        )

        interpretation.append(
            [
                cluster,
                int(row["station_count"]),
                temperature_type,
                precipitation_type,
                range_type,
                description
            ]
        )

    interpretation_df = pd.DataFrame(
        interpretation,
        columns=[
            "cluster",
            "station_count",
            "temperature_class",
            "precipitation_class",
            "temperature_range_class",
            "cluster_description"
        ]
    )

    interpretation_df.to_csv(
        os.path.join(
            KMEANS_DIR,
            "cluster_interpretation.csv"
        ),
        index=False
    )

    print(
        interpretation_df.to_string(
            index=False
        )
    )

    return interpretation_df


# =========================================================
# PCA VISUALIZATION
# =========================================================

def create_pca_visualization(
    X,
    result
):

    print("\n" + "=" * 80)
    print("CREATING PCA CLUSTER VISUALIZATION")
    print("=" * 80)

    pca = PCA(
        n_components=2
    )

    components = pca.fit_transform(
        X
    )

    pca_df = pd.DataFrame(
        components,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    pca_df["cluster"] = (
        result["cluster"]
        .values
    )

    pca_df["station_name"] = (
        result["station_name"]
        .values
    )

    pca_df.to_csv(
        os.path.join(
            KMEANS_DIR,
            "pca_coordinates.csv"
        ),
        index=False
    )

    explained_variance = (
        pca.explained_variance_ratio_
        * 100
    )

    print(
        f"PC1 variance explained: "
        f"{explained_variance[0]:.2f}%"
    )

    print(
        f"PC2 variance explained: "
        f"{explained_variance[1]:.2f}%"
    )

    print(
        f"Combined variance explained: "
        f"{explained_variance.sum():.2f}%"
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    for cluster in sorted(
        pca_df["cluster"].unique()
    ):

        subset = pca_df[
            pca_df["cluster"] == cluster
        ]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            s=80,
            label=f"Cluster {cluster}"
        )

        for _, row in subset.iterrows():

            plt.annotate(
                row["station_name"],
                (
                    row["PC1"],
                    row["PC2"]
                ),
                fontsize=7,
                alpha=0.7
            )

    plt.title(
        "K-Means Clustering of Texas Weather Stations"
    )

    plt.xlabel(
        f"PC1 ({explained_variance[0]:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1]:.1f}% variance)"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "03_pca_cluster_visualization.png"
    )

    return pca


# =========================================================
# GEOGRAPHIC CLUSTER VISUALIZATION
# =========================================================

def create_geographic_visualization(
    result
):

    print("\nCreating geographic cluster visualization...")

    plt.figure(
        figsize=(12, 8)
    )

    for cluster in sorted(
        result["cluster"].unique()
    ):

        subset = result[
            result["cluster"] == cluster
        ]

        plt.scatter(
            subset["longitude"],
            subset["latitude"],
            s=90,
            label=f"Cluster {cluster}"
        )

    plt.title(
        "Geographic Distribution of K-Means Clusters"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "04_geographic_clusters.png"
    )


# =========================================================
# CLUSTER SIZE VISUALIZATION
# =========================================================

def create_cluster_size_plot(
    result
):

    counts = (
        result["cluster"]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.title(
        "Number of Stations per Cluster"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Number of Stations"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "05_cluster_sizes.png"
    )


# =========================================================
# SAVE STATION LISTS
# =========================================================

def save_cluster_station_lists(
    result
):

    print("\nSaving station lists by cluster...")

    for cluster in sorted(
        result["cluster"].unique()
    ):

        cluster_data = result[
            result["cluster"] == cluster
        ].sort_values(
            "station_name"
        )

        filename = (
            f"cluster_{cluster}_stations.csv"
        )

        cluster_data.to_csv(
            os.path.join(
                KMEANS_DIR,
                filename
            ),
            index=False
        )

        print(
            f"  Saved: {filename}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("K-MEANS WEATHER STATION CLUSTERING APPLICATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    analysis, station_summary = load_data()

    # -----------------------------------------------------
    # Create station-level features
    # -----------------------------------------------------

    cluster_data, cluster_features = (
        create_cluster_features(
            analysis,
            station_summary
        )
    )

    # -----------------------------------------------------
    # Standardize
    # -----------------------------------------------------

    X, scaler = standardize_features(
        cluster_data,
        cluster_features
    )

    # -----------------------------------------------------
    # Evaluate K
    # -----------------------------------------------------

    best_k, k_results = evaluate_k_values(
        X
    )

    # -----------------------------------------------------
    # K-Means
    # -----------------------------------------------------

    model, result = run_kmeans(
        cluster_data,
        X,
        best_k
    )

    # -----------------------------------------------------
    # Cluster profiles
    # -----------------------------------------------------

    profile = create_cluster_profile(
        result,
        cluster_features
    )

    # -----------------------------------------------------
    # Cluster interpretation
    # -----------------------------------------------------

    create_cluster_interpretation(
        result,
        profile
    )

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------

    create_pca_visualization(
        X,
        result
    )

    # -----------------------------------------------------
    # Geographic visualization
    # -----------------------------------------------------

    create_geographic_visualization(
        result
    )

    # -----------------------------------------------------
    # Cluster size
    # -----------------------------------------------------

    create_cluster_size_plot(
        result
    )

    # -----------------------------------------------------
    # Station lists
    # -----------------------------------------------------

    save_cluster_station_lists(
        result
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("K-MEANS CLUSTERING COMPLETED")
    print("=" * 80)

    print(
        f"\nSelected number of clusters: {best_k}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        KMEANS_DIR
    )

    print(
        "\nGenerated files:"
    )

    for filename in sorted(
        os.listdir(KMEANS_DIR)
    ):

        print(
            f"  - {filename}"
        )

    print("\n" + "=" * 80)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
