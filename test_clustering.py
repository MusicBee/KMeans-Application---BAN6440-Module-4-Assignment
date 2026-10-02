import unittest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class TestKMeansClustering(unittest.TestCase):

    def setUp(self):

        # Small controlled dataset used for testing.
        # The first three stations are relatively warm/dry,
        # while the last three are relatively cool/wet.

        self.data = pd.DataFrame({
            "station_id": [
                "ST001",
                "ST002",
                "ST003",
                "ST004",
                "ST005",
                "ST006"
            ],

            "station_name": [
                "Station A",
                "Station B",
                "Station C",
                "Station D",
                "Station E",
                "Station F"
            ],

            "tmax_mean": [
                30.0,
                31.0,
                29.5,
                15.0,
                16.0,
                14.5
            ],

            "tmin_mean": [
                20.0,
                21.0,
                19.5,
                5.0,
                6.0,
                4.5
            ],

            "temperature_range": [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0
            ],

            "prcp_total": [
                500.0,
                550.0,
                480.0,
                1500.0,
                1600.0,
                1450.0
            ],

            "tmax_std": [
                5.0,
                5.2,
                4.8,
                6.0,
                6.1,
                5.9
            ],

            "tmin_std": [
                4.0,
                4.1,
                3.9,
                4.5,
                4.6,
                4.4
            ],

            "prcp_std": [
                10.0,
                11.0,
                9.5,
                20.0,
                21.0,
                19.0
            ],

            "tmax_tmin_correlation": [
                0.90,
                0.91,
                0.89,
                0.85,
                0.86,
                0.84
            ]
        })

        self.features = [
            "tmax_mean",
            "tmin_mean",
            "temperature_range",
            "prcp_total",
            "tmax_std",
            "tmin_std",
            "prcp_std",
            "tmax_tmin_correlation"
        ]


    # =====================================================
    # TEST 1: DATASET IS NOT EMPTY
    # =====================================================

    def test_dataset_not_empty(self):

        self.assertGreater(
            len(self.data),
            0
        )


    # =====================================================
    # TEST 2: REQUIRED FEATURES EXIST
    # =====================================================

    def test_required_features_exist(self):

        for feature in self.features:

            self.assertIn(
                feature,
                self.data.columns
            )


    # =====================================================
    # TEST 3: NO MISSING VALUES
    # =====================================================

    def test_no_missing_cluster_features(self):

        missing_count = (
            self.data[
                self.features
            ]
            .isna()
            .sum()
            .sum()
        )

        self.assertEqual(
            missing_count,
            0
        )


    # =====================================================
    # TEST 4: STANDARDIZATION
    # =====================================================

    def test_standardization(self):

        scaler = StandardScaler()

        X = scaler.fit_transform(
            self.data[
                self.features
            ]
        )

        # Verify dimensions

        self.assertEqual(
            X.shape,
            (
                len(self.data),
                len(self.features)
            )
        )

        # Verify approximately zero mean

        means = X.mean(
            axis=0
        )

        np.testing.assert_allclose(
            means,
            np.zeros(
                len(self.features)
            ),
            atol=1e-7
        )


    # =====================================================
    # TEST 5: K-MEANS RETURNS CLUSTERS
    # =====================================================

    def test_kmeans_returns_labels(self):

        scaler = StandardScaler()

        X = scaler.fit_transform(
            self.data[
                self.features
            ]
        )

        model = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X
        )

        self.assertEqual(
            len(labels),
            len(self.data)
        )


    # =====================================================
    # TEST 6: CORRECT NUMBER OF CLUSTERS
    # =====================================================

    def test_kmeans_creates_expected_number_of_clusters(self):

        scaler = StandardScaler()

        X = scaler.fit_transform(
            self.data[
                self.features
            ]
        )

        model = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X
        )

        unique_clusters = np.unique(
            labels
        )

        self.assertEqual(
            len(unique_clusters),
            2
        )


    # =====================================================
    # TEST 7: SILHOUETTE SCORE
    # =====================================================

    def test_silhouette_score_is_valid(self):

        scaler = StandardScaler()

        X = scaler.fit_transform(
            self.data[
                self.features
            ]
        )

        model = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X
        )

        score = silhouette_score(
            X,
            labels
        )

        self.assertGreaterEqual(
            score,
            -1
        )

        self.assertLessEqual(
            score,
            1
        )


    # =====================================================
    # TEST 8: CLUSTER LABELS ARE COMPLETE
    # =====================================================

    def test_cluster_labels_are_complete(self):

        scaler = StandardScaler()

        X = scaler.fit_transform(
            self.data[
                self.features
            ]
        )

        model = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X
        )

        self.assertFalse(
            np.isnan(labels).any()
        )


# =========================================================
# TEST RUNNER
# =========================================================

if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )