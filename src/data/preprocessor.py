"""
preprocessor.py

Data cleaning and preprocessing for
Home Credit Default Risk dataset.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class CreditDataPreprocessor:

    def __init__(self):

        self.numeric_columns = None
        self.categorical_columns = None

        self.numeric_imputer = SimpleImputer(
            strategy="median"
        )

        self.categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    def fit(self, df):

        df = self._basic_cleaning(df)

        self.numeric_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        if "TARGET" in self.numeric_columns:
            self.numeric_columns.remove("TARGET")

        self.categorical_columns = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # Fit imputers

        self.numeric_imputer.fit(
            df[self.numeric_columns]
        )

        self.categorical_imputer.fit(
            df[self.categorical_columns]
        )

        cat_data = self.categorical_imputer.transform(
            df[self.categorical_columns]
        )

        self.encoder.fit(cat_data)

        return self

    def transform(self, df):

        df = self._basic_cleaning(df)

        numeric_data = self.numeric_imputer.transform(
            df[self.numeric_columns]
        )

        categorical_data = self.categorical_imputer.transform(
            df[self.categorical_columns]
        )

        categorical_data = self.encoder.transform(
            categorical_data
        )

        numeric_df = pd.DataFrame(
            numeric_data,
            columns=self.numeric_columns,
            index=df.index
        )

        categorical_df = pd.DataFrame(
            categorical_data,
            columns=self.categorical_columns,
            index=df.index
        )

        processed_df = pd.concat(
            [numeric_df, categorical_df],
            axis=1
        )

        return processed_df

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)

    def save(self, path):

        joblib.dump(self, path)

    @staticmethod
    def load(path):

        return joblib.load(path)

    def _basic_cleaning(self, df):

        df = df.copy()

        # Home Credit anomaly

        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df[
                "DAYS_EMPLOYED"
            ].replace(
                365243,
                np.nan
            )

        # Convert birth days to age

        if "DAYS_BIRTH" in df.columns:

            df["AGE_YEARS"] = (
                abs(df["DAYS_BIRTH"]) / 365
            ).round(1)

        return df