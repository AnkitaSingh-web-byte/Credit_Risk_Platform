"""
query_runner.py

Execute SQL queries generated
by Gemini against Home Credit data.
"""

import sqlite3
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "application_train.csv"
)


class QueryRunner:

    def __init__(self):

        self.conn = sqlite3.connect(
            ":memory:"
        )

        self.load_data()

    def load_data(self):

        print(
            "Loading application_train.csv..."
        )

        df = pd.read_csv(
            DATA_PATH
        )

        df.to_sql(
            "application_train",
            self.conn,
            index=False,
            if_exists="replace"
        )

        print(
            f"Loaded {len(df):,} rows"
        )

    def run_query(
        self,
        sql_query
    ):

        try:

            result = pd.read_sql_query(
                sql_query,
                self.conn
            )

            return result

        except Exception as e:

            print(
                f"SQL Error: {e}"
            )

            return pd.DataFrame()

    def close(self):

        self.conn.close()


if __name__ == "__main__":

    runner = QueryRunner()

    query = """
   SELECT
    COUNT(*) AS total_rows,
    AVG(AMT_INCOME_TOTAL) AS avg_income,
    MAX(AMT_CREDIT) AS max_credit
FROM application_train
    """

    result = runner.run_query(
        query
    )

    print(result)

    runner.close()