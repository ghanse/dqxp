from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType

DUPS_SCHEMA = StructType(
    [
        StructField("unique_key", StringType(), nullable=False),
        StructField("table_name", StringType(), nullable=False),
        StructField("dataset", StringType(), nullable=False),
        StructField("duplicate_count", IntegerType(), nullable=False),
        StructField("run_date", TimestampType(), nullable=False),
        StructField("result", StringType(), nullable=False),
    ]
)


class DupsWriter:
    """Stub writer for the DQDups analysis table.

    Produces a DataFrame with the duplicates schema. Full duplicate-detection
    logic will be implemented in a future iteration.
    """

    schema = DUPS_SCHEMA

    def write(
        self,
        spark: SparkSession,
        source_df: DataFrame,
        target_df: DataFrame,
        key_columns: list[str],
        table_name: str,
    ) -> DataFrame:
        """Create an empty DataFrame with the DQDups schema.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame (unused in stub).
            target_df: Target DataFrame (unused in stub).
            key_columns: Columns forming the unique key (unused in stub).
            table_name: Destination table name (unused in stub).

        Returns:
            An empty DataFrame with the DQDups schema.
        """
        return spark.createDataFrame([], self.schema)
