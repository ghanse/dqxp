from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType, TimestampType


COUNT_SCHEMA = StructType(
    [
        StructField("table_name", StringType(), nullable=False),
        StructField("source_count", LongType(), nullable=False),
        StructField("target_count", LongType(), nullable=False),
        StructField("count_difference", LongType(), nullable=False),
        StructField("threshold", DoubleType(), nullable=False),
        StructField("result", StringType(), nullable=False),
        StructField("run_date", TimestampType(), nullable=False),
    ]
)


class CountWriter:
    """Stub writer for the DQCount analysis table.

    Produces a DataFrame with the count comparison schema. Full row-count
    comparison logic will be implemented in a future iteration.
    """

    schema = COUNT_SCHEMA

    def write(
        self,
        spark: SparkSession,
        source_df: DataFrame,
        target_df: DataFrame,
        key_columns: list[str],
        table_name: str,
        threshold: float = 0.0,
    ) -> DataFrame:
        """Create an empty DataFrame with the DQCount schema.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame (unused in stub).
            target_df: Target DataFrame (unused in stub).
            key_columns: Columns forming the unique key (unused in stub).
            table_name: Destination table name (unused in stub).
            threshold: Count difference threshold (unused in stub).

        Returns:
            An empty DataFrame with the DQCount schema.
        """
        return spark.createDataFrame([], self.schema)
