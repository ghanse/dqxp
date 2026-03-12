from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType, TimestampType


MISMATCH_SCHEMA = StructType(
    [
        StructField("unique_key", StringType(), nullable=False),
        StructField("column_name", StringType(), nullable=False),
        StructField("source_value", StringType(), nullable=True),
        StructField("target_value", StringType(), nullable=True),
        StructField("mismatch_type", StringType(), nullable=False),
        StructField("table_name", StringType(), nullable=False),
        StructField("run_date", TimestampType(), nullable=False),
        StructField("result", StringType(), nullable=False),
    ]
)


class MismatchWriter:
    """Stub writer for the DQMismatch analysis table.

    Produces a DataFrame with the mismatch schema. Full comparison logic will be
    implemented in a future iteration.
    """

    schema = MISMATCH_SCHEMA

    def write(
        self,
        spark: SparkSession,
        source_df: DataFrame,
        target_df: DataFrame,
        key_columns: list[str],
        table_name: str,
    ) -> DataFrame:
        """Create an empty DataFrame with the DQMismatch schema.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame (unused in stub).
            target_df: Target DataFrame (unused in stub).
            key_columns: Columns forming the unique key (unused in stub).
            table_name: Destination table name (unused in stub).

        Returns:
            An empty DataFrame with the DQMismatch schema.
        """
        return spark.createDataFrame([], self.schema)
