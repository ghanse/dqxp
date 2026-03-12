from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType


META_SCHEMA = StructType(
    [
        StructField("table_name", StringType(), nullable=False),
        StructField("source_column_name", StringType(), nullable=False),
        StructField("target_column_name", StringType(), nullable=True),
        StructField("source_data_type", StringType(), nullable=False),
        StructField("target_data_type", StringType(), nullable=True),
        StructField("source_column_count", IntegerType(), nullable=False),
        StructField("target_column_count", IntegerType(), nullable=False),
        StructField("result", StringType(), nullable=False),
        StructField("run_date", TimestampType(), nullable=False),
    ]
)


class MetaWriter:
    """Stub writer for the DQMeta analysis table.

    Produces a DataFrame with the meta/schema validation schema. Full comparison
    logic will be implemented in a future iteration.
    """

    schema = META_SCHEMA

    def write(
        self,
        spark: SparkSession,
        source_df: DataFrame,
        target_df: DataFrame,
        key_columns: list[str],
        table_name: str,
    ) -> DataFrame:
        """Create an empty DataFrame with the DQMeta schema.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame (unused in stub).
            target_df: Target DataFrame (unused in stub).
            key_columns: Columns forming the unique key (unused in stub).
            table_name: Destination table name (unused in stub).

        Returns:
            An empty DataFrame with the DQMeta schema.
        """
        return spark.createDataFrame([], self.schema)
