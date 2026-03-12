from __future__ import annotations

from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
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
    """Writer for the DQDups analysis table.

    Identifies duplicate records in both source and target DataFrames
    based on the provided primary key column(s).
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
        """Detect duplicates in source and target DataFrames.

        For each DataFrame, groups by key_columns, counts occurrences,
        and returns rows where the count exceeds 1. Composite keys are
        joined with ``|``.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame to check for duplicates.
            target_df: Target DataFrame to check for duplicates.
            key_columns: Columns forming the unique key.
            table_name: Destination table name recorded in output.

        Returns:
            A DataFrame conforming to DUPS_SCHEMA containing only
            duplicate records (count > 1) with result = "FAIL".
        """
        run_date = datetime.now(tz=timezone.utc)

        source_dups = self._find_duplicates(source_df, key_columns, "SOURCE", table_name, run_date)
        target_dups = self._find_duplicates(target_df, key_columns, "TARGET", table_name, run_date)

        result = source_dups.unionByName(target_dups)
        return spark.createDataFrame(result.collect(), self.schema)

    @staticmethod
    def _find_duplicates(
        df: DataFrame,
        key_columns: list[str],
        dataset: str,
        table_name: str,
        run_date: datetime,
    ) -> DataFrame:
        """Find duplicate keys in a single DataFrame.

        Args:
            df: The DataFrame to inspect.
            key_columns: Columns forming the unique key.
            dataset: Label for this dataset ("SOURCE" or "TARGET").
            table_name: Table name to include in output rows.
            run_date: Timestamp for the run_date column.

        Returns:
            DataFrame with duplicate rows conforming to DUPS_SCHEMA.
        """
        df_columns = set(df.columns)
        if not all(c in df_columns for c in key_columns):
            return df.sparkSession.createDataFrame([], DUPS_SCHEMA)

        key_expr = F.concat_ws("|", *[F.col(c).cast("string") for c in key_columns])

        grouped = (
            df.groupBy(key_columns)
            .agg(F.count("*").alias("cnt"))
            .filter(F.col("cnt") > 1)
        )

        return grouped.select(
            key_expr.alias("unique_key"),
            F.lit(table_name).alias("table_name"),
            F.lit(dataset).alias("dataset"),
            F.col("cnt").cast("int").alias("duplicate_count"),
            F.lit(run_date).alias("run_date"),
            F.lit("FAIL").alias("result"),
        )
