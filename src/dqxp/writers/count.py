from __future__ import annotations

from datetime import datetime, timezone

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
    """Writer for the DQCount analysis table.

    Compares row counts between source and target DataFrames and evaluates
    against a configurable threshold tolerance. Produces exactly one row.
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
        """Compare row counts between source and target DataFrames.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame to count.
            target_df: Target DataFrame to count.
            key_columns: Columns forming the unique key (unused for count).
            table_name: Destination table name recorded in output.
            threshold: Tolerance percentage (e.g. 0.05 for 5%). Defaults to 0.0.

        Returns:
            A single-row DataFrame conforming to COUNT_SCHEMA with the
            count comparison result.
        """
        run_date = datetime.now(tz=timezone.utc)

        source_count = source_df.count()
        target_count = target_df.count()
        difference = abs(source_count - target_count)

        result = self._evaluate(source_count, target_count, difference, threshold)

        row = (
            table_name,
            source_count,
            target_count,
            difference,
            float(threshold),
            result,
            run_date,
        )
        return spark.createDataFrame([row], self.schema)

    @staticmethod
    def _evaluate(
        source_count: int,
        target_count: int,
        difference: int,
        threshold: float,
    ) -> str:
        """Determine PASS or FAIL based on counts and threshold.

        Rules:
            1. Both zero: PASS
            2. One zero, other non-zero: FAIL
            3. Otherwise: PASS if (difference / max(source, target)) <= threshold
        """
        if source_count == 0 and target_count == 0:
            return "PASS"
        if source_count == 0 or target_count == 0:
            return "FAIL"
        ratio = difference / max(source_count, target_count)
        return "PASS" if ratio <= threshold else "FAIL"
