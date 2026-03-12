from __future__ import annotations

from datetime import datetime, timezone
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
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
    """Writer for the DQMismatch analysis table.

    Compares source and target DataFrames row-by-row using primary key(s) and
    records column-level value mismatches, as well as rows present in only one
    side of the join.
    """

    schema = MISMATCH_SCHEMA

    def _build_composite_key(self, df: DataFrame, key_columns: list[str], prefix: str) -> DataFrame:
        """Add a ``unique_key`` column by concatenating key columns with ``|``."""
        # Build the key expression using prefixed column names
        prefixed_keys = [f"{prefix}_{c}" for c in key_columns]
        # Prefix all columns to avoid ambiguity after the join
        renamed = df.select(
            [F.col(c).alias(f"{prefix}_{c}") for c in df.columns]
        )
        if len(prefixed_keys) == 1:
            key_expr = F.col(prefixed_keys[0]).cast("string")
        else:
            key_expr = reduce(
                lambda a, b: F.concat(a, F.lit("|"), b),
                [F.col(c).cast("string") for c in prefixed_keys],
            )
        return renamed.withColumn("unique_key", key_expr)

    def write(
        self,
        spark: SparkSession,
        source_df: DataFrame,
        target_df: DataFrame,
        key_columns: list[str],
        table_name: str,
    ) -> DataFrame:
        """Compare *source_df* and *target_df* and return a mismatch DataFrame.

        Performs a full outer join on *key_columns*, then:
        - For matched rows: compares each non-key column and records ``VALUE_MISMATCH``.
        - For rows only in source: records ``TARGET_MISSING``.
        - For rows only in target: records ``SOURCE_MISSING``.

        All values are cast to strings. Composite keys are joined with ``|``.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame.
            target_df: Target DataFrame.
            key_columns: Columns forming the unique key.
            table_name: Destination table name stored in the output.

        Returns:
            A DataFrame conforming to :data:`MISMATCH_SCHEMA`.
        """
        run_date = datetime.now(tz=timezone.utc)

        # Determine non-key columns (use source columns as reference)
        all_columns = source_df.columns
        non_key_columns = [c for c in all_columns if c not in key_columns]

        # Prefix source and target to avoid column name collisions
        src = self._build_composite_key(source_df, key_columns, "src")
        tgt = self._build_composite_key(target_df, key_columns, "tgt")

        # Full outer join on composite key
        joined = src.join(tgt, on="unique_key", how="full_outer")

        # We need a marker to know which side is present
        # If any source key column is null -> source missing; if any target key column is null -> target missing
        src_present = F.col(f"src_{key_columns[0]}").isNotNull()
        tgt_present = F.col(f"tgt_{key_columns[0]}").isNotNull()

        mismatch_dfs: list[DataFrame] = []

        # --- VALUE_MISMATCH: both sides present, compare non-key columns ---
        matched = joined.filter(src_present & tgt_present)
        for col_name in non_key_columns:
            src_val = F.col(f"src_{col_name}").cast("string")
            tgt_val = F.col(f"tgt_{col_name}").cast("string")
            mismatched_rows = matched.filter(
                ~src_val.eqNullSafe(tgt_val)
            ).select(
                F.col("unique_key"),
                F.lit(col_name).alias("column_name"),
                src_val.alias("source_value"),
                tgt_val.alias("target_value"),
                F.lit("VALUE_MISMATCH").alias("mismatch_type"),
                F.lit(table_name).alias("table_name"),
                F.lit(run_date).alias("run_date"),
                F.lit("FAIL").alias("result"),
            )
            mismatch_dfs.append(mismatched_rows)

        # --- SOURCE_MISSING: row in target but not in source ---
        source_missing = joined.filter(~src_present & tgt_present)
        if non_key_columns:
            for col_name in non_key_columns:
                tgt_val = F.col(f"tgt_{col_name}").cast("string")
                sm_rows = source_missing.select(
                    F.col("unique_key"),
                    F.lit(col_name).alias("column_name"),
                    F.lit(None).cast("string").alias("source_value"),
                    tgt_val.alias("target_value"),
                    F.lit("SOURCE_MISSING").alias("mismatch_type"),
                    F.lit(table_name).alias("table_name"),
                    F.lit(run_date).alias("run_date"),
                    F.lit("FAIL").alias("result"),
                )
                mismatch_dfs.append(sm_rows)
        else:
            # No non-key columns; emit one record per missing row
            sm_rows = source_missing.select(
                F.col("unique_key"),
                F.lit(None).cast("string").alias("column_name"),
                F.lit(None).cast("string").alias("source_value"),
                F.lit(None).cast("string").alias("target_value"),
                F.lit("SOURCE_MISSING").alias("mismatch_type"),
                F.lit(table_name).alias("table_name"),
                F.lit(run_date).alias("run_date"),
                F.lit("FAIL").alias("result"),
            )
            mismatch_dfs.append(sm_rows)

        # --- TARGET_MISSING: row in source but not in target ---
        target_missing = joined.filter(src_present & ~tgt_present)
        if non_key_columns:
            for col_name in non_key_columns:
                src_val = F.col(f"src_{col_name}").cast("string")
                tm_rows = target_missing.select(
                    F.col("unique_key"),
                    F.lit(col_name).alias("column_name"),
                    src_val.alias("source_value"),
                    F.lit(None).cast("string").alias("target_value"),
                    F.lit("TARGET_MISSING").alias("mismatch_type"),
                    F.lit(table_name).alias("table_name"),
                    F.lit(run_date).alias("run_date"),
                    F.lit("FAIL").alias("result"),
                )
                mismatch_dfs.append(tm_rows)
        else:
            tm_rows = target_missing.select(
                F.col("unique_key"),
                F.lit(None).cast("string").alias("column_name"),
                F.lit(None).cast("string").alias("source_value"),
                F.lit(None).cast("string").alias("target_value"),
                F.lit("TARGET_MISSING").alias("mismatch_type"),
                F.lit(table_name).alias("table_name"),
                F.lit(run_date).alias("run_date"),
                F.lit("FAIL").alias("result"),
            )
            mismatch_dfs.append(tm_rows)

        if not mismatch_dfs:
            return spark.createDataFrame([], self.schema)

        result_df = reduce(DataFrame.unionByName, mismatch_dfs)
        return result_df
