from __future__ import annotations

from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType

META_SCHEMA = StructType(
    [
        StructField("table_name", StringType(), nullable=False),
        StructField("source_column_name", StringType(), nullable=True),
        StructField("target_column_name", StringType(), nullable=True),
        StructField("source_data_type", StringType(), nullable=True),
        StructField("target_data_type", StringType(), nullable=True),
        StructField("source_column_count", IntegerType(), nullable=False),
        StructField("target_column_count", IntegerType(), nullable=False),
        StructField("result", StringType(), nullable=False),
        StructField("run_date", TimestampType(), nullable=False),
    ]
)


class MetaWriter:
    """Writer for the DQMeta analysis table.

    Validates schema compatibility between source and target DataFrames by
    comparing column names, data types, and column counts. Produces one row
    per column found in either source or target.
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
        """Compare schemas of *source_df* and *target_df* column-by-column.

        Matching logic:
        - Columns are matched by name.
        - For matched columns: PASS if data types match and column counts match,
          FAIL otherwise.
        - For columns in source but not target: FAIL with target fields as None.
        - For columns in target but not source: FAIL with source fields as None.
        - ``source_column_count`` and ``target_column_count`` reflect the total
          number of columns in each table and are the same on every row.

        Args:
            spark: Active SparkSession.
            source_df: Source DataFrame.
            target_df: Target DataFrame.
            key_columns: Columns forming the unique key (unused for schema comparison).
            table_name: Destination table name stored in the output.

        Returns:
            A DataFrame conforming to :data:`META_SCHEMA`.
        """
        run_date = datetime.now(tz=timezone.utc)

        source_fields = {f.name: f.dataType.simpleString() for f in source_df.schema.fields}
        target_fields = {f.name: f.dataType.simpleString() for f in target_df.schema.fields}

        source_col_count = len(source_df.columns)
        target_col_count = len(target_df.columns)
        counts_match = source_col_count == target_col_count

        all_columns = list(dict.fromkeys(list(source_fields.keys()) + list(target_fields.keys())))

        rows = []
        for col_name in all_columns:
            src_type = source_fields.get(col_name)
            tgt_type = target_fields.get(col_name)

            in_source = src_type is not None
            in_target = tgt_type is not None

            if in_source and in_target:
                result = "PASS" if (src_type == tgt_type and counts_match) else "FAIL"
                rows.append((
                    table_name,
                    col_name,
                    col_name,
                    src_type,
                    tgt_type,
                    source_col_count,
                    target_col_count,
                    result,
                    run_date,
                ))
            elif in_source and not in_target:
                rows.append((
                    table_name,
                    col_name,
                    None,
                    src_type,
                    None,
                    source_col_count,
                    target_col_count,
                    "FAIL",
                    run_date,
                ))
            else:
                rows.append((
                    table_name,
                    None,
                    col_name,
                    None,
                    tgt_type,
                    source_col_count,
                    target_col_count,
                    "FAIL",
                    run_date,
                ))

        if not rows:
            return spark.createDataFrame([], self.schema)

        return spark.createDataFrame(rows, self.schema)
