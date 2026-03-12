from __future__ import annotations

import logging

from databricks.labs.dqx.config import OutputConfig
from databricks.labs.dqx.engine import DQEngineCore
from databricks.labs.dqx.rule import DQRule
from databricks.labs.dqx.utils import save_dataframe_as_table
from pyspark.sql import DataFrame

from dqxp.writers.count import CountWriter
from dqxp.writers.dups import DupsWriter
from dqxp.writers.meta import MetaWriter
from dqxp.writers.mismatch import MismatchWriter

logger = logging.getLogger(__name__)


class DQEngineExtension:
    """Extension for DQX that produces additional data quality analysis tables.

    Wraps a DQEngineCore instance and, after running DQX checks, computes and
    writes four analysis tables: DQMismatch, DQMeta, DQDups, and DQCount.

    Args:
        engine: A DQEngineCore instance used to run data quality checks.
        mismatch_table_name: Fully-qualified name of the mismatch output table.
        schema_validation_table_name: Fully-qualified name of the meta/schema output table.
        duplicate_count_table_name: Fully-qualified name of the duplicates output table.
        row_count_table_name: Fully-qualified name of the count comparison output table.
    """

    def __init__(
        self,
        engine: DQEngineCore,
        mismatch_table_name: str,
        schema_validation_table_name: str,
        duplicate_count_table_name: str,
        row_count_table_name: str,
    ):
        self._engine = engine
        self._mismatch_table = mismatch_table_name
        self._meta_table = schema_validation_table_name
        self._dups_table = duplicate_count_table_name
        self._count_table = row_count_table_name

        self._mismatch_writer = MismatchWriter()
        self._meta_writer = MetaWriter()
        self._dups_writer = DupsWriter()
        self._count_writer = CountWriter()

    @property
    def engine(self) -> DQEngineCore:
        return self._engine

    def apply_checks_and_save_output_tables(
        self,
        source_df: DataFrame,
        target_df: DataFrame,
        checks: list[DQRule],
        key_columns: list[str],
        output_table: str | None = None,
        quarantine_table: str | None = None,
        ref_dfs: dict[str, DataFrame] | None = None,
        threshold: float = 0.0,
    ) -> dict[str, DataFrame]:
        """Run DQX checks on source_df and produce all analysis tables.

        Steps:
            1. Run DQX checks via ``engine.apply_checks_and_split`` on *source_df*.
            2. Optionally persist good records to *output_table*.
            3. Optionally persist quarantined (bad) records to *quarantine_table*.
            4. Compute and write mismatch, meta, dups, and count analysis tables.

        Args:
            source_df: The source DataFrame to validate.
            target_df: The target DataFrame to compare against.
            checks: List of DQRule checks to apply via DQX.
            key_columns: Column names that together form the unique key for records.
            output_table: Optional table name where good (passing) records are saved.
            quarantine_table: Optional table name where bad (failing) records are saved.
            ref_dfs: Optional dict of reference DataFrames passed through to DQX.
            threshold: Acceptable count difference threshold (0.0 = exact match).

        Returns:
            A dict mapping table kind to the DataFrame that was written:
            ``{"mismatch": ..., "meta": ..., "dups": ..., "count": ...}``.

        Raises:
            ValueError: If key_columns is empty or threshold is negative.
        """
        if not key_columns:
            raise ValueError("key_columns must not be empty")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        spark = self._engine.spark

        extra_kwargs: dict = {}
        if ref_dfs is not None:
            extra_kwargs["ref_dfs"] = ref_dfs
        result = self._engine.apply_checks_and_split(source_df, checks, **extra_kwargs)
        good_df = result[0]
        bad_df = result[1]

        bad_count = bad_df.count()
        good_count = good_df.count()
        logger.info("DQX checks complete: %d good, %d bad", good_count, bad_count)

        if output_table:
            logger.info("Saving %d good records to %s", good_count, output_table)
            save_dataframe_as_table(good_df, OutputConfig(location=output_table))

        if quarantine_table and bad_count > 0:
            logger.info("Saving %d bad records to %s", bad_count, quarantine_table)
            save_dataframe_as_table(bad_df, OutputConfig(location=quarantine_table))

        mismatch_df = self._mismatch_writer.write(spark, source_df, target_df, key_columns, self._mismatch_table)
        meta_df = self._meta_writer.write(spark, source_df, target_df, key_columns, self._meta_table)
        dups_df = self._dups_writer.write(spark, source_df, target_df, key_columns, self._dups_table)
        count_df = self._count_writer.write(
            spark, source_df, target_df, key_columns, self._count_table, threshold=threshold
        )

        return {
            "mismatch": mismatch_df,
            "meta": meta_df,
            "dups": dups_df,
            "count": count_df,
        }
