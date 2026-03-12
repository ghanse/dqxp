from __future__ import annotations

import logging
import warnings

from databricks.labs.dqx.engine import DQEngineCore
from databricks.labs.dqx.rule import DQRule
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
        quarantine_table: str | None = None,
        ref_dfs: dict[str, DataFrame] | None = None,
        threshold: float = 0.0,
    ) -> dict[str, DataFrame]:
        """Run DQX checks on source_df and produce all analysis tables.

        Steps:
            1. Run DQX checks via ``engine.apply_checks_and_split`` on *source_df*.
            2. Optionally persist quarantined (bad) records.
            3. Compute and write mismatch, meta, dups, and count analysis tables.

        Args:
            source_df: The source DataFrame to validate.
            target_df: The target DataFrame to compare against.
            checks: List of DQRule checks to apply via DQX.
            key_columns: Column names that together form the unique key for records.
            quarantine_table: Optional table name for quarantined records.
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

        # Step 1: run DQX checks
        extra_kwargs: dict = {}
        if ref_dfs is not None:
            extra_kwargs["ref_dfs"] = ref_dfs
        result = self._engine.apply_checks_and_split(source_df, checks, **extra_kwargs)
        # result is (good_df, bad_df) or (good_df, bad_df, observation)
        good_df = result[0]
        bad_df = result[1]

        bad_count = bad_df.count()
        logger.info("DQX checks complete: %d good, %d bad", good_df.count(), bad_count)

        # Step 2: quarantine bad records
        if quarantine_table and bad_count > 0:
            # TODO: implement quarantine persistence (write bad_df to quarantine_table)
            warnings.warn(
                f"Quarantine table '{quarantine_table}' specified but quarantine persistence "
                f"is not yet implemented. {bad_count} bad records will not be saved.",
                stacklevel=2,
            )
            raise NotImplementedError(
                f"Quarantine persistence is not yet implemented. "
                f"{bad_count} bad records would be written to '{quarantine_table}'."
            )

        # Step 3: compute and write analysis tables
        mismatch_df = self._mismatch_writer.write(
            spark, source_df, target_df, key_columns, self._mismatch_table
        )
        meta_df = self._meta_writer.write(
            spark, source_df, target_df, key_columns, self._meta_table
        )
        dups_df = self._dups_writer.write(
            spark, source_df, target_df, key_columns, self._dups_table
        )
        count_df = self._count_writer.write(
            spark, source_df, target_df, key_columns, self._count_table, threshold=threshold
        )

        return {
            "mismatch": mismatch_df,
            "meta": meta_df,
            "dups": dups_df,
            "count": count_df,
        }
