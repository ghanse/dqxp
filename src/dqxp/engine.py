from __future__ import annotations

import logging

from databricks.labs.dqx.engine import DQEngineCore
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
        mismatch_table: Fully-qualified name of the mismatch output table.
        meta_table: Fully-qualified name of the meta/schema output table.
        dups_table: Fully-qualified name of the duplicates output table.
        count_table: Fully-qualified name of the count comparison output table.
        key_columns: Column names that together form the unique key for records.
        threshold: Acceptable count difference threshold (0.0 = exact match).
    """

    def __init__(
        self,
        engine: DQEngineCore,
        mismatch_table: str,
        meta_table: str,
        dups_table: str,
        count_table: str,
        key_columns: list[str],
        threshold: float = 0.0,
    ):
        if not key_columns:
            raise ValueError("key_columns must not be empty")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        self._engine = engine
        self._mismatch_table = mismatch_table
        self._meta_table = meta_table
        self._dups_table = dups_table
        self._count_table = count_table
        self._key_columns = key_columns
        self._threshold = threshold

        self._mismatch_writer = MismatchWriter()
        self._meta_writer = MetaWriter()
        self._dups_writer = DupsWriter()
        self._count_writer = CountWriter()

    @property
    def engine(self) -> DQEngineCore:
        return self._engine

    @property
    def key_columns(self) -> list[str]:
        return list(self._key_columns)

    @property
    def threshold(self) -> float:
        return self._threshold

    def apply_checks_and_save(
        self,
        source_df: DataFrame,
        target_df: DataFrame,
        checks: list,
        quarantine_table: str | None = None,
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
            quarantine_table: Optional table name for quarantined records.

        Returns:
            A dict mapping table kind to the DataFrame that was written:
            ``{"mismatch": ..., "meta": ..., "dups": ..., "count": ...}``.
        """
        spark = self._engine.spark

        # Step 1: run DQX checks
        result = self._engine.apply_checks_and_split(source_df, checks)
        # result is (good_df, bad_df) or (good_df, bad_df, observation)
        good_df = result[0]
        bad_df = result[1]

        logger.info("DQX checks complete: %d good, %d bad", good_df.count(), bad_df.count())

        # Step 2: quarantine bad records (placeholder — just log for now)
        if quarantine_table and bad_df.count() > 0:
            logger.info("Would write %d bad records to %s", bad_df.count(), quarantine_table)

        # Step 3: compute and write analysis tables
        mismatch_df = self._mismatch_writer.write(
            spark, source_df, target_df, self._key_columns, self._mismatch_table
        )
        meta_df = self._meta_writer.write(
            spark, source_df, target_df, self._key_columns, self._meta_table
        )
        dups_df = self._dups_writer.write(
            spark, source_df, target_df, self._key_columns, self._dups_table
        )
        count_df = self._count_writer.write(
            spark, source_df, target_df, self._key_columns, self._count_table, threshold=self._threshold
        )

        return {
            "mismatch": mismatch_df,
            "meta": meta_df,
            "dups": dups_df,
            "count": count_df,
        }
