from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

from databricks.labs.dqx.engine import DQEngineCore
from databricks.labs.dqx.rule import DQRule
from pyspark.sql import DataFrame

from dqxp.writers.count import CountWriter
from dqxp.writers.dups import DupsWriter
from dqxp.writers.meta import MetaWriter
from dqxp.writers.mismatch import MismatchWriter

logger = logging.getLogger(__name__)


@dataclass
class CheckedResults:
    """Container for DQX check results and the context needed by writers.

    The ``good_df`` and ``bad_df`` fields are available for consumer code
    (e.g., quarantine workflows) even though the built-in ``save_to_*``
    methods only use ``source_df``, ``target_df``, and ``key_columns``.

    Attributes:
        good_df: DataFrame of records that passed all DQX checks.
        bad_df: DataFrame of records that failed at least one DQX check.
        source_df: The original source DataFrame.
        target_df: The target DataFrame used for comparison.
        key_columns: Column names forming the unique key.
        threshold: Acceptable count difference threshold for the count writer.
    """

    good_df: DataFrame
    bad_df: DataFrame
    source_df: DataFrame
    target_df: DataFrame
    key_columns: list[str]
    threshold: float = 0.0


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

    def get_checked_results(
        self,
        source_df: DataFrame,
        target_df: DataFrame,
        checks: list[DQRule],
        key_columns: list[str],
        ref_dfs: dict[str, DataFrame] | None = None,
        threshold: float = 0.0,
    ) -> CheckedResults:
        """Run DQX checks and return the results without writing any tables.

        Args:
            source_df: The source DataFrame to validate.
            target_df: The target DataFrame to compare against.
            checks: List of DQRule checks to apply via DQX.
            key_columns: Column names that together form the unique key for records.
            ref_dfs: Optional dict of reference DataFrames passed through to DQX.
            threshold: Acceptable count difference threshold (0.0 = exact match).

        Returns:
            A :class:`CheckedResults` containing the good/bad split and context
            needed by the individual ``save_to_*`` methods.

        Raises:
            ValueError: If key_columns is empty or threshold is negative.
        """
        if not key_columns:
            raise ValueError("key_columns must not be empty")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        extra_kwargs: dict = {}
        if ref_dfs is not None:
            extra_kwargs["ref_dfs"] = ref_dfs
        result = self._engine.apply_checks_and_split(source_df, checks, **extra_kwargs)
        good_df = result[0]
        bad_df = result[1]

        logger.info("DQX checks complete: %d good, %d bad", good_df.count(), bad_df.count())

        return CheckedResults(
            good_df=good_df,
            bad_df=bad_df,
            source_df=source_df,
            target_df=target_df,
            key_columns=key_columns,
            threshold=threshold,
        )

    def save_to_mismatch_table(self, *, checked_results: CheckedResults | None = None, **kwargs) -> DataFrame:
        """Save mismatch analysis to the mismatch table.

        Args:
            checked_results: Pre-computed :class:`CheckedResults`. If ``None``,
                calls :meth:`get_checked_results` with the remaining keyword
                arguments (``source_df``, ``target_df``, ``checks``,
                ``key_columns``, and optionally ``ref_dfs`` / ``threshold``).
            **kwargs: Forwarded to :meth:`get_checked_results` when
                *checked_results* is not provided.

        Returns:
            The mismatch DataFrame that was produced.
        """
        cr = checked_results or self.get_checked_results(**kwargs)
        return self._mismatch_writer.write(
            self._engine.spark, cr.source_df, cr.target_df, cr.key_columns, self._mismatch_table
        )

    def save_to_schema_validation_table(self, *, checked_results: CheckedResults | None = None, **kwargs) -> DataFrame:
        """Save schema validation analysis to the meta table.

        Args:
            checked_results: Pre-computed :class:`CheckedResults`. If ``None``,
                calls :meth:`get_checked_results` with the remaining keyword
                arguments.
            **kwargs: Forwarded to :meth:`get_checked_results` when
                *checked_results* is not provided.

        Returns:
            The schema validation DataFrame that was produced.
        """
        cr = checked_results or self.get_checked_results(**kwargs)
        return self._meta_writer.write(self._engine.spark, cr.source_df, cr.target_df, cr.key_columns, self._meta_table)

    def save_to_duplicates_table(self, *, checked_results: CheckedResults | None = None, **kwargs) -> DataFrame:
        """Save duplicate detection analysis to the duplicates table.

        Args:
            checked_results: Pre-computed :class:`CheckedResults`. If ``None``,
                calls :meth:`get_checked_results` with the remaining keyword
                arguments.
            **kwargs: Forwarded to :meth:`get_checked_results` when
                *checked_results* is not provided.

        Returns:
            The duplicates DataFrame that was produced.
        """
        cr = checked_results or self.get_checked_results(**kwargs)
        return self._dups_writer.write(self._engine.spark, cr.source_df, cr.target_df, cr.key_columns, self._dups_table)

    def save_to_row_count_table(self, *, checked_results: CheckedResults | None = None, **kwargs) -> DataFrame:
        """Save row count comparison analysis to the count table.

        Args:
            checked_results: Pre-computed :class:`CheckedResults`. If ``None``,
                calls :meth:`get_checked_results` with the remaining keyword
                arguments.
            **kwargs: Forwarded to :meth:`get_checked_results` when
                *checked_results* is not provided.

        Returns:
            The row count comparison DataFrame that was produced.
        """
        cr = checked_results or self.get_checked_results(**kwargs)
        return self._count_writer.write(
            self._engine.spark, cr.source_df, cr.target_df, cr.key_columns, self._count_table, threshold=cr.threshold
        )

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
        checked_results = self.get_checked_results(
            source_df, target_df, checks, key_columns, ref_dfs=ref_dfs, threshold=threshold
        )

        bad_count = checked_results.bad_df.count()
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

        return {
            "mismatch": self.save_to_mismatch_table(checked_results=checked_results),
            "meta": self.save_to_schema_validation_table(checked_results=checked_results),
            "dups": self.save_to_duplicates_table(checked_results=checked_results),
            "count": self.save_to_row_count_table(checked_results=checked_results),
        }
