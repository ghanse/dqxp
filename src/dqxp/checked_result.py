from dataclasses import dataclass
from pyspark.sql import DataFrame


@dataclass(slots=True)
class CheckedResult:
    """
    Contains DQX check results and the context needed to write data quality metadata.

    The ``good_df`` and ``bad_df`` fields are available for non-DQXP consumers (e.g., quarantine workflows).

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
