from unittest.mock import MagicMock

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from dqxp.engine import DQEngineExtension
from dqxp.writers.count import COUNT_SCHEMA
from dqxp.writers.dups import DUPS_SCHEMA
from dqxp.writers.meta import META_SCHEMA
from dqxp.writers.mismatch import MISMATCH_SCHEMA


@pytest.fixture()
def mock_engine(spark):
    """Create a mock DQEngineCore with a real SparkSession."""
    engine = MagicMock()
    engine.spark = spark
    return engine


@pytest.fixture()
def extension(mock_engine):
    """Create a DQEngineExtension with default test parameters."""
    return DQEngineExtension(
        engine=mock_engine,
        mismatch_table_name="catalog.schema.mismatch",
        schema_validation_table_name="catalog.schema.meta",
        duplicate_count_table_name="catalog.schema.dups",
        row_count_table_name="catalog.schema.count",
    )


class TestDQEngineExtensionInit:
    def test_constructor_stores_engine(self, extension, mock_engine):
        assert extension.engine is mock_engine


class TestWriterSchemas:
    def test_mismatch_schema_field_names(self):
        names = [f.name for f in MISMATCH_SCHEMA.fields]
        assert names == [
            "unique_key",
            "column_name",
            "source_value",
            "target_value",
            "mismatch_type",
            "table_name",
            "run_date",
            "result",
        ]

    def test_meta_schema_field_names(self):
        names = [f.name for f in META_SCHEMA.fields]
        assert names == [
            "table_name",
            "source_column_name",
            "target_column_name",
            "source_data_type",
            "target_data_type",
            "source_column_count",
            "target_column_count",
            "result",
            "run_date",
        ]

    def test_dups_schema_field_names(self):
        names = [f.name for f in DUPS_SCHEMA.fields]
        assert names == [
            "unique_key",
            "table_name",
            "dataset",
            "duplicate_count",
            "run_date",
            "result",
        ]

    def test_count_schema_field_names(self):
        names = [f.name for f in COUNT_SCHEMA.fields]
        assert names == [
            "table_name",
            "source_count",
            "target_count",
            "count_difference",
            "threshold",
            "result",
            "run_date",
        ]


class TestWriterEmptyInputs:
    def test_mismatch_writer_returns_empty_for_empty_inputs(self, spark):
        from dqxp.writers.mismatch import MismatchWriter

        schema = StructType([StructField("id", IntegerType())])
        writer = MismatchWriter()
        source = spark.createDataFrame([], schema)
        target = spark.createDataFrame([], schema)
        result = writer.write(spark, source, target, ["id"], "test_table")
        assert result.count() == 0

    def test_meta_writer_returns_empty_df(self, spark):
        from dqxp.writers.meta import MetaWriter

        writer = MetaWriter()
        source = spark.createDataFrame([], StructType([]))
        target = spark.createDataFrame([], StructType([]))
        result = writer.write(spark, source, target, ["id"], "test_table")
        assert result.count() == 0
        assert result.schema == META_SCHEMA

    def test_dups_writer_returns_empty_df(self, spark):
        from dqxp.writers.dups import DupsWriter

        writer = DupsWriter()
        source = spark.createDataFrame([], StructType([]))
        target = spark.createDataFrame([], StructType([]))
        result = writer.write(spark, source, target, ["id"], "test_table")
        assert result.count() == 0
        assert result.schema == DUPS_SCHEMA

    def test_count_writer_returns_one_row(self, spark):
        from dqxp.writers.count import CountWriter

        writer = CountWriter()
        source = spark.createDataFrame([], StructType([]))
        target = spark.createDataFrame([], StructType([]))
        result = writer.write(spark, source, target, ["id"], "test_table", threshold=0.1)
        assert result.count() == 1
        assert result.schema == COUNT_SCHEMA
        row = result.collect()[0]
        assert row["result"] == "PASS"


class TestApplyChecksAndSaveOutputTables:
    def test_calls_engine_and_returns_all_tables(self, spark, extension, mock_engine):
        schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "alice"), (2, "bob")], schema)
        target = spark.createDataFrame([(1, "alice"), (3, "carol")], schema)
        good = spark.createDataFrame([(1, "alice")], schema)
        bad = spark.createDataFrame([(2, "bob")], schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        result = extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
        )

        mock_engine.apply_checks_and_split.assert_called_once_with(source, [])
        assert set(result.keys()) == {"mismatch", "meta", "dups", "count"}
        # mismatch now returns real data (id=2 TARGET_MISSING, id=3 SOURCE_MISSING)
        assert result["mismatch"].count() > 0
        # meta is now implemented (returns 1 row per column)
        assert result["meta"].count() > 0
        # dups writer returns no duplicates for unique keys
        assert result["dups"].count() == 0
        # count writer now produces exactly 1 row
        assert result["count"].count() == 1

    def test_rejects_empty_key_columns(self, spark, extension, mock_engine):
        schema = StructType([StructField("id", IntegerType())])
        source = spark.createDataFrame([], schema)
        target = spark.createDataFrame([], schema)
        with pytest.raises(ValueError, match="key_columns must not be empty"):
            extension.apply_checks_and_save_output_tables(
                source,
                target,
                checks=[],
                key_columns=[],
            )

    def test_rejects_negative_threshold(self, spark, extension, mock_engine):
        schema = StructType([StructField("id", IntegerType())])
        source = spark.createDataFrame([], schema)
        target = spark.createDataFrame([], schema)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            extension.apply_checks_and_save_output_tables(
                source,
                target,
                checks=[],
                key_columns=["id"],
                threshold=-1.0,
            )

    def test_quarantine_raises_not_implemented(self, spark, extension, mock_engine):
        schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "alice")], schema)
        target = spark.createDataFrame([(1, "alice")], schema)
        good = spark.createDataFrame([], schema)
        bad = spark.createDataFrame([(1, "alice")], schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            extension.apply_checks_and_save_output_tables(
                source,
                target,
                checks=[],
                key_columns=["id"],
                quarantine_table="catalog.schema.quarantine",
            )

    def test_passes_ref_dfs_to_engine(self, spark, extension, mock_engine):
        schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "alice")], schema)
        target = spark.createDataFrame([(1, "alice")], schema)
        good = spark.createDataFrame([(1, "alice")], schema)
        bad = spark.createDataFrame([], schema)
        ref = spark.createDataFrame([(1, "ref")], schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
            ref_dfs={"reference": ref},
        )

        mock_engine.apply_checks_and_split.assert_called_once_with(
            source,
            [],
            ref_dfs={"reference": ref},
        )
