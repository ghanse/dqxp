from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import SparkSession
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
        mismatch_table="catalog.schema.mismatch",
        meta_table="catalog.schema.meta",
        dups_table="catalog.schema.dups",
        count_table="catalog.schema.count",
        key_columns=["id"],
        threshold=0.05,
    )


class TestDQEngineExtensionInit:
    def test_constructor_stores_engine(self, extension, mock_engine):
        assert extension.engine is mock_engine

    def test_constructor_stores_key_columns(self, extension):
        assert extension.key_columns == ["id"]

    def test_key_columns_returns_copy(self, extension):
        cols = extension.key_columns
        cols.append("extra")
        assert extension.key_columns == ["id"]

    def test_constructor_stores_threshold(self, extension):
        assert extension.threshold == pytest.approx(0.05)

    def test_constructor_default_threshold(self, mock_engine):
        ext = DQEngineExtension(
            engine=mock_engine,
            mismatch_table="t1",
            meta_table="t2",
            dups_table="t3",
            count_table="t4",
            key_columns=["id"],
        )
        assert ext.threshold == pytest.approx(0.0)

    def test_constructor_rejects_empty_key_columns(self, mock_engine):
        with pytest.raises(ValueError, match="key_columns must not be empty"):
            DQEngineExtension(
                engine=mock_engine,
                mismatch_table="t1",
                meta_table="t2",
                dups_table="t3",
                count_table="t4",
                key_columns=[],
            )

    def test_constructor_rejects_negative_threshold(self, mock_engine):
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            DQEngineExtension(
                engine=mock_engine,
                mismatch_table="t1",
                meta_table="t2",
                dups_table="t3",
                count_table="t4",
                key_columns=["id"],
                threshold=-1.0,
            )


class TestWriterSchemas:
    def test_mismatch_schema_field_names(self):
        names = [f.name for f in MISMATCH_SCHEMA.fields]
        assert names == [
            "unique_key", "column_name", "source_value", "target_value",
            "mismatch_type", "table_name", "run_date", "result",
        ]

    def test_meta_schema_field_names(self):
        names = [f.name for f in META_SCHEMA.fields]
        assert names == [
            "table_name", "source_column_name", "target_column_name",
            "source_data_type", "target_data_type",
            "source_column_count", "target_column_count",
            "result", "run_date",
        ]

    def test_dups_schema_field_names(self):
        names = [f.name for f in DUPS_SCHEMA.fields]
        assert names == [
            "unique_key", "table_name", "dataset",
            "duplicate_count", "run_date", "result",
        ]

    def test_count_schema_field_names(self):
        names = [f.name for f in COUNT_SCHEMA.fields]
        assert names == [
            "table_name", "source_count", "target_count",
            "count_difference", "threshold", "result", "run_date",
        ]


class TestWriterStubs:
    def test_mismatch_writer_returns_empty_df(self, spark):
        from dqxp.writers.mismatch import MismatchWriter
        writer = MismatchWriter()
        source = spark.createDataFrame([], StructType([]))
        target = spark.createDataFrame([], StructType([]))
        result = writer.write(spark, source, target, ["id"], "test_table")
        assert result.count() == 0
        assert result.schema == MISMATCH_SCHEMA

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

    def test_count_writer_returns_empty_df(self, spark):
        from dqxp.writers.count import CountWriter
        writer = CountWriter()
        source = spark.createDataFrame([], StructType([]))
        target = spark.createDataFrame([], StructType([]))
        result = writer.write(spark, source, target, ["id"], "test_table", threshold=0.1)
        assert result.count() == 0
        assert result.schema == COUNT_SCHEMA


class TestApplyChecksAndSave:
    def test_calls_engine_and_returns_all_tables(self, spark, extension, mock_engine):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        source = spark.createDataFrame([(1, "alice"), (2, "bob")], schema)
        target = spark.createDataFrame([(1, "alice"), (3, "carol")], schema)
        good = spark.createDataFrame([(1, "alice")], schema)
        bad = spark.createDataFrame([(2, "bob")], schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        result = extension.apply_checks_and_save(source, target, checks=[])

        mock_engine.apply_checks_and_split.assert_called_once_with(source, [])
        assert set(result.keys()) == {"mismatch", "meta", "dups", "count"}
        for df in result.values():
            assert df.count() == 0
