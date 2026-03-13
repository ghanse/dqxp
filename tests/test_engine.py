from unittest.mock import MagicMock

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from dqxp.engine import CheckedResults, DQEngineExtension
from dqxp.writers.count import COUNT_SCHEMA
from dqxp.writers.dups import DUPS_SCHEMA
from dqxp.writers.meta import META_SCHEMA
from dqxp.writers.mismatch import MISMATCH_SCHEMA


@pytest.fixture()
def mock_engine(spark):
    """Create a mock DQEngine with a real SparkSession."""
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


@pytest.fixture()
def id_name_schema():
    """Reusable two-column schema used by most engine tests."""
    return StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ]
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

    def test_output_table_saves_good_records(self, spark, extension, mock_engine, id_name_schema):
        source = spark.createDataFrame([(1, "alice"), (2, "bob")], id_name_schema)
        target = spark.createDataFrame([(1, "alice")], id_name_schema)
        good = spark.createDataFrame([(1, "alice")], id_name_schema)
        bad = spark.createDataFrame([(2, "bob")], id_name_schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        result = extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
            output_table="catalog.schema.output",
        )

        mock_engine.save_results_in_table.assert_called_once()
        call_kwargs = mock_engine.save_results_in_table.call_args[1]
        assert call_kwargs["output_config"].location == "catalog.schema.output"
        assert call_kwargs["output_df"] is not None
        assert call_kwargs["quarantine_df"] is None
        assert set(result.keys()) == {"mismatch", "meta", "dups", "count"}

    def test_quarantine_table_saves_bad_records(self, spark, extension, mock_engine, id_name_schema):
        source = spark.createDataFrame([(1, "alice")], id_name_schema)
        target = spark.createDataFrame([(1, "alice")], id_name_schema)
        good = spark.createDataFrame([], id_name_schema)
        bad = spark.createDataFrame([(1, "alice")], id_name_schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        result = extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
            quarantine_table="catalog.schema.quarantine",
        )

        mock_engine.save_results_in_table.assert_called_once()
        call_kwargs = mock_engine.save_results_in_table.call_args[1]
        assert call_kwargs["quarantine_config"].location == "catalog.schema.quarantine"
        assert call_kwargs["quarantine_df"] is not None
        assert call_kwargs["output_df"] is None
        assert set(result.keys()) == {"mismatch", "meta", "dups", "count"}

    def test_both_output_and_quarantine_tables(self, spark, extension, mock_engine, id_name_schema):
        source = spark.createDataFrame([(1, "alice"), (2, "bob")], id_name_schema)
        target = spark.createDataFrame([(1, "alice")], id_name_schema)
        good = spark.createDataFrame([(1, "alice")], id_name_schema)
        bad = spark.createDataFrame([(2, "bob")], id_name_schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
            output_table="catalog.schema.output",
            quarantine_table="catalog.schema.quarantine",
        )

        mock_engine.save_results_in_table.assert_called_once()
        call_kwargs = mock_engine.save_results_in_table.call_args[1]
        assert call_kwargs["output_config"].location == "catalog.schema.output"
        assert call_kwargs["quarantine_config"].location == "catalog.schema.quarantine"
        assert call_kwargs["output_df"] is not None
        assert call_kwargs["quarantine_df"] is not None

    def test_neither_output_nor_quarantine_skips_save(self, spark, extension, mock_engine, id_name_schema):
        source = spark.createDataFrame([(1, "alice")], id_name_schema)
        target = spark.createDataFrame([(1, "alice")], id_name_schema)
        good = spark.createDataFrame([(1, "alice")], id_name_schema)
        bad = spark.createDataFrame([], id_name_schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        result = extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
        )

        mock_engine.save_results_in_table.assert_not_called()
        assert set(result.keys()) == {"mismatch", "meta", "dups", "count"}

    def test_quarantine_passes_df_when_table_provided(self, spark, extension, mock_engine, id_name_schema):
        source = spark.createDataFrame([(1, "alice")], id_name_schema)
        target = spark.createDataFrame([(1, "alice")], id_name_schema)
        good = spark.createDataFrame([(1, "alice")], id_name_schema)
        bad = spark.createDataFrame([], id_name_schema)

        mock_engine.apply_checks_and_split.return_value = (good, bad)

        extension.apply_checks_and_save_output_tables(
            source,
            target,
            checks=[],
            key_columns=["id"],
            quarantine_table="catalog.schema.quarantine",
        )

        mock_engine.save_results_in_table.assert_called_once()
        call_kwargs = mock_engine.save_results_in_table.call_args[1]
        assert call_kwargs["quarantine_df"] is not None
        assert call_kwargs["quarantine_config"].location == "catalog.schema.quarantine"
        assert call_kwargs["output_df"] is None

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


# ---------------------------------------------------------------------------
# Helper for new method tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_schema_3col():
    return StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("age", IntegerType()),
        ]
    )


def _setup_engine_split(mock_engine, source_df, schema):
    """Configure mock engine to return good/bad split (all good, none bad)."""
    spark = mock_engine.spark
    good_df = source_df
    bad_df = spark.createDataFrame([], schema)
    mock_engine.apply_checks_and_split.return_value = (good_df, bad_df)


# ---------------------------------------------------------------------------
# Tests for get_checked_results
# ---------------------------------------------------------------------------


class TestGetCheckedResults:
    """Tests for get_checked_results method."""

    def test_returns_checked_results_type(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema_3col)
        target = spark.createDataFrame([(1, "alice", 31)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.get_checked_results(source, target, [], ["id"])

        assert isinstance(result, CheckedResults)
        assert result.source_df is source
        assert result.target_df is target
        assert result.key_columns == ["id"]
        assert result.threshold == 0.0

    def test_passes_checks_to_engine(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        checks = [MagicMock()]

        extension.get_checked_results(source, target, checks, ["id"])

        mock_engine.apply_checks_and_split.assert_called_once_with(source, checks)

    def test_passes_ref_dfs(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        ref_dfs = {"ref": target}

        extension.get_checked_results(source, target, [], ["id"], ref_dfs=ref_dfs)

        mock_engine.apply_checks_and_split.assert_called_once_with(source, [], ref_dfs=ref_dfs)

    def test_good_and_bad_dfs(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.get_checked_results(source, target, [], ["id"])

        assert result.good_df.count() == 2
        assert result.bad_df.count() == 0

    def test_custom_threshold_stored(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.get_checked_results(source, target, [], ["id"], threshold=0.05)

        assert result.threshold == 0.05

    def test_empty_key_columns_raises(self, spark, extension, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)

        with pytest.raises(ValueError, match="key_columns must not be empty"):
            extension.get_checked_results(source, target, [], [])

    def test_negative_threshold_raises(self, spark, extension, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)

        with pytest.raises(ValueError, match="threshold must be non-negative"):
            extension.get_checked_results(source, target, [], ["id"], threshold=-1.0)


# ---------------------------------------------------------------------------
# Tests for save_to_mismatch_table
# ---------------------------------------------------------------------------


class TestSaveToMismatchTable:
    """Tests for save_to_mismatch_table method."""

    def test_with_checked_results(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "alice", 30), (2, "bob", 25)], simple_schema_3col)
        target = spark.createDataFrame([(1, "alice", 31), (3, "carol", 35)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_mismatch_table(checked_results=cr)

        assert isinstance(result, DataFrame)
        rows = result.collect()
        assert len(rows) > 0
        types = {r["mismatch_type"] for r in rows}
        assert "VALUE_MISMATCH" in types
        assert "TARGET_MISSING" in types
        assert "SOURCE_MISSING" in types

    def test_without_checked_results(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.save_to_mismatch_table(source_df=source, target_df=target, checks=[], key_columns=["id"])

        assert isinstance(result, DataFrame)
        mock_engine.apply_checks_and_split.assert_called_once()

    def test_uses_configured_table_name(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "b", 20)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_mismatch_table(checked_results=cr)

        rows = result.collect()
        for r in rows:
            assert r["table_name"] == "catalog.schema.mismatch"


# ---------------------------------------------------------------------------
# Tests for save_to_schema_validation_table
# ---------------------------------------------------------------------------


class TestSaveToSchemaValidationTable:
    """Tests for save_to_schema_validation_table method."""

    def test_with_checked_results(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_schema_validation_table(checked_results=cr)

        assert isinstance(result, DataFrame)
        rows = result.collect()
        assert len(rows) == 3  # id, name, age

    def test_without_checked_results(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.save_to_schema_validation_table(
            source_df=source, target_df=target, checks=[], key_columns=["id"]
        )

        assert isinstance(result, DataFrame)
        mock_engine.apply_checks_and_split.assert_called_once()

    def test_matching_schemas_pass(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(2, "b", 20)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_schema_validation_table(checked_results=cr)

        for r in result.collect():
            assert r["result"] == "PASS"

    def test_uses_configured_table_name(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(2, "b", 20)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_schema_validation_table(checked_results=cr)

        for r in result.collect():
            assert r["table_name"] == "catalog.schema.meta"


# ---------------------------------------------------------------------------
# Tests for save_to_duplicates_table
# ---------------------------------------------------------------------------


class TestSaveToDuplicatesTable:
    """Tests for save_to_duplicates_table method."""

    def test_no_duplicates(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(3, "c", 30)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_duplicates_table(checked_results=cr)

        assert isinstance(result, DataFrame)
        assert result.count() == 0

    def test_with_duplicates(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(2, "c", 30)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_duplicates_table(checked_results=cr)

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["dataset"] == "SOURCE"
        assert rows[0]["duplicate_count"] == 2

    def test_without_checked_results(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.save_to_duplicates_table(source_df=source, target_df=target, checks=[], key_columns=["id"])

        assert isinstance(result, DataFrame)
        mock_engine.apply_checks_and_split.assert_called_once()

    def test_uses_configured_table_name(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(2, "c", 30)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_duplicates_table(checked_results=cr)

        for r in result.collect():
            assert r["table_name"] == "catalog.schema.dups"


# ---------------------------------------------------------------------------
# Tests for save_to_row_count_table
# ---------------------------------------------------------------------------


class TestSaveToRowCountTable:
    """Tests for save_to_row_count_table method."""

    def test_matching_counts(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(3, "c", 30), (4, "d", 40)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_row_count_table(checked_results=cr)

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["source_count"] == 2
        assert rows[0]["target_count"] == 2
        assert rows[0]["result"] == "PASS"

    def test_count_difference_fails(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(3, "c", 30)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_row_count_table(checked_results=cr)

        rows = result.collect()
        assert rows[0]["source_count"] == 2
        assert rows[0]["target_count"] == 1
        assert rows[0]["count_difference"] == 1
        assert rows[0]["result"] == "FAIL"

    def test_threshold_passes_through(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(3, "c", 30)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"], threshold=0.5)

        result = extension.save_to_row_count_table(checked_results=cr)

        rows = result.collect()
        assert rows[0]["threshold"] == 0.5
        assert rows[0]["result"] == "PASS"

    def test_without_checked_results(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        result = extension.save_to_row_count_table(source_df=source, target_df=target, checks=[], key_columns=["id"])

        assert isinstance(result, DataFrame)
        mock_engine.apply_checks_and_split.assert_called_once()

    def test_uses_configured_table_name(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        target = spark.createDataFrame([(2, "b", 20)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)
        cr = extension.get_checked_results(source, target, [], ["id"])

        result = extension.save_to_row_count_table(checked_results=cr)

        for r in result.collect():
            assert r["table_name"] == "catalog.schema.count"


# ---------------------------------------------------------------------------
# Tests for checked_results reuse
# ---------------------------------------------------------------------------


class TestCheckedResultsReuse:
    """Test that checked_results can be computed once and reused."""

    def test_reuse_across_all_save_methods(self, spark, extension, mock_engine, simple_schema_3col):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema_3col)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema_3col)
        _setup_engine_split(mock_engine, source, simple_schema_3col)

        cr = extension.get_checked_results(source, target, [], ["id"])

        mismatch = extension.save_to_mismatch_table(checked_results=cr)
        meta = extension.save_to_schema_validation_table(checked_results=cr)
        dups = extension.save_to_duplicates_table(checked_results=cr)
        count = extension.save_to_row_count_table(checked_results=cr)

        # Engine should only have been called once
        assert mock_engine.apply_checks_and_split.call_count == 1

        assert isinstance(mismatch, DataFrame)
        assert isinstance(meta, DataFrame)
        assert isinstance(dups, DataFrame)
        assert isinstance(count, DataFrame)
