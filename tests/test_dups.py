"""Tests for the DQDups writer."""

from __future__ import annotations

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from dqxp.writers.dups import DUPS_SCHEMA, DupsWriter


@pytest.fixture()
def writer():
    return DupsWriter()


@pytest.fixture()
def simple_schema():
    return StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("age", IntegerType()),
        ]
    )


@pytest.fixture()
def composite_schema():
    return StructType(
        [
            StructField("region", StringType()),
            StructField("store_id", IntegerType()),
            StructField("revenue", IntegerType()),
        ]
    )


class TestDuplicatesInSourceOnly:
    """Test detection of duplicates only in the source DataFrame."""

    def test_source_duplicates_detected(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30), (1, "alice", 31)], simple_schema)
        target = spark.createDataFrame([(2, "bob", 25)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "test_table")

        rows = result.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["unique_key"] == "1"
        assert row["dataset"] == "SOURCE"
        assert row["duplicate_count"] == 2
        assert row["result"] == "FAIL"
        assert row["table_name"] == "test_table"

    def test_source_triple_duplicate(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 1), (1, "b", 2), (1, "c", 3)], simple_schema)
        target = spark.createDataFrame([(2, "d", 4)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["duplicate_count"] == 3
        assert rows[0]["dataset"] == "SOURCE"


class TestDuplicatesInTargetOnly:
    """Test detection of duplicates only in the target DataFrame."""

    def test_target_duplicates_detected(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(2, "bob", 25), (2, "bob", 26)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["unique_key"] == "2"
        assert row["dataset"] == "TARGET"
        assert row["duplicate_count"] == 2
        assert row["result"] == "FAIL"


class TestDuplicatesInBoth:
    """Test detection of duplicates in both source and target."""

    def test_duplicates_in_both_datasets(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema)
        target = spark.createDataFrame([(2, "c", 30), (2, "d", 40)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2
        datasets = {r["dataset"]: r for r in rows}
        assert "SOURCE" in datasets
        assert "TARGET" in datasets
        assert datasets["SOURCE"]["unique_key"] == "1"
        assert datasets["TARGET"]["unique_key"] == "2"
        assert datasets["SOURCE"]["duplicate_count"] == 2
        assert datasets["TARGET"]["duplicate_count"] == 2

    def test_same_key_duplicated_in_both(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema)
        target = spark.createDataFrame([(1, "c", 30), (1, "d", 40)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2
        datasets = {r["dataset"] for r in rows}
        assert datasets == {"SOURCE", "TARGET"}
        for r in rows:
            assert r["unique_key"] == "1"
            assert r["duplicate_count"] == 2


class TestNoDuplicates:
    """Test that no duplicates produce an empty result."""

    def test_no_duplicates_empty_result(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema)
        target = spark.createDataFrame([(3, "c", 30), (4, "d", 40)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0

    def test_both_empty(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0


class TestCompositeKeys:
    """Test composite key handling with pipe-separated values."""

    def test_composite_key_joined_with_pipe(self, spark, writer, composite_schema):
        source = spark.createDataFrame([("US", 1, 100), ("US", 1, 200)], composite_schema)
        target = spark.createDataFrame([("US", 2, 300)], composite_schema)

        result = writer.write(spark, source, target, ["region", "store_id"], "t")

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["unique_key"] == "US|1"
        assert rows[0]["dataset"] == "SOURCE"
        assert rows[0]["duplicate_count"] == 2

    def test_composite_key_different_combinations(self, spark, writer, composite_schema):
        source = spark.createDataFrame([("US", 1, 100), ("US", 1, 200), ("EU", 1, 300)], composite_schema)
        target = spark.createDataFrame([("EU", 1, 400), ("EU", 1, 500)], composite_schema)

        result = writer.write(spark, source, target, ["region", "store_id"], "t")

        rows = result.collect()
        assert len(rows) == 2
        by_dataset = {r["dataset"]: r for r in rows}
        assert by_dataset["SOURCE"]["unique_key"] == "US|1"
        assert by_dataset["TARGET"]["unique_key"] == "EU|1"


class TestMultipleDuplicateKeys:
    """Test scenarios with multiple different keys having duplicates."""

    def test_multiple_duplicate_keys_in_source(self, spark, writer, simple_schema):
        source = spark.createDataFrame(
            [(1, "a", 10), (1, "b", 20), (2, "c", 30), (2, "d", 40), (3, "e", 50)],
            simple_schema,
        )
        target = spark.createDataFrame([(4, "f", 60)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2
        keys = {r["unique_key"] for r in rows}
        assert keys == {"1", "2"}
        for r in rows:
            assert r["dataset"] == "SOURCE"
            assert r["duplicate_count"] == 2

    def test_multiple_duplicate_keys_mixed(self, spark, writer, simple_schema):
        source = spark.createDataFrame(
            [(1, "a", 10), (1, "b", 20), (2, "c", 30), (2, "d", 40)],
            simple_schema,
        )
        target = spark.createDataFrame(
            [(3, "e", 50), (3, "f", 60), (4, "g", 70), (4, "h", 80)],
            simple_schema,
        )

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 4
        source_keys = {r["unique_key"] for r in rows if r["dataset"] == "SOURCE"}
        target_keys = {r["unique_key"] for r in rows if r["dataset"] == "TARGET"}
        assert source_keys == {"1", "2"}
        assert target_keys == {"3", "4"}


class TestSchemaConformance:
    """Verify the output DataFrame conforms to DUPS_SCHEMA."""

    def test_output_schema_field_names_and_types(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema)
        target = spark.createDataFrame([(2, "c", 30)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        expected_fields = [(f.name, f.dataType) for f in DUPS_SCHEMA.fields]
        actual_fields = [(f.name, f.dataType) for f in result.schema.fields]
        assert actual_fields == expected_fields

    def test_empty_result_schema_matches(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(2, "b", 20)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0
        expected_fields = [(f.name, f.dataType) for f in DUPS_SCHEMA.fields]
        actual_fields = [(f.name, f.dataType) for f in result.schema.fields]
        assert actual_fields == expected_fields

    def test_run_date_populated(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema)
        target = spark.createDataFrame([(2, "c", 30)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        for r in rows:
            assert r["run_date"] is not None

    def test_all_rows_have_result_fail(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (1, "b", 20)], simple_schema)
        target = spark.createDataFrame([(2, "c", 30), (2, "d", 40)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        for r in rows:
            assert r["result"] == "FAIL"
