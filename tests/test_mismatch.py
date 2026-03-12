"""Tests for the DQMismatch writer."""
from __future__ import annotations

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from dqxp.writers.mismatch import MISMATCH_SCHEMA, MismatchWriter


@pytest.fixture()
def writer():
    return MismatchWriter()


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


class TestValueMismatch:
    """Test VALUE_MISMATCH detection for matched rows with differing values."""

    def test_single_key_single_column_mismatch(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(1, "alice", 31)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "test_table")

        rows = result.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["unique_key"] == "1"
        assert row["column_name"] == "age"
        assert row["source_value"] == "30"
        assert row["target_value"] == "31"
        assert row["mismatch_type"] == "VALUE_MISMATCH"
        assert row["result"] == "FAIL"
        assert row["table_name"] == "test_table"

    def test_single_key_multiple_column_mismatches(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(1, "bob", 31)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "test_table")

        rows = {r["column_name"]: r for r in result.collect()}
        assert len(rows) == 2
        assert rows["name"]["source_value"] == "alice"
        assert rows["name"]["target_value"] == "bob"
        assert rows["age"]["source_value"] == "30"
        assert rows["age"]["target_value"] == "31"
        for r in rows.values():
            assert r["mismatch_type"] == "VALUE_MISMATCH"

    def test_multiple_rows_with_mismatches(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30), (2, "bob", 25)], simple_schema)
        target = spark.createDataFrame([(1, "alice", 31), (2, "bob", 26)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2
        keys = {r["unique_key"] for r in rows}
        assert keys == {"1", "2"}
        for r in rows:
            assert r["mismatch_type"] == "VALUE_MISMATCH"
            assert r["column_name"] == "age"


class TestCompositeKeys:
    """Test composite key handling."""

    def test_composite_key_joined_with_pipe(self, spark, writer, composite_schema):
        source = spark.createDataFrame([("US", 1, 100)], composite_schema)
        target = spark.createDataFrame([("US", 1, 200)], composite_schema)

        result = writer.write(spark, source, target, ["region", "store_id"], "t")

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["unique_key"] == "US|1"
        assert rows[0]["column_name"] == "revenue"
        assert rows[0]["source_value"] == "100"
        assert rows[0]["target_value"] == "200"

    def test_composite_key_missing_row(self, spark, writer, composite_schema):
        source = spark.createDataFrame([("US", 1, 100)], composite_schema)
        target = spark.createDataFrame([("US", 2, 200)], composite_schema)

        result = writer.write(spark, source, target, ["region", "store_id"], "t")

        rows = result.collect()
        types = {r["unique_key"]: r["mismatch_type"] for r in rows}
        # US|1 only in source -> TARGET_MISSING, US|2 only in target -> SOURCE_MISSING
        assert types["US|1"] == "TARGET_MISSING"
        assert types["US|2"] == "SOURCE_MISSING"


class TestSourceMissing:
    """Test SOURCE_MISSING: rows in target but not in source."""

    def test_source_missing_records(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([(1, "alice", 30)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2  # one per non-key column
        for r in rows:
            assert r["unique_key"] == "1"
            assert r["mismatch_type"] == "SOURCE_MISSING"
            assert r["source_value"] is None
            assert r["result"] == "FAIL"

    def test_source_missing_has_target_values(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([(1, "alice", 30)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = {r["column_name"]: r for r in result.collect()}
        assert rows["name"]["target_value"] == "alice"
        assert rows["age"]["target_value"] == "30"


class TestTargetMissing:
    """Test TARGET_MISSING: rows in source but not in target."""

    def test_target_missing_records(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2  # one per non-key column
        for r in rows:
            assert r["unique_key"] == "1"
            assert r["mismatch_type"] == "TARGET_MISSING"
            assert r["target_value"] is None
            assert r["result"] == "FAIL"

    def test_target_missing_has_source_values(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = {r["column_name"]: r for r in result.collect()}
        assert rows["name"]["source_value"] == "alice"
        assert rows["age"]["source_value"] == "30"


class TestNoMismatches:
    """Test that identical DataFrames produce no mismatch records."""

    def test_identical_dataframes(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30), (2, "bob", 25)], simple_schema)
        target = spark.createDataFrame([(1, "alice", 30), (2, "bob", 25)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0

    def test_both_empty(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0


class TestMixedScenario:
    """Test combination of mismatches, source-missing, and target-missing."""

    def test_mixed_mismatch_and_missing(self, spark, writer, simple_schema):
        source = spark.createDataFrame(
            [(1, "alice", 30), (2, "bob", 25), (4, "dave", 40)],
            simple_schema,
        )
        target = spark.createDataFrame(
            [(1, "alice", 31), (3, "carol", 35), (4, "dave", 40)],
            simple_schema,
        )

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        by_key = {}
        for r in rows:
            by_key.setdefault(r["unique_key"], []).append(r)

        # id=1: VALUE_MISMATCH on age
        assert len(by_key["1"]) == 1
        assert by_key["1"][0]["mismatch_type"] == "VALUE_MISMATCH"
        assert by_key["1"][0]["column_name"] == "age"

        # id=2: TARGET_MISSING (2 non-key columns)
        assert len(by_key["2"]) == 2
        for r in by_key["2"]:
            assert r["mismatch_type"] == "TARGET_MISSING"

        # id=3: SOURCE_MISSING (2 non-key columns)
        assert len(by_key["3"]) == 2
        for r in by_key["3"]:
            assert r["mismatch_type"] == "SOURCE_MISSING"

        # id=4: no mismatches (not in result)
        assert "4" not in by_key


class TestSchemaConformance:
    """Verify the output DataFrame conforms to MISMATCH_SCHEMA."""

    def test_output_schema_field_names_and_types(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(1, "bob", 31)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        expected_fields = [(f.name, f.dataType) for f in MISMATCH_SCHEMA.fields]
        actual_fields = [(f.name, f.dataType) for f in result.schema.fields]
        assert actual_fields == expected_fields

    def test_empty_result_schema_matches(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(1, "alice", 30)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0
