"""Tests for the DQMeta writer."""

from __future__ import annotations

import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from dqxp.writers.meta import META_SCHEMA, MetaWriter


@pytest.fixture()
def writer():
    return MetaWriter()


@pytest.fixture()
def simple_schema():
    return StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("age", IntegerType()),
        ]
    )


class TestMatchingSchemas:
    """Test that identical schemas produce all PASS results."""

    def test_all_pass_when_schemas_match(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(2, "bob", 25)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "test_table")

        rows = result.collect()
        assert len(rows) == 3
        for r in rows:
            assert r["result"] == "PASS"
            assert r["table_name"] == "test_table"
            assert r["source_column_name"] == r["target_column_name"]
            assert r["source_data_type"] == r["target_data_type"]

    def test_run_date_populated(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(2, "bob", 25)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        for r in rows:
            assert r["run_date"] is not None


class TestTypeMismatch:
    """Test FAIL when data types differ between source and target."""

    def test_type_mismatch_fails(self, spark, writer):
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("value", IntegerType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("value", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, 100)], source_schema)
        target = spark.createDataFrame([(1, "100")], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = {r["source_column_name"]: r for r in result.collect()}
        assert rows["id"]["result"] == "PASS"
        assert rows["value"]["result"] == "FAIL"
        assert rows["value"]["source_data_type"] == "int"
        assert rows["value"]["target_data_type"] == "string"

    def test_multiple_type_mismatches(self, spark, writer):
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("a", IntegerType()),
                StructField("b", StringType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("a", LongType()),
                StructField("b", DoubleType()),
            ]
        )
        source = spark.createDataFrame([(1, 1, "x")], source_schema)
        target = spark.createDataFrame([(1, 1, 2.0)], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = {r["source_column_name"]: r for r in result.collect()}
        assert rows["id"]["result"] == "PASS"
        assert rows["a"]["result"] == "FAIL"
        assert rows["b"]["result"] == "FAIL"


class TestColumnInSourceNotTarget:
    """Test columns present in source but missing from target."""

    def test_source_only_column_fails(self, spark, writer):
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
                StructField("extra", IntegerType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "a", 10)], source_schema)
        target = spark.createDataFrame([(1, "a")], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        extra_row = [r for r in rows if r["source_column_name"] == "extra"]
        assert len(extra_row) == 1
        r = extra_row[0]
        assert r["result"] == "FAIL"
        assert r["target_column_name"] is None
        assert r["target_data_type"] is None
        assert r["source_data_type"] == "int"


class TestColumnInTargetNotSource:
    """Test columns present in target but missing from source."""

    def test_target_only_column_fails(self, spark, writer):
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
                StructField("extra", IntegerType()),
            ]
        )
        source = spark.createDataFrame([(1, "a")], source_schema)
        target = spark.createDataFrame([(1, "a", 10)], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        extra_row = [r for r in rows if r["target_column_name"] == "extra"]
        assert len(extra_row) == 1
        r = extra_row[0]
        assert r["result"] == "FAIL"
        assert r["source_column_name"] is None
        assert r["source_data_type"] is None
        assert r["target_data_type"] == "int"


class TestColumnCounts:
    """Test that column counts are populated correctly on every row."""

    def test_same_column_counts(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(2, "b", 20)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        for r in rows:
            assert r["source_column_count"] == 3
            assert r["target_column_count"] == 3

    def test_different_column_counts(self, spark, writer):
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("a", StringType()),
                StructField("b", StringType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("a", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "x", "y")], source_schema)
        target = spark.createDataFrame([(1, "x")], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        for r in rows:
            assert r["source_column_count"] == 3
            assert r["target_column_count"] == 2

    def test_different_column_counts_cause_fail_even_with_type_match(self, spark, writer):
        """When column counts differ, even matching columns should FAIL."""
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
                StructField("extra", StringType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "a", "e")], source_schema)
        target = spark.createDataFrame([(1, "a")], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        # All rows should FAIL because column counts differ
        for r in rows:
            assert r["result"] == "FAIL"


class TestEmptyDataFrames:
    """Test handling of empty DataFrames."""

    def test_both_empty_same_schema(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 3
        for r in rows:
            assert r["result"] == "PASS"

    def test_both_empty_different_schemas(self, spark, writer):
        source_schema = StructType([StructField("id", IntegerType())])
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("extra", StringType()),
            ]
        )
        source = spark.createDataFrame([], source_schema)
        target = spark.createDataFrame([], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 2
        results = {(r["target_column_name"] or r["source_column_name"]): r for r in rows}
        # id is in both but counts differ -> FAIL
        assert results["id"]["result"] == "FAIL"
        # extra only in target -> FAIL
        assert results["extra"]["result"] == "FAIL"


class TestSchemaConformance:
    """Verify the output DataFrame conforms to META_SCHEMA."""

    def test_output_schema_field_names_and_types(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "alice", 30)], simple_schema)
        target = spark.createDataFrame([(2, "bob", 25)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        expected_fields = [(f.name, f.dataType) for f in META_SCHEMA.fields]
        actual_fields = [(f.name, f.dataType) for f in result.schema.fields]
        assert actual_fields == expected_fields

    def test_empty_result_schema_matches(self, spark, writer):
        source_schema = StructType([])
        target_schema = StructType([])
        source = spark.createDataFrame([], source_schema)
        target = spark.createDataFrame([], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        assert result.count() == 0
        expected_fields = [(f.name, f.dataType) for f in META_SCHEMA.fields]
        actual_fields = [(f.name, f.dataType) for f in result.schema.fields]
        assert actual_fields == expected_fields


class TestOneRowPerColumn:
    """Verify the writer produces exactly one row per column."""

    def test_row_count_equals_union_of_columns(self, spark, writer):
        source_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("a", StringType()),
                StructField("b", StringType()),
            ]
        )
        target_schema = StructType(
            [
                StructField("id", IntegerType()),
                StructField("b", StringType()),
                StructField("c", StringType()),
            ]
        )
        source = spark.createDataFrame([(1, "x", "y")], source_schema)
        target = spark.createDataFrame([(1, "y", "z")], target_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        # Union of columns: id, a, b, c = 4
        assert result.count() == 4
