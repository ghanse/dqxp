"""Tests for the DQCount writer."""

from __future__ import annotations

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from dqxp.writers.count import COUNT_SCHEMA, CountWriter


@pytest.fixture()
def writer():
    return CountWriter()


@pytest.fixture()
def simple_schema():
    return StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("age", IntegerType()),
        ]
    )


class TestExactMatch:
    """Test exact match scenarios (threshold=0)."""

    def test_equal_counts_pass(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema)
        target = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "test_table")

        rows = result.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["source_count"] == 2
        assert row["target_count"] == 2
        assert row["count_difference"] == 0
        assert row["result"] == "PASS"
        assert row["table_name"] == "test_table"

    def test_unequal_counts_fail_with_zero_threshold(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10), (2, "b", 20)], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=0.0)

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["result"] == "FAIL"
        assert rows[0]["count_difference"] == 1


class TestWithinThreshold:
    """Test scenarios where the difference is within threshold."""

    def test_small_difference_within_threshold(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(i, "x", i) for i in range(100)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(95)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=0.05)

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["source_count"] == 100
        assert rows[0]["target_count"] == 95
        assert rows[0]["count_difference"] == 5
        assert rows[0]["result"] == "PASS"

    def test_exactly_at_threshold_passes(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(i, "x", i) for i in range(100)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(90)], simple_schema)

        # difference/max = 10/100 = 0.10
        result = writer.write(spark, source, target, ["id"], "t", threshold=0.10)

        rows = result.collect()
        assert rows[0]["result"] == "PASS"

    def test_target_larger_within_threshold(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(i, "x", i) for i in range(90)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(100)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=0.10)

        rows = result.collect()
        assert rows[0]["source_count"] == 90
        assert rows[0]["target_count"] == 100
        assert rows[0]["count_difference"] == 10
        assert rows[0]["result"] == "PASS"


class TestOverThreshold:
    """Test scenarios where the difference exceeds threshold."""

    def test_over_threshold_fails(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(i, "x", i) for i in range(100)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(80)], simple_schema)

        # difference/max = 20/100 = 0.20 > 0.10
        result = writer.write(spark, source, target, ["id"], "t", threshold=0.10)

        rows = result.collect()
        assert rows[0]["result"] == "FAIL"
        assert rows[0]["count_difference"] == 20

    def test_just_over_threshold_fails(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(i, "x", i) for i in range(100)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(89)], simple_schema)

        # difference/max = 11/100 = 0.11 > 0.10
        result = writer.write(spark, source, target, ["id"], "t", threshold=0.10)

        rows = result.collect()
        assert rows[0]["result"] == "FAIL"


class TestEmptyTables:
    """Test edge cases with empty DataFrames."""

    def test_both_empty_pass(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["source_count"] == 0
        assert rows[0]["target_count"] == 0
        assert rows[0]["count_difference"] == 0
        assert rows[0]["result"] == "PASS"

    def test_source_empty_target_not_fail(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert rows[0]["result"] == "FAIL"
        assert rows[0]["source_count"] == 0
        assert rows[0]["target_count"] == 1

    def test_target_empty_source_not_fail(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert rows[0]["result"] == "FAIL"
        assert rows[0]["source_count"] == 1
        assert rows[0]["target_count"] == 0

    def test_source_empty_target_not_fail_even_with_threshold(self, spark, writer, simple_schema):
        source = spark.createDataFrame([], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=1.0)

        rows = result.collect()
        assert rows[0]["result"] == "FAIL"


class TestThresholdValues:
    """Test various threshold configurations."""

    def test_threshold_one_always_passes_nonzero(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(100)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=1.0)

        rows = result.collect()
        assert rows[0]["result"] == "PASS"

    def test_threshold_stored_in_output(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=0.05)

        rows = result.collect()
        assert rows[0]["threshold"] == pytest.approx(0.05)

    def test_default_threshold_is_zero(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert rows[0]["threshold"] == pytest.approx(0.0)


class TestSchemaConformance:
    """Verify the output DataFrame conforms to COUNT_SCHEMA."""

    def test_output_schema_field_names_and_types(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        expected_fields = [(f.name, f.dataType) for f in COUNT_SCHEMA.fields]
        actual_fields = [(f.name, f.dataType) for f in result.schema.fields]
        assert actual_fields == expected_fields

    def test_produces_exactly_one_row(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(i, "x", i) for i in range(50)], simple_schema)
        target = spark.createDataFrame([(i, "x", i) for i in range(30)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t", threshold=0.5)

        assert result.count() == 1

    def test_run_date_populated(self, spark, writer, simple_schema):
        source = spark.createDataFrame([(1, "a", 10)], simple_schema)
        target = spark.createDataFrame([(1, "a", 10)], simple_schema)

        result = writer.write(spark, source, target, ["id"], "t")

        rows = result.collect()
        assert rows[0]["run_date"] is not None
