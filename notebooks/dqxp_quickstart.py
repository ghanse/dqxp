# Databricks notebook source
# MAGIC %md
# MAGIC # DQXP Quickstart
# MAGIC
# MAGIC **DQXP** extends [Databricks Labs DQX](https://databrickslabs.github.io/dqx/) with four additional
# MAGIC data quality analysis tables that compare a **source** dataset against a **target**:
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | **Mismatch** | Row-level value differences between source and target |
# MAGIC | **Schema Validation** | Column name, type, and count comparison |
# MAGIC | **Duplicates** | Duplicate key detection in both datasets |
# MAGIC | **Row Count** | Row count comparison with configurable threshold |
# MAGIC
# MAGIC This notebook walks through installation, sample data generation, and all key APIs.

# COMMAND ----------

# MAGIC %pip install dqxp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from dqxp.__about__ import __version__, __dqx_version__
print(__version__)
print(__dqx_version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Set the catalog and schema where DQXP will write its analysis tables.

# COMMAND ----------

CATALOG = "main"
SCHEMA = "dqxp_demo"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Sample Data
# MAGIC
# MAGIC We create a **source** (the "truth") and a **target** that has intentional discrepancies:
# MAGIC a modified value, a missing row, an extra row, and a duplicate key.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("region", StringType(), True),
    StructField("balance", DoubleType(), True),
])

source_data = [
    (1, "Alice", "US", 1000.0),
    (2, "Bob", "EU", 2500.0),
    (3, "Carol", "US", 750.0),
    (4, "Dave", "APAC", 3200.0),
    (5, "Eve", "EU", 1800.0),
]

target_data = [
    (1, "Alice", "US", 1000.0),       # matches source
    (2, "Bob", "EU", 2499.0),         # balance mismatch
    (3, "Carol", "UK", 750.0),        # region mismatch
    # customer_id=4 missing             target missing
    (5, "Eve", "EU", 1800.0),         # matches source
    (6, "Frank", "US", 500.0),        # source missing
    (5, "Eve", "EU", 1800.0),         # duplicate key in target
]

source_df = spark.createDataFrame(source_data, schema)
target_df = spark.createDataFrame(target_data, schema)

print("Source:")
source_df.show()
print("Target:")
target_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define DQX Checks
# MAGIC
# MAGIC DQX rules validate the **source** data. Records that fail are quarantined;
# MAGIC passing records flow into the output table. DQXP then compares source vs target
# MAGIC for the four analysis tables.

# COMMAND ----------

from databricks.labs.dqx.rule import DQRowRule
from databricks.labs.dqx.check_funcs import is_not_null, is_not_empty, is_in_list

checks = [
    DQRowRule(check_func=is_not_null, column="customer_id", name="customer_id_not_null"),
    DQRowRule(check_func=is_not_null, column="name", name="name_not_null"),
    DQRowRule(check_func=is_not_empty, column="name", name="name_not_empty"),
    DQRowRule(
        check_func=is_in_list,
        column="region",
        name="region_valid",
        check_func_kwargs={"allowed": ["US", "EU", "APAC"]},
    ),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run All Checks at Once
# MAGIC
# MAGIC `apply_checks_and_save_output_tables` is the main entry point. It runs DQX checks,
# MAGIC optionally saves output/quarantine tables, and writes all four analysis tables.

# COMMAND ----------

from databricks.labs.dqx.engine import DQEngine
from databricks.sdk import WorkspaceClient
from dqxp import DQEngineExtension

engine = DQEngine(WorkspaceClient())

ext = DQEngineExtension(
    engine=engine,
    mismatch_table_name=f"{TABLE_PREFIX}.dq_mismatch",
    schema_validation_table_name=f"{TABLE_PREFIX}.dq_schema_validation",
    duplicate_count_table_name=f"{TABLE_PREFIX}.dq_duplicates",
    row_count_table_name=f"{TABLE_PREFIX}.dq_row_count",
)

results = ext.apply_checks_and_save_output_tables(
    source_df=source_df,
    target_df=target_df,
    checks=checks,
    key_columns=["customer_id"],
    output_table=f"{TABLE_PREFIX}.validated_output",
    quarantine_table=f"{TABLE_PREFIX}.quarantine",
    threshold=0.05,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Mismatch Table
# MAGIC
# MAGIC Shows row-level value differences, plus rows present in only one dataset.

# COMMAND ----------

results["mismatch"].show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Schema Validation Table
# MAGIC
# MAGIC Compares column names, data types, and column counts between source and target.

# COMMAND ----------

results["meta"].show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c. Duplicates Table
# MAGIC
# MAGIC Detects duplicate keys in both source and target datasets.

# COMMAND ----------

results["dups"].show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4d. Row Count Table
# MAGIC
# MAGIC Compares total row counts with threshold tolerance (5% in this example).

# COMMAND ----------

results["count"].show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Granular API
# MAGIC
# MAGIC For more control, use `get_checked_result` to run checks once, then call
# MAGIC individual `save_to_*` methods. This avoids re-running checks when you only
# MAGIC need a subset of the analysis tables.

# COMMAND ----------

checked = ext.get_checked_result(
    source_df=source_df,
    target_df=target_df,
    checks=checks,
    key_columns=["customer_id"],
    threshold=0.05,
)

print(f"Good records: {checked.good_df.count()}")
print(f"Bad records:  {checked.bad_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save individual tables using the cached result

# COMMAND ----------

mismatch_df = ext.save_to_mismatch_table(checked_result=checked)
mismatch_df.show(truncate=False)

# COMMAND ----------

count_df = ext.save_to_row_count_table(checked_result=checked)
count_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cleanup

# COMMAND ----------

spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | API | Use Case |
# MAGIC |-----|----------|
# MAGIC | `apply_checks_and_save_output_tables()` | Run checks + write all 4 analysis tables in one call |
# MAGIC | `get_checked_result()` | Run checks without writing; returns reusable `CheckedResult` |
# MAGIC | `save_to_mismatch_table()` | Write only the mismatch analysis table |
# MAGIC | `save_to_schema_validation_table()` | Write only the schema validation table |
# MAGIC | `save_to_duplicates_table()` | Write only the duplicates table |
# MAGIC | `save_to_row_count_table()` | Write only the row count table |
# MAGIC
# MAGIC For more details, see the [DQXP GitHub repo](https://github.com/ghanse/dqxp).