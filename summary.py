import os
import time
import logging
import pyspark
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# from pyspark.sql.functions import coalesce, lit, col, concat_ws, when, udf, round
from pyspark.sql.types import LongType, StringType

# Initialize logging
logging.basicConfig(level=logging.INFO)


# Function to create a Spark session
def create_spark_session():
    try:
        start_time = time.time()
        spark = SparkSession.builder \
            .appName("Expedia Lodging Summary and Guest-Reviews") \
            .master("local[*]") \
            .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.local.type", "hadoop") \
            .config("spark.sql.catalog.local.warehouse", "summary-warehouse") \
            .getOrCreate()
        end_time = time.time()
        logging.info(f"Spark session created successfully in {end_time - start_time:.2f} seconds")
        return spark
    except Exception as e:
        logging.error("Error creating Spark Session", exc_info=True)
        return None

# Function to read data from JSON files
def read_data(spark, file_path):
    try:
        start_time = time.time()
        df = spark.read.json(file_path)
        end_time = time.time()
        logging.info(f"Data from {file_path} read successfully in {end_time - start_time:.2f} seconds")
        return df
    except Exception as e:
        logging.error(f"Error reading data from {file_path}", exc_info=True)
        return None


def generate_state_mappings():
    try:
        
        with open("input/state_name_abbr.json", "r") as usa_state_name, open("input/state_name_abbr_ca.json", "r") as canada_state_name:
            state_mapping_usa = json.load(usa_state_name)
            state_mapping_canada = json.load(canada_state_name)

        flat_usa_mapping = {}
        for abbr, names in state_mapping_usa.items():
            if isinstance(names, list):
                for name in names:
                    flat_usa_mapping[name] = abbr
            else:
                flat_usa_mapping[names] = abbr

        # Convert dictionaries to lists of tuples for Spark
        usa_map_entries = list(flat_usa_mapping.items())
        canada_map_entries = list(state_mapping_canada.items())

        return usa_map_entries, canada_map_entries

    except Exception as e:
        print(f"An error occurred while generating state mappings: {e}")
        return None, None

# 
# Function to transform summary_df
def transform_summary_df(summary_df, usa_map_entries, canada_map_entries):
    try:
        start_time = time.time()
        transformed_df = summary_df.select(
            F.coalesce(F.col("propertyId.expedia"), F.lit("")).alias("expedia_id"),
            F.coalesce(F.col("address1"), F.lit("")).alias("address"),
            F.coalesce(F.col("propertyName"), F.lit("")).alias("property_name"),
            F.coalesce(F.col("city"), F.lit("")).alias("city"),
            F.coalesce(F.col("province"), F.lit("")).alias("state"),
            F.coalesce(F.col("postalCode"), F.lit("")).alias("zip"),
            F.coalesce(F.col("country"), F.lit("")).alias("country"),
            F.when(
                (F.col("geoLocation.latitude").isNull()) & (F.col("geoLocation.longitude").isNull() & F.col("geoLocation").isNull()), F.lit("")  # Handle null lat/lon
            ).otherwise(
                F.concat_ws(",", F.col("geoLocation.latitude"), F.col("geoLocation.longitude"))  # Concatenate lat/lon
            ).alias("latlon"),
            F.coalesce(
                F.when(
                    (F.col("city").isNotNull()) & (F.col("state").isNotNull()) & (F.col("country").isNotNull()),
                    F.concat_ws(", ", F.col("city"), F.col("state"), F.col("country"))
                ).when(
                    (F.col("city").isNotNull()) & (F.col("state").isNotNull()),
                    F.concat_ws(", ", F.col("city"), F.col("state"))
                ).when(
                    (F.col("city").isNotNull()) & (F.col("country").isNotNull()),
                    F.concat_ws(", ", F.col("city"), F.col("country"))
                ).when(
                    (F.col("state").isNotNull()) & (F.col("country").isNotNull()),
                    F.concat_ws(", ", F.col("state"), F.col("country"))
                ).when(
                    F.col("city").isNotNull(), F.col("city")
                ).when(
                    F.col("state").isNotNull(), F.col("state")
                ).when(
                    F.col("country").isNotNull(), F.col("country")
                ).otherwise(F.lit("")), F.lit("")
            ).alias("display"),
            F.coalesce(
                F.when(
                    F.col("country").isin("USA", "United States", "United States of America", "US"),
                    F.element_at(F.create_map(*[F.lit(x) for x in sum(usa_map_entries, ())]), F.col("state"))
                ).when(
                    F.col("country").isin("Canada", "CAN"),
                    F.element_at(F.create_map(*[F.lit(x) for x in sum(canada_map_entries, ())]), F.col("state"))
                ),
                F.lit("")
            ).alias("state_abbr")
        )
        end_time = time.time()
        logging.info(f"Summary DataFrame transformed successfully in {end_time - start_time:.2f} seconds")
        return transformed_df
    except Exception as e:
        logging.error("Error transforming summary DataFrame", exc_info=True)
        return None

# Function to transform guest_review_df
def transform_guest_review_df(guest_review_df):
    try:
        start_time = time.time()
        transformed_df = guest_review_df.select(
            F.coalesce(F.col("propertyId.expedia"), F.lit("")).alias("expedia_Id"),
            F.coalesce(F.col("guestRating.expedia.avgRating"), F.lit("")).alias("review_score"),
            F.coalesce((F.round(F.coalesce(F.col("guestRating.expedia.avgRating").cast(LongType()), F.lit(0)) / 2, 1)).cast(StringType()), F.lit("")).alias("review_score_general"),
            F.coalesce(F.col("guestRating.expedia.reviewCount").cast(LongType()), F.lit(0)).alias("number_of_review")
        )
        end_time = time.time()
        logging.info(f"Guest Review DataFrame transformed successfully in {end_time - start_time:.2f} seconds")
        return transformed_df
    except Exception as e:
        logging.error("Error transforming guest review DataFrame", exc_info=True)
        return None

# Function to join transformed DataFrames
def join_dataframes(summary_df, guest_review_df):
    try:
        start_time = time.time()
        joined_df = summary_df.join(guest_review_df, "expedia_id", "inner")
        end_time = time.time()
        logging.info(f"DataFrames joined successfully in {end_time - start_time:.2f} seconds")
        return joined_df
    except Exception as e:
        logging.error("Error joining DataFrames", exc_info=True)
        return None

def write_to_iceberg(spark, df):
    try:
        start_time = time.time()

        spark.sql("""
            CREATE TABLE IF NOT EXISTS  local.default.expedia_summary(
                expedia_id STRING,
                address STRING,
                property_name STRING,
                city STRING,
                state STRING,
                zip STRING,
                country STRING,
                latlon STRING,
                display STRING,
                state_abbr STRING,
                review_score STRING,
                review_score_general STRING,
                number_Of_review LONG
            ) USING iceberg
            OPTIONS ('format-version'='2')
            """
        )
        df.writeTo("local.default.expedia_summary").overwrite(F.lit(True))
        end_time = time.time()
        logging.info(f"DataFrame written to Iceberg table local.default.expedia_summary successfully in {end_time - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error writing DataFrame to Iceberg table local.default.expedia_summary", exc_info=True)

# Main function to execute the whole process
def main():
    start_time = time.time()
    spark = create_spark_session()
    if spark:
        guest_review_df = read_data(spark, "input/expedia-lodging-guestreviews-1-all.jsonl")
        summary_df = read_data(spark, "input/expedia-lodging-summary-en_us-1-all.jsonl")
        if guest_review_df and summary_df:
            usa_map_entries, canada_map_entries = generate_state_mappings()
            transformed_summary_df = transform_summary_df(summary_df, usa_map_entries, canada_map_entries)
            transformed_guest_review_df = transform_guest_review_df(guest_review_df)
            if transformed_summary_df and transformed_guest_review_df:
                joined_df = join_dataframes(transformed_summary_df, transformed_guest_review_df)
                if joined_df:
                    joined_df.show(5, truncate=False)
                    write_to_iceberg(spark, joined_df)

    spark.sql("SELECT * FROM local.default.expedia_summary").show(5, truncate = False)

    #result = spark.sql("SELECT COUNT(*) AS total_rows FROM local.default.expedia_summary").collect()
    #total_rows = result[0]['total_rows']

    # Print the total number of rows
    #logging.info(f"Total rows: {total_rows}")

    end_time = time.time()
    logging.info(f"The script executed in {end_time - start_time:.2f} seconds")

    spark.stop()


if __name__ == "__main__":
    main()