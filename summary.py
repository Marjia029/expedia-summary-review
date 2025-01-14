import os
import time
import logging
import pyspark
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import coalesce, lit, col, concat_ws, when, udf, round
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

# State abbreviation mapping dictionaries
# state_mapping_usa = {
#     "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
#     "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
#     "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
#     "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
#     "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
#     "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
#     "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
#     "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
#     "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
#     "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
#     "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": [
#         "Washington", "Washington, D.C.", "District of Columbia", "Dist. of Columbia"
#     ],
#     "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "VI": "Virgin Islands",
#     "SJ": ["St John", "Saint John", "St. John Island", "Saint John Island"],
#     "ST": ["St Thomas", "Saint Thomas", "St. Thomas Island", "Saint Thomas Island"],
#     "SX": ["St Croix", "Saint Croix", "St. Croix Island", "Saint Croix Island"]
# }

# state_mapping_canada = {
#     "NL": "Newfoundland and Labrador", "PE": "Prince Edward Island", "NS": "Nova Scotia",
#     "NB": "New Brunswick", "QC": "Quebec", "ON": "Ontario", "MB": "Manitoba",
#     "SK": "Saskatchewan", "AB": "Alberta", "BC": "British Columbia", "YT": "Yukon",
#     "NT": "Northwest Territories", "NU": "Nunavut"
# }

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
            coalesce(col("propertyId.expedia"), lit("")).alias("Expedia_ID"),
            coalesce(col("address1"), lit("")).alias("Address"),
            coalesce(col("propertyName"), lit("")).alias("Property_Name"),
            coalesce(col("city"), lit("")).alias("City"),
            coalesce(col("province"), lit("")).alias("State"),
            coalesce(col("postalCode"), lit("")).alias("Zip"),
            coalesce(col("country"), lit("")).alias("Country"),
            when(
                (col("geoLocation.latitude").isNull()) & (col("geoLocation.longitude").isNull() & col("geoLocation").isNull()), lit("")  # Handle null lat/lon
            ).otherwise(
                concat_ws(",", col("geoLocation.latitude"), col("geoLocation.longitude"))  # Concatenate lat/lon
            ).alias("LatLon"),
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
            ).alias("Display"),
            F.coalesce(
                F.when(
                    F.col("country").isin("USA", "United States", "United States of America", "US"),
                    F.element_at(F.create_map(*[F.lit(x) for x in sum(usa_map_entries, ())]), F.col("state"))
                ).when(
                    F.col("country").isin("Canada", "CAN"),
                    F.element_at(F.create_map(*[F.lit(x) for x in sum(canada_map_entries, ())]), F.col("state"))
                ),
                F.lit("")
            ).alias("State_Abbr")
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
            coalesce(col("propertyId.expedia"), lit("")).alias("Expedia_Id"),
            coalesce(col("guestRating.expedia.avgRating"), lit("")).alias("Review_Score"),
            coalesce((round(coalesce(col("guestRating.expedia.avgRating").cast(LongType()), lit(0)) / 2, 1)).cast(StringType()), lit("")).alias("Review_Score_General"),
            coalesce(col("guestRating.expedia.reviewCount").cast(LongType()), lit(0)).alias("Number_Of_Reviews")
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
        joined_df = summary_df.join(guest_review_df, "Expedia_ID", "inner")
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
                Expedia_ID STRING,
                Address STRING,
                Property_Name STRING,
                City STRING,
                State STRING,
                Zip STRING,
                Country STRING,
                LatLon STRING,
                Display STRING,
                State_Abbr STRING,
                Review_Score STRING,
                Review_Score_General STRING,
                Number_Of_Reviews LONG
            ) USING iceberg
            """
        )
        df.writeTo("local.default.expedia_summary").append()
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

    result = spark.sql("SELECT COUNT(*) AS total_rows FROM local.default.expedia_summary").collect()
    total_rows = result[0]['total_rows']

    # Print the total number of rows
    logging.info(f"Total rows: {total_rows}")

    end_time = time.time()
    logging.info(f"The script executed in {end_time - start_time:.2f} seconds")

    spark.stop()


if __name__ == "__main__":
    main()