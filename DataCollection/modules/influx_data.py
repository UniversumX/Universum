from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client import WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import signal
import sys
import asyncio
from datetime import datetime
import csv
from pathlib import Path


# Set the environment variables
bucket = "<my-bucket>"
org = "<my-org>"
token = "<my-token>"
# Store the URL of your InfluxDB instance
url="https://us-west-2-1.aws.cloud2.influxdata.com"

# Initialize InfluxDB client
client = InfluxDBClient(
   url=url,
   token=token,
   org=org
)

write_api = client.write_api(write_options=SYNCHRONOUS)

p = Point("my_measurement").tag("location", "Prague").field("temperature", 25.3)
write_api.write(bucket=bucket, org=org, record=p) 

# Write data to InfluxDB
def write_data_to_influx(label, data):
    point = Point(label)
    for key, value in data.items():
        point = point.tag("sensor", key).field("value", value)
    write_api.write(bucket=bucket, org=org, record=point)
    
# Query data from InfluxDB
query_api = client.query_api()

query = 'from(bucket:"my-bucket")\
|> range(start: -10m)\
|> filter(fn:(r) => r._measurement == "my_measurement")\
|> filter(fn:(r) => r.tag == "sensor")\
|> filter(fn:(r) => r._field == "value")'

result = query_api.query(org=org, query=query)
results = []
for table in result:
    for record in table.records:
        results.append((record.get_field(), record.get_value()))

print(results)
