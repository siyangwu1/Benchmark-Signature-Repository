# read the result parquet file and print the first 10 rows

import pyarrow.parquet as pq

table = pq.read_table("output/results.parquet")