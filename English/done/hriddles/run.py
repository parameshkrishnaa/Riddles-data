import pandas as pd
import sys

df = pd.read_parquet(sys.argv[1])
df.to_csv("output.csv", index=False)

