from entsoe import EntsoePandasClient
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

key_path = Path("key.txt")
if not key_path.exists():
    print("key.txt missing")
    raise SystemExit(1)

api_key = key_path.read_text().strip()
client = EntsoePandasClient(api_key=api_key)

target_time = (datetime.now(timezone.utc) - timedelta(hours=5)).replace(
    minute=0, second=0, microsecond=0
)
start = pd.Timestamp(target_time).tz_convert("UTC")
end = pd.Timestamp(target_time + timedelta(hours=1)).tz_convert("UTC")

import traceback

try:
    for domain in ("10YSE-1--------K", "10Y1001A1001A46N", "SE"):
        print("Trying domain:", domain)
        try:
            ts = client.query_load(domain, start=start, end=end)
            print("  Series length:", len(ts))
            print(ts.head())
        except Exception as e:
            print("  Exception for", domain, "->", type(e).__name__)
            import traceback as _tb

            _tb.print_exc()
            continue
except Exception as e:
    print("Error from EntsoePandasClient:")
    traceback.print_exc()
