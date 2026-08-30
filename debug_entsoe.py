import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

KEY_PATH = Path("key.txt")
if not KEY_PATH.exists():
    print("key.txt not found")
    raise SystemExit(1)

API_KEY = KEY_PATH.read_text().strip()

ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
DOMAIN_SVERIGE = "10YSE-1--------K"

target_time = (datetime.now(timezone.utc) - timedelta(hours=5)).replace(
    minute=0, second=0, microsecond=0
)
start_str = target_time.strftime("%Y%m%d%H00")
end_str = (target_time + timedelta(hours=1)).strftime("%Y%m%d%H00")
params = {
    "securityToken": API_KEY,
    "documentType": "A65",
    "processType": "A16",
    "outBiddingZone_Domain": DOMAIN_SVERIGE,
    "periodStart": start_str,
    "periodEnd": end_str,
}

print("Requesting:", ENTSOE_URL)
print("Params:", params)

resp = requests.get(ENTSOE_URL, params=params, timeout=30)

print("Status code:", resp.status_code)
print("Content length:", len(resp.content))
preview = resp.content[:2000]
try:
    print(preview.decode("utf-8"))
except Exception:
    print(preview)

open("entsoe_debug.xml", "wb").write(resp.content)
print("Saved entsoe_debug.xml")
