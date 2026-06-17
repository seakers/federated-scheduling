import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# PASTE YOUR PERSONAL ACCESS TOKEN HERE
# =====================================================================

CDS_API_KEY = os.environ.get("CDS_API_KEY")

# Define the exact path the library is screaming for
cds_api_path = Path(os.environ.get("USERPROFILE")) / ".cdsapirc"

# Write the formal credentials file structure required by ECMWF
config_content = f"""url: https://cds.climate.copernicus.eu/api
key: {CDS_API_KEY}
"""

with open(cds_api_path, "w", encoding="utf-8") as f:
    f.write(config_content)

print(f"✅ Bulletproof config written successfully to: {cds_api_path}")