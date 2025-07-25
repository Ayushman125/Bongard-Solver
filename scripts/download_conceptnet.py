import conceptnet_lite
import os

db_path = "data/conceptnet_build/conceptnet.db"

if os.path.exists(db_path):
    print("ConceptNet database already exists. Skipping download.")
else:
    print("Downloading ConceptNet database...")
    conceptnet_lite.connect(
       db_path
    )
    print("Download complete.")
