from pathlib import  Path
import json

path = Path("./earthquakes.geojson")

if path.exists:
    info = json.loads(path.read_text())
    print(info)
else:
    print(f"Not working. {path}")

