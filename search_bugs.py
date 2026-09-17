import re

with open("app/measurement.py") as f:
    lines = f.readlines()

for pattern in ["_wall_key", "IS1200_OPENING_DEDUCT_M2", "external_faces", "internal_faces"]:
    for i, line in enumerate(lines, 1):
        if pattern in line:
            print(f"Line {i}: {line.rstrip()}")