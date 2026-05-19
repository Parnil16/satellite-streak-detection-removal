import os
from pathlib import Path

# =========================
# 🔥 CONFIG
# =========================

base_dir = Path("dataset")
folders = ["images", "masks", "clean_images"]

faulty_numbers = [
    19,53,55,69,71,77,82,114,140,164,184,199,212,216,218,219,268,293,313,314,
    329,338,339,350,366,406,449,465,489,511,536,542,551,568,580,596,602,634,
    654,662,665,668,674,684,722,734,735,747,749,763,806,809,822,826,837,841,
    868,875,877,880,885,890,895,896,901,902,903,904,917,928,950,972,986,989,
    1010,1017,1027,1040,1042,1052,1061,1065,1082,1100,1114,1135,1150,1168,1188,
    1224,1243,1247,1271,1273,1292,1298,1300,1303,1316,1321,1322,1334,1360,1388,
    1402,1422,1446,1458,1473,1488,1497,1505,1559,1567,1574,1590,1608,1624,1660,
    1665,1666,1684,1694,1695,1700,1711,1718,1725,1751,1766,1770,1781,1812,1831,
    1841,1842,1857,1858,1868,1869,1884,1890,1891,1894,1934,1966,1972,1979,2000,
    2006,2019,4132,4158,4189,4351,4371,4423,4448,4475,4477,4505,4528,4588,4662,
    4664,4718,4725,4794,4793,4807,4819,4872
]

faulty_files = {f"{num:05d}.png" for num in faulty_numbers}

# =========================
# 🧹 STEP 1: DELETE FAULTY
# =========================

print("🔴 Deleting faulty files...\n")

deleted = 0

for fname in faulty_files:
    for folder in folders:
        path = base_dir / folder / fname
        if path.exists():
            path.unlink()
            deleted += 1

print(f"✅ Deleted {deleted} files (across all folders)\n")

# =========================
# 🔄 STEP 2: REINDEX DATASET
# =========================

print("🔄 Reindexing dataset...\n")

# get remaining files from images folder (source of truth)
image_files = sorted((base_dir / "images").glob("*.png"))

# temporary rename to avoid overwrite conflicts
temp_mapping = {}

for i, old_path in enumerate(image_files):
    temp_name = f"temp_{i:05d}.png"
    temp_mapping[old_path.name] = temp_name

    for folder in folders:
        old_file = base_dir / folder / old_path.name
        if old_file.exists():
            old_file.rename(base_dir / folder / temp_name)

# final rename to correct indices
for new_idx, temp_name in enumerate(sorted(temp_mapping.values())):
    new_name = f"{new_idx:05d}.png"

    for folder in folders:
        temp_file = base_dir / folder / temp_name
        if temp_file.exists():
            temp_file.rename(base_dir / folder / new_name)

print("✅ Reindexing complete!\n")

# =========================
# 📊 FINAL STATUS
# =========================

final_count = len(list((base_dir / "images").glob("*.png")))

print("══════════════════════════════")
print("🎉 DATASET CLEANED & FIXED")
print("══════════════════════════════")
print(f"Total samples now: {final_count}")
print("Folders synced:")
for folder in folders:
    count = len(list((base_dir / folder).glob("*.png")))
    print(f"  {folder}: {count}")
print("══════════════════════════════")