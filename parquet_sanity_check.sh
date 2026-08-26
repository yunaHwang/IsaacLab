#!/usr/bin/env bash
#
# parquet_sanity_check.sh
#
# Usage:
#   ./parquet_sanity_check.sh ./lerobot_dataset_*
#
# Runs a series of sanity checks on every Parquet file in a LeRobot dataset:
#   1. file(1) identifies it as Apache Parquet
#   2. reports file size
#   3. checks the PAR1 header
#   4. checks the PAR1 footer
#   5. asks PyArrow to read the schema
#   6. asks PyArrow to read every Parquet file in the dataset
#

set -u

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <lerobot_dataset_directory>"
    echo
    echo "Example:"
    echo "  $0 ./lerobot_dataset_0810/ID-visuomotor-based"
    exit 1
fi

ROOT="$1"

if [ ! -d "$ROOT" ]; then
    echo "ERROR: Directory does not exist:"
    echo "  $ROOT"
    exit 1
fi

echo "============================================================"
echo "LeRobot Parquet Sanity Check"
echo "============================================================"
echo "Dataset:"
echo "  $ROOT"
echo

# ------------------------------------------------------------
# Find Parquet files
# ------------------------------------------------------------

mapfile -t FILES < <(find "$ROOT" -type f -name "*.parquet" | sort)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "ERROR: No .parquet files found."
    exit 1
fi

echo "Found ${#FILES[@]} Parquet file(s)."
echo

TOTAL=0
FAIL=0

# ------------------------------------------------------------
# Checks 1-4 for every Parquet file
# ------------------------------------------------------------

for P in "${FILES[@]}"; do
    TOTAL=$((TOTAL + 1))

    echo "------------------------------------------------------------"
    echo "[$TOTAL/${#FILES[@]}] $P"
    echo "------------------------------------------------------------"

    # 1. file(1)
    FILE_RESULT=$(file "$P")
    echo "[1] file:"
    echo "    $FILE_RESULT"

    if [[ "$FILE_RESULT" != *"Parquet"* ]]; then
        echo "    ❌ FAIL: file does not identify this as Parquet."
        FAIL=$((FAIL + 1))
    else
        echo "    ✅ PASS"
    fi

    # 2. file size
    echo "[2] size:"
    ls -lh "$P"

    # 3. PAR1 header
    HEADER=$(head -c 4 "$P" | xxd -p)
    echo "[3] header:"
    echo "    $HEADER"

    if [ "$HEADER" = "50415231" ]; then
        echo "    ✅ PASS: starts with PAR1"
    else
        echo "    ❌ FAIL: missing PAR1 header"
        FAIL=$((FAIL + 1))
    fi

    # 4. PAR1 footer
    FOOTER=$(tail -c 4 "$P" | xxd -p)
    echo "[4] footer:"
    echo "    $FOOTER"

    if [ "$FOOTER" = "50415231" ]; then
        echo "    ✅ PASS: ends with PAR1"
    else
        echo "    ❌ FAIL: missing PAR1 footer"
        FAIL=$((FAIL + 1))
    fi

    echo
done

# ------------------------------------------------------------
# Check 5: PyArrow schema read
# ------------------------------------------------------------

echo "============================================================"
echo "[5] PyArrow schema checks"
echo "============================================================"

PYARROW_FAILED=0

for P in "${FILES[@]}"; do
    echo
    echo "Checking:"
    echo "  $P"

    if python - "$P" <<'PY'
import sys
import pyarrow.parquet as pq

path = sys.argv[1]

try:
    schema = pq.read_schema(path)
    print(schema)
    print("PARQUET OK")
except Exception as e:
    print(f"PARQUET FAILED: {e}")
    sys.exit(1)
PY
    then
        echo "  ✅ PASS"
    else
        echo "  ❌ FAIL"
        PYARROW_FAILED=1
    fi
done

if [ "$PYARROW_FAILED" -ne 0 ]; then
    FAIL=$((FAIL + 1))
fi

# ------------------------------------------------------------
# Check 6: PyArrow read of ALL files
# ------------------------------------------------------------

echo
echo "============================================================"
echo "[6] Full PyArrow check of ALL Parquet files"
echo "============================================================"

if python - "$ROOT" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
files = sorted(root.rglob("*.parquet"))

print(f"Found {len(files)} parquet files")

failed = []

for path in files:
    try:
        table = pq.read_table(path)
        print(f"OK: {path}  rows={table.num_rows}, columns={table.num_columns}")
    except Exception as e:
        print(f"FAILED: {path}")
        print(f"  {e}")
        failed.append(path)

print()

if failed:
    print(f"FAILED FILES: {len(failed)}")
    sys.exit(1)

print("ALL PARQUET FILES OK")
PY
then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    FAIL=$((FAIL + 1))
fi

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Dataset:       $ROOT"
echo "Parquet files: $TOTAL"

if [ "$FAIL" -eq 0 ]; then
    echo
    echo "✅ ALL CHECKS PASSED"
    echo
    echo "The Parquet files:"
    echo "  - are recognized as Parquet"
    echo "  - have a PAR1 header"
    echo "  - have a PAR1 footer"
    echo "  - can be read by PyArrow"
    echo "  - can be fully loaded by PyArrow"
    exit 0
else
    echo
    echo "❌ $FAIL CHECK(S) FAILED"
    echo
    echo "The dataset should NOT be considered fully sane yet."
    exit 1
fi
