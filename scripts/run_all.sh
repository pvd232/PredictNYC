#!/usr/bin/env bash
set -euo pipefail
python parse_dem_primary_data.py
python sim_dem_primary_final.py
python parse_ed_borough_map.py
python parse_boe_2021_general.py
python parse_reg_by_ed.py
python the_city.py
cd .. && make all
echo "🎯 Done."
