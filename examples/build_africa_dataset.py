"""
Fetches real African health indicators from World Bank API + OWID malaria data.
Saves combined CSV to data/africa_health.csv
"""
import json, urllib.request, time, csv, io, os, subprocess
import pandas as pd

COUNTRIES = "GH;KE;NG;ZA;ET;TZ;UG;CM;SN;ML;BF;GN;CI;MZ;ZM;ZW;RW;TG;BJ;NE;MW;LS;SL;LR;GM"

INDICATORS = {
    "SH.STA.MMRT":    "maternal_mortality_per100k",
    "SP.DYN.IMRT.IN": "infant_mortality_per1000",
    "SH.TBS.INCD":    "tb_incidence_per100k",
    "SH.HIV.INCD.ZS": "hiv_incidence_per1000",
    "SP.DYN.LE00.IN": "life_expectancy",
}

def fetch_wb(indicator):
    url = (f"https://api.worldbank.org/v2/country/{COUNTRIES}"
           f"/indicator/{indicator}?format=json&per_page=2000&date=2000:2022")
    result = subprocess.run(['curl', '-s', '--max-time', '30', url],
                            capture_output=True, text=True)
    return json.loads(result.stdout.lstrip('﻿'))[1]

def fetch_owid_malaria():
    url = ("https://ourworldindata.org/grapher/incidence-of-malaria.csv"
           "?v=1&csvType=full&useColumnShortNames=false")
    result = subprocess.run(['curl', '-s', '--max-time', '30', url],
                            capture_output=True, text=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))

print("Fetching World Bank health indicators...")
rows = {}
for ind, col in INDICATORS.items():
    print(f"  {col}...")
    for item in fetch_wb(ind):
        if item['value'] is None:
            continue
        key = (item['country']['value'], item['countryiso3code'], item['date'])
        rows.setdefault(key, {})[col] = item['value']
    time.sleep(0.3)

print("Fetching OWID malaria incidence...")
for row in fetch_owid_malaria():
    if not row.get('Code') or len(row['Code']) != 3:
        continue
    for key in rows:
        if key[1] == row['Code'] and key[2] == row['Year']:
            rows[key]['malaria_incidence_per1000'] = float(
                row['Incidence of malaria (per 1,000 population at risk)'])

records = []
for (country, iso3, year), vals in rows.items():
    if len(vals) >= 3:
        records.append({'country': country, 'iso3': iso3, 'year': int(year), **vals})

df = pd.DataFrame(records).sort_values(['country', 'year']).reset_index(drop=True)

out = '/Users/macos/Downloads/projects/neurodata/data/africa_health.csv'
df.to_csv(out, index=False)

print(f"\nSaved: {out}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print(f"Countries: {df['country'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")
print(df.head(10).to_string())
