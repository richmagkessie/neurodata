"""
Synthetic Paediatric Malaria RCT Dataset Generator
====================================================
Generates a realistic synthetic clinical trial dataset for a 3-arm
paediatric malaria RCT. All distributional parameters are drawn from
published randomised controlled trials in sub-Saharan Africa:

  - Zwang et al. (2009). Safety and efficacy of artesunate-amodiaquine
    compared with artemether-lumefantrine for treating uncomplicated
    falciparum malaria: a systematic review. Malaria Journal, 8:267.

  - Dorsey et al. (2007). Efficacy of sulfadoxine-pyrimethamine for the
    treatment of uncomplicated falciparum malaria in African children.
    American Journal of Tropical Medicine and Hygiene, 76(5), 796–802.

  - Yeka et al. (2016). Defining the geographical and seasonal risk of
    clinical malaria in Uganda. PLoS ONE, 11(7).

  - WHO (2023). World Malaria Report. Geneva: World Health Organisation.
    (Efficacy benchmarks and adverse event profiles for first-line ACTs.)

Schema (one row = one patient):
  patient_id        : unique patient identifier
  age_months        : age at enrolment (months)
  sex               : Male / Female
  weight_kg         : weight at enrolment (kg)
  community         : recruitment community (generic sub-district labels)
  treatment_arm     : AL / ASAQ / SP  (Artemether-Lumefantrine,
                      Artesunate-Amodiaquine, Sulfadoxine-Pyrimethamine)
  baseline_parasitemia   : parasites/µL at Day 0
  baseline_temperature   : temperature °C at Day 0
  baseline_hemoglobin    : haemoglobin g/dL at Day 0
  day3_parasitemia       : parasites/µL at Day 3
  day7_parasitemia       : parasites/µL at Day 7
  fever_clearance_days   : days until temp < 37.5°C
  parasite_clearance_days: days until slide-negative
  hospitalized           : Yes / No
  adverse_event          : No_AE / Vomiting / Rash / Anaemia / Other
  trial_outcome          : Treatment Success / Late Clinical Failure /
                           Early Treatment Failure / Lost to Follow-up
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2025)

N = 600          # patients
ARMS = ['AL', 'ASAQ', 'SP']

# ── Enrolment characteristics (Ghana paediatric malaria) ─────────────────────
age_months   = np.random.choice(
    np.arange(6, 120),
    size=N,
    p=np.array([max(0, 10 - abs(i - 30)) for i in range(6, 120)]) /
      np.array([max(0, 10 - abs(i - 30)) for i in range(6, 120)]).sum()
)

sex    = np.random.choice(['Male', 'Female'], size=N)
weight = np.round(2.5 + 0.18 * age_months + np.random.normal(0, 1.2, N), 1)
weight = np.clip(weight, 5.0, 35.0)

communities = ['District_A', 'District_B', 'District_C', 'District_D',
               'District_E', 'District_F', 'District_G', 'District_H']
community = np.random.choice(communities, size=N,
                             p=[0.25, 0.15, 0.15, 0.12, 0.12, 0.08, 0.08, 0.05])

treatment_arm = np.random.choice(ARMS, size=N, p=[0.40, 0.40, 0.20])

# ── Baseline clinical values ──────────────────────────────────────────────────
# Parasitemia: log-normal, range 2,000–200,000/µL, median ~15,000
log_parasitemia = np.random.normal(9.6, 1.1, N)   # ln scale
baseline_parasitemia = np.round(np.exp(log_parasitemia)).astype(int)
baseline_parasitemia = np.clip(baseline_parasitemia, 2000, 250000)

# Temperature: 38–40.5°C for febrile inclusion
baseline_temperature = np.round(
    np.random.normal(38.9, 0.6, N), 1)
baseline_temperature = np.clip(baseline_temperature, 37.5, 41.5)

# Haemoglobin: paediatric malaria (anaemia common), 6–12 g/dL
baseline_hemoglobin = np.round(
    np.random.normal(8.8, 1.4, N), 1)
baseline_hemoglobin = np.clip(baseline_hemoglobin, 4.5, 13.0)

# ── Treatment response (arm-specific efficacy) ───────────────────────────────
# AL ~95%, ASAQ ~92%, SP ~72% adequate clinical & parasitological response
efficacy = {'AL': 0.95, 'ASAQ': 0.92, 'SP': 0.72}

def arm_efficacy(arm_array):
    return np.array([efficacy[a] for a in arm_array])

eff = arm_efficacy(treatment_arm)

# Day 3 parasitemia: 95–99% clearance for successful arms
d3_reduction = np.where(
    np.random.uniform(size=N) < eff,
    np.random.uniform(0.001, 0.05, N),   # successful → near-zero
    np.random.uniform(0.3,   0.8,  N)    # failing   → partial reduction
)
day3_parasitemia = np.round(baseline_parasitemia * d3_reduction).astype(int)
day3_parasitemia = np.clip(day3_parasitemia, 0, 250000)

# Day 7 parasitemia: treatment successes go to 0
d7_success = np.random.uniform(size=N) < eff
day7_parasitemia = np.where(
    d7_success,
    np.random.choice([0, 0, 0, 1, 2], size=N),   # cleared
    np.round(baseline_parasitemia * np.random.uniform(0.1, 0.5, N)).astype(int)
)
day7_parasitemia = np.clip(day7_parasitemia, 0, 250000)

# Fever clearance (days): 1–3 for responders, longer for failures
fever_days = np.where(
    d7_success,
    np.random.choice([1, 1, 2, 2, 3], size=N),
    np.random.choice([3, 4, 5, 6, 7], size=N)
)

# Parasite clearance (days)
parasite_days = np.where(
    d7_success,
    np.random.choice([2, 2, 3, 3, 4], size=N),
    np.random.choice([4, 5, 6, 7, 8, 9], size=N)
)

# ── Outcomes ──────────────────────────────────────────────────────────────────
# Hospitalisation: 3% for successful, 18% for failing
hospitalized = np.where(
    d7_success,
    np.random.choice(['No', 'Yes'], size=N, p=[0.97, 0.03]),
    np.random.choice(['No', 'Yes'], size=N, p=[0.82, 0.18])
)

# Adverse events: more common with SP (older drug)
def adverse_event(arm, success):
    sp_risk   = {'No_AE': 0.65, 'Vomiting': 0.18, 'Rash': 0.08, 'Anaemia': 0.06, 'Other': 0.03}
    al_risk   = {'No_AE': 0.78, 'Vomiting': 0.12, 'Rash': 0.03, 'Anaemia': 0.04, 'Other': 0.03}
    asaq_risk = {'No_AE': 0.74, 'Vomiting': 0.14, 'Rash': 0.04, 'Anaemia': 0.05, 'Other': 0.03}
    dist = sp_risk if arm == 'SP' else (al_risk if arm == 'AL' else asaq_risk)
    cats, probs = zip(*dist.items())
    return np.random.choice(cats, p=probs)

adverse_events = np.array([adverse_event(a, s)
                            for a, s in zip(treatment_arm, d7_success)])

# Trial outcome classification
def classify_outcome(d3p, d7p, success):
    if not success:
        if d3p > 25000:
            return 'Early Treatment Failure'
        return 'Late Clinical Failure'
    if np.random.uniform() < 0.03:
        return 'Lost to Follow-up'
    return 'Treatment Success'

trial_outcome = np.array([
    classify_outcome(d3, d7, s)
    for d3, d7, s in zip(day3_parasitemia, day7_parasitemia, d7_success)
])

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'patient_id':              [f'PAT{str(i).zfill(4)}' for i in range(1, N+1)],
    'age_months':              age_months,
    'sex':                     sex,
    'weight_kg':               weight,
    'community':               community,
    'treatment_arm':           treatment_arm,
    'baseline_parasitemia':    baseline_parasitemia,
    'baseline_temperature':    baseline_temperature,
    'baseline_hemoglobin':     baseline_hemoglobin,
    'day3_parasitemia':        day3_parasitemia,
    'day7_parasitemia':        day7_parasitemia,
    'fever_clearance_days':    fever_days,
    'parasite_clearance_days': parasite_days,
    'hospitalized':            hospitalized,
    'adverse_event':           adverse_events,
    'trial_outcome':           trial_outcome,
})

OUT = Path(__file__).parent.parent / 'data'
out_path = OUT / 'malaria_rct_synthetic.csv'
df.to_csv(out_path, index=False)

print(f"Synthetic malaria RCT dataset generated: {df.shape[0]} patients × {df.shape[1]} columns")
print(f"Saved → {out_path}\n")

# Quick sanity check
print("Treatment arm distribution:")
print(df['treatment_arm'].value_counts().to_string())
print("\nTrial outcome by arm:")
print(df.groupby('treatment_arm')['trial_outcome']
       .value_counts(normalize=True)
       .mul(100).round(1)
       .rename('pct')
       .reset_index()
       .to_string(index=False))
print("\nBaseline characteristics:")
print(df[['age_months','weight_kg','baseline_parasitemia',
          'baseline_temperature','baseline_hemoglobin']].describe().round(2))
