# Extract and Load All Tables
import pandas as pd
import os
import json
from zipfile import ZipFile

zip_filename = 'mimic-iii-clinical-database-demo-1.4.zip'
extract_path = os.path.join(os.path.dirname(__file__), 'mimiciii_demo')
with ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Locate actual CSV path
nested_dir = os.path.join(extract_path, 'mimic-iii-clinical-database-demo-1.4')

# Load relevant CSVs
csv_files = {
    'PATIENTS': pd.read_csv(f"{nested_dir}/PATIENTS.csv"),
    'ADMISSIONS': pd.read_csv(f"{nested_dir}/ADMISSIONS.csv"),
    'ICUSTAYS': pd.read_csv(f"{nested_dir}/ICUSTAYS.csv"),
    'CHARTEVENTS': pd.read_csv(f"{nested_dir}/CHARTEVENTS.csv", low_memory=False),
    'LABEVENTS': pd.read_csv(f"{nested_dir}/LABEVENTS.csv"),
    'INPUTEVENTS': pd.read_csv(f"{nested_dir}/INPUTEVENTS_CV.csv", low_memory=False),
    'OUTPUTEVENTS': pd.read_csv(f"{nested_dir}/OUTPUTEVENTS.csv"),
    'PRESCRIPTIONS': pd.read_csv(f"{nested_dir}/PRESCRIPTIONS.csv"),
    'NOTEEVENTS': pd.read_csv(f"{nested_dir}/NOTEEVENTS.csv"),
}


# Build Patient-wise JSON
# Use lowercase 'subject_id' for consistency
subject_col = 'subject_id'
patient_jsons = {}

for subject_id in csv_files['PATIENTS'][subject_col].unique():
    patient_record = {}
    for table_name, df in csv_files.items():
        if subject_col in df.columns:
            filtered = df[df[subject_col] == subject_id]
            filtered = filtered.drop(columns=[col for col in filtered.columns if 'row_id' in col], errors='ignore')
            if not filtered.empty:
                patient_record[table_name] = filtered.to_dict(orient='records')
    patient_jsons[str(subject_id)] = patient_record


# Save and Download JSON
json_path = '/content/mimiciii_full_patients.json'
with open(json_path, 'w') as f:
    json.dump(patient_jsons, f, indent=2)

# Download it
files.download(json_path)



# Preview what json file looks like (print first patient and print first 1–2 entries per table)
json_path = '/content/mimiciii_full_patients.json'
with open(json_path, 'r') as f:
    patient_data = json.load(f)

first_patient_id = list(patient_data.keys())[0]
first_patient = patient_data[first_patient_id]

print(f"📌 Patient ID: {first_patient_id}")
print("📁 Tables included:", list(first_patient.keys()))

for table, rows in first_patient.items():
    print(f"\n📄 {table} ({len(rows)} records):")
    for i, row in enumerate(rows[:2]):
        print(f"Record {i+1}:")
        print(json.dumps(row, indent=2))


json_path = '/content/mimiciii_full_patients.json'

# Load JSON
with open(json_path, 'r') as f:
    all_data = json.load(f)

# Filter: Keep only PATIENTS, ADMISSIONS, ICUSTAYS
filtered_data = {
    pid: {k: v for k, v in tables.items() if k in ['PATIENTS', 'ADMISSIONS', 'ICUSTAYS']}
    for pid, tables in all_data.items()
}

# Save new JSON
filtered_json_path = '/content/mimic_small.json'
with open(filtered_json_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

# Download
files.download(filtered_json_path)