import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


output_dir = Path("data_overview")
output_dir.mkdir(exist_ok=True)

data_dir = Path("/projects/f_miarc_1/Hypertension")
df = pd.read_csv(data_dir / "Hypertension With Comorbidities 20250404.csv", encoding="latin1",dtype=str, 
    engine="pyarrow")  
print(f"Shape: {df.shape}")
df.head(2)


print("Null values:", df["CURRENT_ICD10_LIST"].isna().sum())
print("Sample values:\n", df["CURRENT_ICD10_LIST"].dropna().head(10).tolist())
df["CURRENT_ICD10_LIST"].dropna().head(10).tolist()


labs = pd.read_csv(data_dir / "Hypertension Labs 20250417.csv", sep="|", quotechar='"',encoding="latin1",dtype=str, 
    engine="pyarrow")
print(f"Shape: {labs.shape}")
labs.head(5)

print("\n--- Column Names List ---")
print(labs.columns.tolist())


meds = pd.read_csv(data_dir / "Hypertension Medications 20250417.csv", encoding="latin1",dtype=str, 
    engine="pyarrow")
meds.head(5)
print("\n--- Column Names List ---")
print(meds.columns.tolist())



unique_mrn_count = meds["PAT_MRN_ID_ENCRYPT"].nunique()

print(f"Number of unique MRN codes: {unique_mrn_count:,}")


unique_mrn_count_l = labs["PAT_MRN_ID_ENCRYPT"].nunique()

print(f"Number of unique MRN codes: {unique_mrn_count_l:,}")

unique_mrn_count_d = df["PAT_MRN_ID_ENCRYPT"].nunique()

print(f"Number of unique MRN codes: {unique_mrn_count_d:,}")



num_patients = df["PAT_MRN_ID_ENCRYPT"].nunique()
print(f"Unique patients: {num_patients:,}")
race = df["RACE"].nunique()
print(f"Races: {race:,}")
print(race)
icd10_count = df["CURRENT_ICD10_LIST"].nunique()
print(f"Unique icd10_count: {icd10_count:,}")
ethn = df["ETHNICITY"].nunique()
print(f"Ethnicity: {ethn:,}")
print(ethn)
lang = df["LANGUAGE"].nunique()
print(f"Language: {lang:,}")
print(lang)
print("Unique Race values:")
print(df["RACE"].unique())

print("\nUnique Ethnicity values:")
print(df["ETHNICITY"].unique())

print("\nUnique Language values:")
print(df["LANGUAGE"].unique())

print("\nUnique Sex values:")
print(df["SEX"].unique())

diagnosis_count = df["DX_NAME"].nunique()
print(f"Unique Diagnosis name: {diagnosis_count:,}")




age = pd.to_numeric(df["PATIENT_AGE"], errors="coerce")

print("\n--- Age Summary ---")
print(f"Minimum age: {age.min()}")
print(f"Maximum age: {age.max()}")
print(f"Median age: {age.median()}")
print(f"Average age: {age.mean():.2f}")



fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")


df["SEX"].value_counts().plot(kind="bar", ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Sex Distribution")
axes[0,0].set_xlabel("")


df["RACE"].value_counts().head(10).plot(kind="barh", ax=axes[0,1], color="steelblue")
axes[0,1].set_title("Top 10 Race Categories")


df["ETHNICITY"].value_counts().plot(kind="barh", ax=axes[1,0], color="steelblue")
axes[1,0].set_title("Ethnicity Distribution")


df["PATIENT_AGE"].dropna().plot(kind="hist", bins=20, ax=axes[1,1], color="steelblue", edgecolor="white")
axes[1,1].set_title("Age Distribution")
axes[1,1].set_xlabel("Age")


df["LANGUAGE"].value_counts().head(10).plot(kind="barh", ax=axes[2,0], color="steelblue")
axes[2,0].set_title("Top 10 Languages")


axes[2,1].set_title("Discharge Disposition")

plt.tight_layout()
# Save plot
plot_path = output_dir / "dataset_overview.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

print(f"Plot saved to: {plot_path}")
plt.show()

print(" Debug 6")