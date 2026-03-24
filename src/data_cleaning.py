import pandas as pd

# Load dataset
df = pd.read_csv("data/blood_cell_anomaly_detection.csv")

print("\n🔍 BASIC INFO")
print(df.info())

print("\n📊 FIRST 5 ROWS")
print(df.head())

print("\n❓ MISSING VALUES")
print(df.isnull().sum())

print("\n🧠 DATA TYPES")
print(df.dtypes)

# -----------------------------
# KEEP ONLY NUMERIC COLUMNS
# -----------------------------
df_numeric = df.select_dtypes(include=["number"])

print("\n✅ NUMERIC DATA SHAPE:", df_numeric.shape)

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df_numeric = df_numeric.fillna(df_numeric.mean())

print("\n✅ Missing values after cleaning:")
print(df_numeric.isnull().sum().sum())

# -----------------------------
# SAVE CLEANED DATA
# -----------------------------
df_numeric.to_csv("data/cleaned_data.csv", index=False)

print("\n💾 Cleaned data saved as data/cleaned_data.csv")