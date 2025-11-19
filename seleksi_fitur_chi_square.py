import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

DATA_PATH = "teen_phone_addiction_dataset.csv"

def make_target(classes_from, bins=[0.0, 4.0, 7.0, 10.0001], labels=["Low","Medium","High"]):
    return pd.cut(classes_from, bins=bins, labels=labels, right=False, include_lowest=True)

def main():
    df = pd.read_csv(DATA_PATH)
    df["Addiction_Class"] = make_target(df["Addiction_Level"])
    target_col = "Addiction_Class"
    exclude_cols = {"ID", "Name", "Addiction_Level", target_col}

    rows = []
    for col in df.columns:
        if col in exclude_cols:
            continue

        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                binned = pd.qcut(s, q=4, duplicates="drop")
            except ValueError:
                try:
                    binned = pd.qcut(s, q=2, duplicates="drop")
                except Exception:
                    continue
            contingency = pd.crosstab(binned, df[target_col])
        else:
            contingency = pd.crosstab(s.astype(str), df[target_col])

        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(contingency)
            rows.append({
                "feature": col,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "categories": contingency.shape[0],
                "significant_0.05": p < 0.05
            })

    out = pd.DataFrame(rows).sort_values(["p_value","chi2"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv("chi_square_results.csv", index=False)
    print("✅ chi_square_results.csv berhasil dibuat dengan", len(out), "fitur.")

if __name__ == "__main__":
    main()
