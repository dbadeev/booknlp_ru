import pandas as pd
import glob
import os

REPORT_DIR = "reports/tokenization_audit"


def analyze():
    files = glob.glob(f"{REPORT_DIR}/*.csv")
    if not files:
        print("No reports found.")
        return

    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except:
            pass

    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)

    print("\n=== 1. TOTAL ERRORS BY MODEL & DATASET ===")
    pivot = pd.crosstab(df['model'], df['dataset'])
    print(pivot)

    print("\n=== 2. ERROR TYPES BY MODEL ===")
    print(pd.crosstab(df['model'], df['error_type']))

    print("\n=== 3. TOP-5 FREQUENT ERRORS ===")
    for model in df['model'].unique():
        print(f"\nMODEL: {model}")
        subset = df[df['model'] == model]
        # Группируем по паре (Gold -> Pred) чтобы увидеть самые частые баги
        top = subset.groupby(['gold', 'pred']).size().reset_index(name='count')
        print(top.sort_values('count', ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    analyze()
