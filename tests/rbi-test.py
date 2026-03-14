"""
Diagnostic script to check RBI repo rate data.
Run this to see what's actually in your Excel file.
"""

import pandas as pd
import numpy as np

# Load the Excel file
excel_path = "data/raw/Book1__1_.xlsx"

print("="*80)
print("DIAGNOSTIC: Checking RBI Repo Rate Data")
print("="*80)

# Load Clean Data sheet
df = pd.read_excel(excel_path, sheet_name="Clean Data")

print(f"\n1. Sheet columns (first 30):")
print(df.columns[:30].tolist())

print(f"\n2. Looking for repo rate columns...")
repo_cols = [col for col in df.columns if 'repo' in str(col).lower() or 'rbi' in str(col).lower()]
print(f"Found columns: {repo_cols}")

# Check the data
print(f"\n3. Data preview (rows 1-5):")
if len(df) > 1:
    print(df.iloc[1:6, 26:30])  # Columns 26-29 should have RBI data

# Extract RBI Repo data
print(f"\n4. RBI Repo Rate column analysis:")

# Try column index 26 (RBI_Repo_Rate)
if len(df.columns) > 26:
    dates = pd.to_datetime(df.iloc[1:, 0], errors='coerce')
    repo_rate = pd.to_numeric(df.iloc[1:, 26], errors='coerce')
    
    print(f"   Column 26 (should be RBI_Repo_Rate):")
    print(f"   - Non-null values: {repo_rate.notna().sum()}")
    print(f"   - Sample values: {repo_rate.dropna().head(10).tolist()}")
    
    # Check for changes
    repo_changes = repo_rate.diff()
    non_zero_changes = repo_changes[repo_changes.abs() > 1e-6]
    
    print(f"\n5. RBI Repo Rate CHANGES:")
    print(f"   - Total non-zero changes: {len(non_zero_changes)}")
    
    if len(non_zero_changes) > 0:
        print(f"   - Change magnitudes: {non_zero_changes.unique()}")
        print(f"\n   First 10 changes:")
        for i, (date, change) in enumerate(zip(dates[non_zero_changes.index], non_zero_changes.values)):
            if pd.notna(date) and i < 10:
                print(f"      {date.date()}: {change:+.4f} ({change*100:+.2f}%)")
    else:
        print("   WARNING: No repo rate changes found!")
        print("   Possible reasons:")
        print("   - Data is constant (rate never changed)")
        print("   - Wrong column")
        print("   - Data needs cleaning")

# Try column index 28 (RBI_Repo_Change)
print(f"\n6. RBI Repo Change column (col 28) analysis:")
if len(df.columns) > 28:
    repo_change = pd.to_numeric(df.iloc[1:, 28], errors='coerce')
    
    print(f"   - Non-null values: {repo_change.notna().sum()}")
    print(f"   - Non-zero values: {(repo_change != 0).sum()}")
    print(f"   - Sample values: {repo_change.dropna().head(20).tolist()}")
    
    non_zero = repo_change[repo_change.abs() > 1e-6]
    if len(non_zero) > 0:
        print(f"\n   Non-zero changes found: {len(non_zero)}")
        print(f"   Magnitudes: {non_zero.values}")
        
        for i, (date, change) in enumerate(zip(dates[non_zero.index], non_zero.values)):
            if pd.notna(date) and i < 15:
                print(f"      {date.date()}: {change:+.6f} ({change*100:+.2f}%)")
    else:
        print("   WARNING: Column 28 is all zeros or NaN!")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)

# Determine which column to use
if len(df.columns) > 28:
    repo_change_col = pd.to_numeric(df.iloc[1:, 28], errors='coerce')
    if (repo_change_col.abs() > 1e-6).sum() > 0:
        print("✓ Use column 28 (RBI_Repo_Change) - has actual changes")
    else:
        print("✗ Column 28 appears to be empty/zeros")
        print("  Check if you need to calculate changes manually")

if len(df.columns) > 26:
    repo_rate_col = pd.to_numeric(df.iloc[1:, 26], errors='coerce')
    repo_changes_calc = repo_rate_col.diff()
    if (repo_changes_calc.abs() > 1e-6).sum() > 0:
        print("✓ Can calculate changes from column 26 (RBI_Repo_Rate)")
        print(f"  Found {(repo_changes_calc.abs() > 1e-6).sum()} changes")

print("="*80)