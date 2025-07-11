import pandas as pd
import yaml
from cleaning import DataCleaner
from paths import get_config_path, get_data_path, setup_project_structure

def main():
    """Test all cleaning functions step by step."""
    
    print(" Testing Data Cleaning Functions")
    print("=" * 40)
    
    # Set up project structure
    pm = setup_project_structure()
    
    # Load config
    config_file = get_config_path()
    print(f"\n Loading config from: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f" Config file not found: {config_file}")
        return
    
    # Load dataset
    data_file = get_data_path()
    print(f" Loading data from: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        print(f"   Original shape: {df.shape}")
    except FileNotFoundError:
        print(f" Data file not found: {data_file}")
        return
    
    # Check for duplicates
    print(f"\n Initial Data Quality Check:")
    print(f"   Duplicate Rows: {df.duplicated().sum()}")
    print(f"   Total Rows: {len(df)}")
    
    # Initialize cleaner
    cleaner = DataCleaner(config)
    
    # Test 1: Company name cleaning
    print(f"\n1 Testing Company Name Cleaning...")
    print(f"   Sample company names before:")
    print(f"   {df['Company Name'].head(3).tolist()}")
    
    df_cleaned = cleaner.clean_company_names(df)
    print(f"   Sample company names after:")
    print(f"   {df_cleaned['Company Name'].head(3).tolist()}")
    
    # Test 2: Salary parsing
    print(f"\n2 Testing Salary Parsing...")
    df_cleaned = cleaner.parse_salary(df_cleaned)
    
    print(f"   Random Sample of Parsed Salaries:")
    sample_data = df_cleaned[['Salary Estimate', 'Avg Salary']].sample(5, random_state=42)
    for idx, row in sample_data.iterrows():
        print(f"   '{row['Salary Estimate']}' → ${row['Avg Salary']:,.0f}")
    
    print(f"\n   Salary Summary:")
    salary_stats = df_cleaned['Avg Salary'].describe()
    print(f"   Min: ${salary_stats['min']:,.0f}")
    print(f"   Max: ${salary_stats['max']:,.0f}")
    print(f"   Mean: ${salary_stats['mean']:,.0f}")
    print(f"   Median: ${salary_stats['50%']:,.0f}")
    
    # Test 3: Missing value imputation
    print(f"\n3 Testing Missing Value Imputation...")
    
    # Show before
    print(f"   Before imputation:")
    for col in ['Rating', 'Founded']:
        missing = (df_cleaned[col] == -1).sum() + df_cleaned[col].isna().sum()
        print(f"   {col}: {missing} missing values")
    
    df_cleaned = cleaner.impute_missing(df_cleaned)
    
    # Show after
    print(f"   After imputation:")
    for col in ['Rating', 'Founded']:
        missing = (df_cleaned[col] == -1).sum() + df_cleaned[col].isna().sum()
        print(f"   {col}: {missing} missing values")
        print(f"   {col} range: {df_cleaned[col].min():.1f} - {df_cleaned[col].max():.1f}")
    
    # Test 4: Text normalization
    print(f"\n4 Testing Text Normalization...")
    print(f"   Sample before normalization:")
    print(f"   Job Title: {df_cleaned['Job Title'].iloc[0]}")
    
    df_cleaned = cleaner.normalize_text(df_cleaned)
    
    print(f"   Sample after normalization:")
    print(f"   Job Title: {df_cleaned['Job Title'].iloc[0]}")
    
    # Test 5: Outlier handling
    print(f"\n5 Testing Outlier Handling...")
    
    # Show before
    print(f"   Before outlier handling:")
    print(f"   Rating range: {df_cleaned['Rating'].min():.1f} - {df_cleaned['Rating'].max():.1f}")
    print(f"   Founded range: {df_cleaned['Founded'].min():.0f} - {df_cleaned['Founded'].max():.0f}")
    
    df_cleaned = cleaner.handle_outliers(df_cleaned)
    
    # Show after
    print(f"   After outlier handling:")
    print(f"   Rating range: {df_cleaned['Rating'].min():.1f} - {df_cleaned['Rating'].max():.1f}")
    print(f"   Founded range: {df_cleaned['Founded'].min():.0f} - {df_cleaned['Founded'].max():.0f}")
    print(f"   Rows after outlier removal: {len(df_cleaned)}")
    
    # Test 6: Feature engineering
    print(f"\n6 Testing Feature Engineering...")
    df_cleaned = cleaner.extract_features(df_cleaned)
    
    skill_cols = [col for col in df_cleaned.columns if col.startswith('skill_')]
    print(f"   Created {len(skill_cols)} skill features:")
    for skill_col in skill_cols[:5]:  # Show first 5
        skill_count = df_cleaned[skill_col].sum()
        print(f"   {skill_col}: {skill_count} jobs")
    
    # Test 7: Categorical encoding
    print(f"\n7 Testing Categorical Encoding...")
    print(f"   Columns before encoding: {len(df_cleaned.columns)}")
    
    df_cleaned = cleaner.encode_categorical(df_cleaned)
    
    print(f"   Columns after encoding: {len(df_cleaned.columns)}")
    
    # Show encoded columns
    encoded_cols = [col for col in df_cleaned.columns if col.startswith(('Industry_', 'Location_', 'Sector_', 'Type of ownership_'))]
    print(f"   Created {len(encoded_cols)} dummy variables")
    
    # Final validation
    print(f"\n8 Final Data Validation...")
    validation_report = cleaner.validate_data(df_cleaned)
    
    print(f"   Final shape: {validation_report['total_rows']} × {validation_report['total_columns']}")
    print(f"   Missing values: {validation_report['missing_values']}")
    print(f"   Duplicate rows: {validation_report['duplicate_rows']}")
    print(f"   Memory usage: {validation_report['memory_usage_mb']:.1f} MB")
    
    # Save cleaned data for inspection
    output_path = pm.get_output_dir() / "test_cleaned_data.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n Test cleaned data saved to: {output_path}")
    
    print(f"\n All cleaning tests completed successfully!")
    
    return df_cleaned

if __name__ == "__main__":
    try:
        cleaned_df = main()
        print(f"Final cleaned dataset shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"\n Error during testing: {e}")
        import traceback
        traceback.print_exc()