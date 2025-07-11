import pandas as pd
import yaml
from pipeline import PipelineSelector
from paths import get_config_path, get_data_path, setup_project_structure, get_output_path

def main(dataset_name: str = "Uncleaned_DS_jobs.csv"):
    """Main function to run data cleaning pipeline."""
    
    print(" Starting Data Cleaning Pipeline")
    print("=" * 40)
    
    # Set up project structure
    path_manager = setup_project_structure()
    
    # Load config using path manager
    config_file = get_config_path()
    print(f"\n Loading config from: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f" Config file not found: {config_file}")
        return
    
    # Load dataset using path manager
    if dataset_name == "SyntheticData.csv":
        data_file = get_data_path("SyntheticData.csv")
    else:
        data_file = get_data_path("Uncleaned_DS_jobs.csv")
    
    print(f" Loading data from: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        print(f"   Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f" Data file not found: {data_file}")
        print("Please ensure your data file is in the data directory.")
        return
    
    # Define cleaning combinations to test
    cleaning_combinations = [
        {
            'missing_values': {
                'Rating': {'method': 'median'},
                'Founded': {'method': 'median'},
                'Competitors': {'method': 'replace', 'value': 'Unknown'},
                'Revenue': {'method': 'replace', 'value': 'Unknown'}
            },
            'outlier_handling': {
                'columns': ['Rating', 'Founded'],
                'method': 'cap'
            },
            'categorical_encoding': {
                'columns': ['Industry', 'Location', 'Sector', 'Type of ownership'],
                'method': 'one-hot'
            }
        },
        {
            'missing_values': {
                'Rating': {'method': 'mean'},
                'Founded': {'method': 'mean'},
                'Competitors': {'method': 'replace', 'value': 'Unknown'},
                'Revenue': {'method': 'replace', 'value': 'Unknown'}
            },
            'outlier_handling': {
                'columns': ['Rating', 'Founded'],
                'method': 'remove'
            },
            'categorical_encoding': {
                'columns': ['Industry', 'Location', 'Sector', 'Type of ownership'],
                'method': 'label'
            }
        },
        {
            'missing_values': {
                'Rating': {'method': 'median'},
                'Founded': {'method': 'mode'},
                'Competitors': {'method': 'replace', 'value': 'Unknown'},
                'Revenue': {'method': 'replace', 'value': 'Unknown'}
            },
            'outlier_handling': {
                'columns': ['Rating', 'Founded'],
                'method': 'cap'
            },
            'categorical_encoding': {
                'columns': ['Industry', 'Location', 'Sector', 'Type of ownership'],
                'method': 'one-hot'
            }
        }
    ]
    
    # Initialize pipeline with path-aware config
    print(f"\n Initializing pipeline selector...")
    selector = PipelineSelector(config_file)
    
    # Evaluate pipelines
    print(f"\n Evaluating {len(cleaning_combinations)} pipeline combinations...")
    best_result = selector.evaluate_pipeline(df, cleaning_combinations)
    
    if best_result:
        print("\n BEST CONFIGURATION FOUND:")
        print(f"   MSE: {best_result['mse']:,.0f}")
        print(f"   RMSE: ${best_result['rmse']:,.0f}")
        print(f"   Samples: {best_result['n_samples']:,}")
        print(f"   Best Config: {best_result['config']}")
        
        # Apply best pipeline and save results
        print(f"\n🧹 Applying best pipeline...")
        df_clean = selector.run_best_pipeline(df, best_result)
        
        # Save cleaned data
        output_file = get_output_path(f"cleaned_{dataset_name}")
        df_clean.to_csv(output_file, index=False)
        print(f" Cleaned data saved to: {output_file}")
        
        print(f"\n Cleaned Data Summary:")
        print(f"   Shape: {df_clean.shape}")
        print(f"   Columns: {len(df_clean.columns)}")
        
        # Show sample of cleaned data
        print(f"\n Sample of cleaned data:")
        print(df_clean.head(3))
        
        return best_result, df_clean
    else:
        print(" No valid pipeline configuration found!")
        return None, None

def test_both_datasets():
    """Test on both original and synthetic datasets."""
    
    print(" Testing on Multiple Datasets")
    print("=" * 50)
    
    # Test on original dataset
    print("\n1 Testing on Original Dataset:")
    result1, cleaned1 = main("Uncleaned_DS_jobs.csv")
    
    # Test on synthetic dataset
    print("\n2 Testing on Synthetic Dataset:")
    result2, cleaned2 = main("SyntheticData.csv")
    
    # Compare results
    if result1 and result2:
        print("\n COMPARISON RESULTS:")
        print(f"Original Dataset MSE: {result1['mse']:,.0f}")
        print(f"Synthetic Dataset MSE: {result2['mse']:,.0f}")
        
        if result1['mse'] < result2['mse']:
            print(" Original dataset has better performance")
        else:
            print(" Synthetic dataset has better performance")

if __name__ == "__main__":
    try:
        # Run on default dataset
        main()
        
        # Optionally test both datasets
        # test_both_datasets()
        
    except Exception as e:
        print(f"\n Error occurred: {e}")
        print("\nTrying to set up project structure...")
        setup_project_structure()