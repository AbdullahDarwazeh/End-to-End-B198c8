import pandas as pd
import numpy as np
import yaml
import time
from paths import get_config_path, get_data_path, get_output_path, setup_project_structure

def test_path_management():
    """Test the path management system."""
    print("1 Testing Path Management System...")
    
    try:
        # Test path manager setup
        pm = setup_project_structure()
        
        # Test key paths
        config_path = get_config_path()
        data_path = get_data_path()
        output_path = get_output_path("test_output.csv")
        
        print(f"    Config path: {config_path}")
        print(f"    Data path: {data_path}")
        print(f"    Output path: {output_path}")
        
        # Test file existence
        import os
        if os.path.exists(config_path):
            print(f"    Config file exists")
        else:
            print(f"    Config file missing")
            
        if os.path.exists(data_path):
            print(f"    Data file exists")
        else:
            print(f"    Data file missing (will use sample)")
        
        return True
        
    except Exception as e:
        print(f"    Path management test failed: {e}")
        return False

def test_basic_cleaning():
    """Test the basic cleaning functionality."""
    print("\n2 Testing Basic Cleaning Functions...")
    
    try:
        from cleaning import DataCleaner
        
        # Load config
        with open(get_config_path(), 'r') as f:
            config = yaml.safe_load(f)
        
        # Create sample data for testing
        sample_data = {
            'Company Name': ['TechCorp\n4.2', 'DataInc\n3.8', 'CleanCorp'],
            'Salary Estimate': ['$100K-$150K (Glassdoor est.)', 'Unknown', '$80K-$120K (Glassdoor est.)'],
            'Rating': [-1, 3.8, 4.0],
            'Founded': [2010, -1, 2015],
            'Job Description': ['Python SQL ML', 'Data analysis R', 'Statistics Python'],
            'Industry': ['Tech', 'Tech', 'Finance'],
            'Location': ['NY', 'CA', 'IL'],
            'Sector': ['Technology', 'Technology', 'Finance'],
            'Type of ownership': ['Private', 'Public', 'Private']
        }
        
        df = pd.DataFrame(sample_data)
        cleaner = DataCleaner(config)
        
        # Test each cleaning step
        print("   Testing company name cleaning...")
        df_clean = cleaner.clean_company_names(df)
        
        print("   Testing salary parsing...")
        df_clean = cleaner.parse_salary(df_clean)
        
        print("   Testing missing value imputation...")
        df_clean = cleaner.impute_missing(df_clean)
        
        print("   Testing feature extraction...")
        df_clean = cleaner.extract_features(df_clean)
        
        print("   Testing categorical encoding...")
        df_clean = cleaner.encode_categorical(df_clean)
        
        print(f"    Basic cleaning completed. Shape: {df_clean.shape}")
        return True
        
    except Exception as e:
        print(f"    Basic cleaning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_pipeline():
    """Test the advanced pipeline selector."""
    print("\n3 Testing Advanced Pipeline Selector...")
    
    try:
        from advanced_pipeline_selector import AdvancedPipelineSelector
        
        # Load real or sample data
        try:
            df = pd.read_csv(get_data_path())
        except FileNotFoundError:
            print("   Using sample data for testing...")
            # Create larger sample for meaningful testing
            df = create_test_dataset(100)
        
        print(f"   Dataset shape: {df.shape}")
        
        # Initialize selector
        selector = AdvancedPipelineSelector()
        
        # Test combination generation
        combinations = selector.generate_cleaning_combinations(max_combinations=5)
        print(f"    Generated {len(combinations)} cleaning combinations")
        
        # Test pipeline evaluation (with timeout)
        print("   Testing pipeline evaluation (this may take a moment)...")
        start_time = time.time()
        
        best_result = selector.adaptive_pipeline_selection(df, early_stopping=True)
        
        elapsed_time = time.time() - start_time
        print(f"    Pipeline selection completed in {elapsed_time:.2f}s")
        print(f"    Best MSE: {best_result['mse']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"    Advanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scalability_optimizer():
    """Test the scalability optimizer."""
    print("\n4 Testing Scalability Optimizer...")
    
    try:
        from scalability_optimizer import ScalabilityOptimizer
        
        # Load or create test data
        try:
            df = pd.read_csv(get_data_path())
        except FileNotFoundError:
            print("   Using sample data for testing...")
            df = create_test_dataset(500)  # Larger dataset for scalability testing
        
        print(f"   Dataset shape: {df.shape}")
        
        # Initialize optimizer
        optimizer = ScalabilityOptimizer()
        
        # Test requirement estimation
        requirements = optimizer.estimate_processing_requirements(df)
        print(f"    Processing requirements estimated")
        print(f"      Strategy: {requirements['recommended_strategy']}")
        print(f"      Memory: {requirements['current_memory_mb']:.1f}MB")
        
        # Test performance monitoring
        with optimizer.monitor_performance("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        print(f"    Performance monitoring working")
        
        # Test chunk size calculation
        chunk_size = optimizer.adaptive_chunk_sizing(df)
        print(f"    Adaptive chunk size: {chunk_size}")
        
        return True
        
    except Exception as e:
        print(f"    Scalability optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_integration():
    """Test the complete integrated system."""
    print("\n5 Testing Full System Integration...")
    
    try:
        # Test running the complete advanced pipeline
        from advanced_pipeline_selector import run_advanced_pipeline
        
        print("   Running complete advanced pipeline...")
        
        # Use sample data if real data not available
        try:
            data_path = get_data_path()
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print("   Creating sample data for integration test...")
            df = create_test_dataset(200)
            sample_path = get_output_path("test_sample_data.csv")
            df.to_csv(sample_path, index=False)
            data_path = sample_path
        
        # Run the pipeline
        result, cleaned_data = run_advanced_pipeline(data_path)
        
        print(f"    Integration test completed")
        print(f"      Original shape: {df.shape}")
        print(f"      Cleaned shape: {cleaned_data.shape}")
        print(f"      Best MSE: {result['mse']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"    Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_dataset(n_rows: int) -> pd.DataFrame:
    """Create a test dataset for testing purposes."""
    np.random.seed(42)
    
    data = {
        'index': range(n_rows),
        'Job Title': np.random.choice(['Data Scientist', 'Senior Data Scientist', 'Data Analyst', 'ML Engineer'], n_rows),
        'Salary Estimate': [f"${np.random.randint(80, 200)}K-${np.random.randint(120, 300)}K (Glassdoor est.)" 
                           if np.random.random() > 0.1 else 'Unknown' for _ in range(n_rows)],
        'Job Description': ['Python SQL Machine Learning ' + ' '.join(np.random.choice(['Statistics', 'R', 'Tableau'], 2)) 
                           for _ in range(n_rows)],
        'Rating': [round(np.random.uniform(1.0, 5.0), 1) if np.random.random() > 0.1 else -1.0 for _ in range(n_rows)],
        'Company Name': [f"Company_{i}" + ('\n' + str(round(np.random.uniform(1.0, 5.0), 1)) if np.random.random() > 0.8 else '') 
                        for i in range(n_rows)],
        'Location': np.random.choice(['New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Boston, MA', 'Seattle, WA'], n_rows),
        'Headquarters': np.random.choice(['New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Boston, MA'], n_rows),
        'Size': np.random.choice(['1-50 employees', '51-200 employees', '201-1000 employees', '1000+ employees'], n_rows),
        'Founded': [np.random.randint(1900, 2023) if np.random.random() > 0.1 else -1 for _ in range(n_rows)],
        'Type of ownership': np.random.choice(['Private', 'Public', 'Non-profit', 'Government'], n_rows),
        'Industry': np.random.choice(['IT Services', 'Software', 'Finance', 'Healthcare', 'Consulting'], n_rows),
        'Sector': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Business Services'], n_rows),
        'Revenue': np.random.choice(['$1M-$10M', '$10M-$100M', '$100M-$1B', '$1B+', 'Unknown'], n_rows),
        'Competitors': [f"Competitor_{np.random.randint(1, 10)}" for _ in range(n_rows)]
    }
    
    return pd.DataFrame(data)

def run_comprehensive_test():
    """Run all tests in sequence."""
    
    print(" COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    
    # Track test results
    test_results = {
        'path_management': False,
        'basic_cleaning': False,
        'advanced_pipeline': False,
        'scalability_optimizer': False,
        'full_integration': False
    }
    
    # Run tests
    test_results['path_management'] = test_path_management()
    test_results['basic_cleaning'] = test_basic_cleaning()
    test_results['advanced_pipeline'] = test_advanced_pipeline()
    test_results['scalability_optimizer'] = test_scalability_optimizer()
    test_results['full_integration'] = test_full_integration()
    
    # Summary
    print(f"\n TEST RESULTS SUMMARY:")
    print("=" * 30)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = " PASS" if result else " FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(" ALL TESTS PASSED! Your advanced data cleaning system is ready!")
    else:
        print(" Some tests failed. Please check the error messages above.")
        
    return test_results

if __name__ == "__main__":
    try:
        results = run_comprehensive_test()
        
        # Save test report
        output_path = get_output_path("test_report.txt")
        with open(output_path, 'w') as f:
            f.write("Advanced Data Cleaning System Test Report\n")
            f.write("=" * 45 + "\n\n")
            for test_name, result in results.items():
                status = "PASS" if result else "FAIL"
                f.write(f"{test_name.replace('_', ' ').title()}: {status}\n")
        
        print(f"\n Test report saved to: {output_path}")
        
    except Exception as e:
        print(f"\n Test runner failed: {e}")
        import traceback
        traceback.print_exc()