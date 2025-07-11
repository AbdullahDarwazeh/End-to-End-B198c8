import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Dict, List, Tuple, Any
import yaml
import warnings
from paths import get_config_path, get_data_path, get_output_path
warnings.filterwarnings('ignore')

class DynamicCleaningTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that applies cleaning steps with proper fit/transform."""
    
    def __init__(self, cleaning_config: Dict[str, Any], base_config: Dict[str, Any]):
        self.cleaning_config = cleaning_config
        self.base_config = base_config
        self.cleaner = None
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """Fit the cleaning transformer."""
        # Import here to avoid circular imports
        from cleaning import DataCleaner
        
        # Create cleaner with merged config
        config = self.base_config.copy()
        config['cleaning'].update(self.cleaning_config)
        self.cleaner = DataCleaner(config)

        # Fit the cleaner to the data
        X_temp = self._apply_cleaning(X.copy())
        self.feature_names_ = X_temp.columns.tolist()
        return self
    
    def transform(self, X):
        """Transform the data using fitted cleaning steps."""
        if self.cleaner is None:
            raise ValueError("Transformer not fitted yet.")
        
        X_clean = self._apply_cleaning(X.copy())
        
        # Ensure consistent features
        for col in self.feature_names_:
            if col not in X_clean.columns:
                X_clean[col] = 0  
        
        # Keep only fitted features in same order
        X_clean = X_clean[self.feature_names_]
        return X_clean
    
    def _apply_cleaning(self, df):
        """Apply all cleaning steps."""
        df = self.cleaner.clean_company_names(df)
        df = self.cleaner.parse_salary(df)
        df = self.cleaner.impute_missing(df)
        df = self.cleaner.normalize_text(df)
        df = self.cleaner.handle_outliers(df)
        df = self.cleaner.extract_features(df)
        df = self.cleaner.encode_categorical(df)
        
        # Select only numeric columns for ML
        feature_cols = ['Rating', 'Founded'] + \
                      [f'skill_{skill.lower().replace(" ", "_")}' 
                       for skill in self.cleaner.config['cleaning']['feature_engineering']['skills']] + \
                      [col for col in df.columns if col.startswith(('Industry_', 'Location_', 'Sector_', 'Type of ownership_'))]
        
        available_features = [col for col in feature_cols if col in df.columns]
        return df[available_features]

class AdvancedPipelineSelector:
    """Advanced pipeline selector with dynamic technique combinations."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = get_config_path()
            
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Define technique options
        self.technique_options = {
            'missing_imputation': {
                'Rating': [{'method': 'mean'}, {'method': 'median'}, {'method': 'drop'}],
                'Founded': [{'method': 'mean'}, {'method': 'median'}, {'method': 'mode'}]
            },
            'outlier_handling': [
                {'method': 'cap', 'columns': ['Rating', 'Founded']},
                {'method': 'remove', 'columns': ['Rating', 'Founded']},
                {'method': 'cap', 'columns': ['Rating']},  
            ],
            'categorical_encoding': [
                {'method': 'one-hot', 'columns': ['Industry', 'Location', 'Sector', 'Type of ownership']},
                {'method': 'label', 'columns': ['Industry', 'Location', 'Sector', 'Type of ownership']},
                {'method': 'one-hot', 'columns': ['Industry', 'Location']}, 
            ]
        }
    
    def generate_cleaning_combinations(self, max_combinations: int = None) -> List[Dict]:
        """Generate all possible cleaning technique combinations."""
        if max_combinations is None:
            max_combinations = self.base_config.get('scalability', {}).get('max_combinations_to_test', 20)
        
        combinations = []
        
        # Generate combinations of missing value methods
        rating_methods = self.technique_options['missing_imputation']['Rating']
        founded_methods = self.technique_options['missing_imputation']['Founded']
        outlier_methods = self.technique_options['outlier_handling']
        encoding_methods = self.technique_options['categorical_encoding']
        
        count = 0
        for rating_method in rating_methods:
            for founded_method in founded_methods:
                for outlier_method in outlier_methods:
                    for encoding_method in encoding_methods:
                        if count >= max_combinations:
                            break
                        
                        combination = {
                            'missing_values': {
                                'Rating': rating_method,
                                'Founded': founded_method,
                                'Competitors': {'method': 'replace', 'value': 'Unknown'},
                                'Revenue': {'method': 'replace', 'value': 'Unknown'},
                                'Industry': {'method': 'replace', 'value': 'Other'}
                            },
                            'outlier_handling': outlier_method,
                            'categorical_encoding': encoding_method
                        }
                        combinations.append(combination)
                        count += 1
                        
        self.logger.info(f"Generated {len(combinations)} cleaning combinations")
        return combinations[:max_combinations]
    
    def evaluate_pipeline_performance(self, df: pd.DataFrame, cleaning_config: Dict) -> Dict:
        """Evaluate a single cleaning pipeline with proper CV."""
        try:
            # Create pipeline with cleaning and scaling
            cleaning_transformer = DynamicCleaningTransformer(cleaning_config, self.base_config)
            
            pipeline = Pipeline([
                ('cleaning', cleaning_transformer),
                ('scaling', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Prepare target variable
            from cleaning import DataCleaner
            temp_cleaner = DataCleaner(self.base_config)
            df_temp = temp_cleaner.clean_company_names(df.copy())
            df_temp = temp_cleaner.parse_salary(df_temp)
            
            if 'Avg Salary' not in df_temp.columns or df_temp['Avg Salary'].isna().all():
                return {'error': 'No valid target variable'}
            
            # Remove rows with missing target
            valid_idx = df_temp['Avg Salary'].notna()
            X = df[valid_idx].copy()
            y = df_temp.loc[valid_idx, 'Avg Salary']
            
            if len(X) < 10:  # Minimum samples for CV
                return {'error': 'Insufficient samples after cleaning'}
            
            # Perform cross-validation with proper splitting
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipeline, X, y, 
                cv=cv, 
                scoring='neg_mean_squared_error',
                error_score='raise'
            )
            
            mse = -cv_scores.mean()
            mse_std = cv_scores.std()
            
            # Calculate additional metrics
            pipeline.fit(X, y)
            n_features = pipeline.named_steps['cleaning'].transform(X).shape[1]
            
            return {
                'mse': mse,
                'mse_std': mse_std,
                'rmse': np.sqrt(mse),
                'n_samples': len(X),
                'n_features': n_features,
                'cv_scores': cv_scores.tolist(),
                'config': cleaning_config
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline evaluation failed: {str(e)}")
            return {'error': str(e), 'config': cleaning_config}
    
    def adaptive_pipeline_selection(self, df: pd.DataFrame, 
                                  performance_threshold: float = None,
                                  early_stopping: bool = True) -> Dict:
        """Adaptively select best cleaning pipeline with early stopping."""
        
        # Get thresholds from config
        if performance_threshold is None:
            performance_threshold = self.base_config.get('performance_thresholds', {}).get('target_mse_threshold', float('inf'))
        
        early_stopping_patience = self.base_config.get('performance_thresholds', {}).get('early_stopping_patience', 5)
        
        combinations = self.generate_cleaning_combinations()
        results = []
        best_mse = float('inf')
        no_improvement_count = 0
        
        self.logger.info(f"Starting adaptive pipeline selection with {len(combinations)} combinations")
        self.logger.info(f"Performance threshold: {performance_threshold}, Early stopping patience: {early_stopping_patience}")
        
        for i, combo in enumerate(combinations):
            self.logger.info(f"Evaluating combination {i+1}/{len(combinations)}")
            result = self.evaluate_pipeline_performance(df, combo)
            
            if 'error' not in result:
                results.append(result)
                current_mse = result['mse']
                
                # Check for improvement
                if current_mse < best_mse:
                    improvement = best_mse - current_mse
                    best_mse = current_mse
                    no_improvement_count = 0
                    self.logger.info(f" New best MSE: {best_mse:,.0f} (improvement: {improvement:,.0f})")
                else:
                    no_improvement_count += 1
                    self.logger.info(f" No improvement. Current: {current_mse:,.0f}, Best: {best_mse:,.0f}")
                
                # Early stopping
                if early_stopping and no_improvement_count >= early_stopping_patience:
                    self.logger.info(f" Early stopping: No improvement for {early_stopping_patience} iterations")
                    break
                
                # Performance threshold stopping
                if current_mse <= performance_threshold:
                    self.logger.info(f" Performance threshold {performance_threshold:,.0f} reached!")
                    break
            else:
                self.logger.warning(f" Combination {i+1} failed: {result['error']}")
        
        if not results:
            raise ValueError("No valid pipeline configurations found")
        
        # Return best result with ranking
        best_result = min(results, key=lambda x: x['mse'])
        
        # Add ranking information
        sorted_results = sorted(results, key=lambda x: x['mse'])
        best_result['ranking'] = {
            'position': 1,
            'total_evaluated': len(results),
            'improvement_over_worst': sorted_results[-1]['mse'] - best_result['mse'] if len(sorted_results) > 1 else 0,
            'top_3_configs': [r['config'] for r in sorted_results[:3]]
        }
        
        self.logger.info(f" Best pipeline selected with MSE: {best_result['mse']:,.0f}")
        return best_result
    
    def explain_pipeline_selection(self, result: Dict) -> str:
        """Generate explanation for why this pipeline was selected."""
        config = result['config']
        
        explanation = f"""
 **BEST PIPELINE SELECTED**

**Performance Metrics:**
• MSE: {result['mse']:,.0f}
• RMSE: ${result['rmse']:,.0f}
• Samples Used: {result['n_samples']:,}
• Features Created: {result['n_features']}
• Cross-Validation Std: {result.get('mse_std', 0):,.0f}

**Pipeline Configuration:**
• Missing Values: 
  - Rating: {config['missing_values']['Rating']['method'].title()}
  - Founded: {config['missing_values']['Founded']['method'].title()}
• Outlier Handling: {config['outlier_handling']['method'].title()} method
• Categorical Encoding: {config['categorical_encoding']['method'].replace('_', '-').title()}

**Ranking Information:**
• Position: #{result['ranking']['position']} out of {result['ranking']['total_evaluated']} evaluated
• Improvement over worst: ${result['ranking']['improvement_over_worst']:,.0f} MSE reduction

**Why This Pipeline Won:**
This configuration achieved the lowest prediction error while maintaining good feature balance and data quality.
        """
        
        return explanation.strip()

    def run_best_pipeline(self, df: pd.DataFrame, best_result: Dict) -> pd.DataFrame:
        """Apply the best cleaning pipeline to get cleaned data."""
        from cleaning import DataCleaner
        
        # Create cleaner with best config
        config = self.base_config.copy()
        config['cleaning'].update(best_result['config'])
        cleaner = DataCleaner(config)
        
        # Apply all cleaning steps
        df_clean = df.copy()
        df_clean = cleaner.clean_company_names(df_clean)
        df_clean = cleaner.parse_salary(df_clean)
        df_clean = cleaner.impute_missing(df_clean)
        df_clean = cleaner.normalize_text(df_clean)
        df_clean = cleaner.handle_outliers(df_clean)
        df_clean = cleaner.extract_features(df_clean)
        df_clean = cleaner.encode_categorical(df_clean)
        
        self.logger.info(f"Applied best pipeline. Final shape: {df_clean.shape}")
        return df_clean

# Example usage function
def run_advanced_pipeline(data_path: str = None):
    """Example of running the advanced pipeline selector."""
    
    print(" Starting Advanced Data Cleaning System")
    print("=" * 50)
    
    # Initialize selector
    selector = AdvancedPipelineSelector()
    
    # Load data
    if data_path is None:
        data_path = get_data_path()
    
    print(f" Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Run adaptive selection
    print("\n Starting adaptive pipeline selection...")
    best_result = selector.adaptive_pipeline_selection(df)
    
    # Print explanation
    print("\n" + "=" * 50)
    print(selector.explain_pipeline_selection(best_result))
    
    # Get cleaned data
    print("\n Applying best pipeline to clean data...")
    df_clean = selector.run_best_pipeline(df, best_result)
    print(f"   Cleaned data shape: {df_clean.shape}")
    
    # Save results
    output_path = get_output_path("advanced_cleaned_data.csv")
    df_clean.to_csv(output_path, index=False)
    print(f" Cleaned data saved to: {output_path}")
    
    return best_result, df_clean

if __name__ == "__main__":
    try:
        result, cleaned_data = run_advanced_pipeline()
        print(f"\n Advanced cleaning complete! Best MSE: {result['mse']:,.0f}")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()