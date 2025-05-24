import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple

def save_results(results: Dict,
                best_model: Tuple[str, Any, Any],
                output_dir: str = 'data') -> None:
    """Save training results and best model."""
    metrics_dir = f"{output_dir}/results"
    models_dir = f"{output_dir}/models"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics
    metrics_path = f"{metrics_dir}/metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save best model
    model_name, model, scaler = best_model
    model_path = f"{models_dir}/champion_{model_name}_{timestamp}.pkl"
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(results[model_name].get('feature_names', [])),
        'metrics': results[model_name],
        'timestamp': timestamp
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Results saved to {metrics_path}")
    print(f"Best model saved to {model_path}")