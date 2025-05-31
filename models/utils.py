import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np

def save_results(results: Dict,
                best_model: Tuple[str, Any, Any],
                output_dir: str = 'data') -> None:
    """Save training results and best model."""
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results dictionary
    json_results = {
        model_name: {
            k: convert_numpy(v) for k, v in metrics.items()
        } for model_name, metrics in results.items()
    }
    
    # Save metrics
    metrics_dir = os.path.join(output_dir, 'results')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f'metrics_ensemble_{timestamp}.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    # Save best model
    model_name, model, scaler = best_model
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'best_model_{timestamp}.pkl')
    
    model_data = {
        'name': model_name,
        'model': model,
        'scaler': scaler
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Results saved to {metrics_path}")
    print(f"Best model saved to {model_path}")