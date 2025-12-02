"""
🦷 ENHANCED FLASK API FOR BITEMARK CLASSIFICATION
Production-grade REST API with comprehensive error handling and monitoring

Features:
- Unified preprocessing ensuring train/inference consistency
- Comprehensive error handling and logging
- Real-time monitoring and analytics
- Secure file handling with validation
- Production-ready performance optimization
"""

import os
import sys
import json
import uuid
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Setup comprehensive logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from shared_preprocessing import SharedPreprocessor, PreprocessingConfig
    from global_utils import setup_gpu, GlobalConfig, save_metrics
    from enhanced_cnn import EnhancedBiteMarkCNN  # Import for custom layers
except ImportError as e:
    logger.error(f"Critical import error: {e}")
    # Create fallback implementations
    SharedPreprocessor = None
    PreprocessingConfig = None
    EnhancedBiteMarkCNN = None

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Enhanced Configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
    'UPLOAD_FOLDER': 'uploads',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
    'MODEL_PATH': '../models/best_model.h5',
    'METRICS_PATH': '../outputs/metrics.json',
    'LOG_PREDICTIONS': True,
    'MAX_HISTORY_SIZE': 1000
})

# Global variables with proper initialization
model: Optional[tf.keras.Model] = None
preprocessor: Optional[SharedPreprocessor] = None
class_names: List[str] = ['dog', 'human', 'snake']
model_metrics: Dict[str, Any] = {}
analysis_history: List[Dict[str, Any]] = []
api_stats: Dict[str, Any] = {
    'total_predictions': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'start_time': datetime.now().isoformat(),
    'model_loaded_time': None
}


def initialize_api():
    """Initialize API with proper error handling"""
    global model, preprocessor, model_metrics
    
    logger.info("Initializing Enhanced BiteMark API...")
    
    try:
        # Setup directories
        Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
        
        # Setup GPU if available
        if 'setup_gpu' in globals():
            gpu_available = setup_gpu()
            logger.info(f"GPU Available: {gpu_available}")
        
        # Initialize preprocessor with exact training configuration
        if SharedPreprocessor and PreprocessingConfig:
            # Use same config as training for consistency
            config = PreprocessingConfig(
                img_size=(224, 224),
                channels=3,  # RGB for real forensic images
                normalize=True,
                adaptive_histogram=True,
                denoise=True,
                clahe_clip_limit=2.0,
                clahe_grid_size=(8, 8)
            )
            preprocessor = SharedPreprocessor(config)
            logger.info("Shared preprocessor initialized")
        else:
            logger.error("SharedPreprocessor not available - using fallback")
        
        # Load trained model with custom object scope
        model_path = app.config['MODEL_PATH']
        if os.path.exists(model_path):
            try:
                # Import and register custom layers
                if EnhancedBiteMarkCNN:
                    from enhanced_cnn import AttentionModule
                    
                    # Load model with custom objects
                    custom_objects = {'AttentionModule': AttentionModule}
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                else:
                    # Fallback - try loading without custom objects
                    model = tf.keras.models.load_model(model_path)
                    
                logger.info(f"Model loaded successfully: {model_path}")
                logger.info(f"   Input shape: {model.input_shape}")
                logger.info(f"   Output shape: {model.output_shape}")
                logger.info(f"   Total parameters: {model.count_params():,}")
                api_stats['model_loaded_time'] = datetime.now().isoformat()
                
            except Exception as model_error:
                logger.error(f"Model loading failed: {model_error}")
                logger.info("Trying to load alternative model...")
                
                # Try loading the enhanced working model instead
                alternative_path = '../models/best_model_enhanced.h5'
                if os.path.exists(alternative_path):
                    try:
                        if EnhancedBiteMarkCNN:
                            from enhanced_cnn import AttentionModule
                            custom_objects = {'AttentionModule': AttentionModule}
                            model = tf.keras.models.load_model(alternative_path, custom_objects=custom_objects)
                        else:
                            model = tf.keras.models.load_model(alternative_path)
                        logger.info(f"Alternative model loaded: {alternative_path}")
                        api_stats['model_loaded_time'] = datetime.now().isoformat()
                    except Exception as alt_error:
                        logger.error(f"Alternative model loading also failed: {alt_error}")
                        model = None
                else:
                    logger.error(f"Alternative model not found: {alternative_path}")
                    model = None
        else:
            logger.error(f"Model file not found: {model_path}")
            model = None
        
        # Load model metrics
        metrics_path = app.config['METRICS_PATH']
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
            logger.info(f"Model metrics loaded: {len(model_metrics)} metrics")
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")
            model_metrics = {}
        
        logger.info("API initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"API initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


def validate_image(file_path: str) -> bool:
    """Validate uploaded image file"""
    try:
        # Try to load image
        image = cv2.imread(file_path)
        if image is None:
            return False
        
        # Check image dimensions
        h, w = image.shape[:2]
        if h < 50 or w < 50:  # Minimum size check
            return False
        
        if h > 4096 or w > 4096:  # Maximum size check
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False


def preprocess_image_for_prediction(image_path: str) -> Optional[np.ndarray]:
    """Preprocess image using shared preprocessor for consistency"""
    try:
        if preprocessor is not None:
            # Use shared preprocessor for consistency with training
            processed_image = preprocessor.load_and_preprocess_image(image_path)
            
            # Add batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            logger.info(f"Image preprocessed with SharedPreprocessor: {processed_image.shape}")
            return processed_image
        else:
            # Fallback preprocessing if shared module not available
            logger.warning("Using fallback preprocessing")
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to match training
            img = cv2.resize(img, (224, 224))
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            logger.info(f"Image preprocessed with fallback: {img.shape}")
            return img
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None


def log_prediction(image_filename: str, prediction_result: Dict[str, Any], 
                  processing_time: float, success: bool):
    """Log prediction for monitoring and analytics"""
    try:
        if not app.config['LOG_PREDICTIONS']:
            return
        
        # Update global stats
        api_stats['total_predictions'] += 1
        if success:
            api_stats['successful_predictions'] += 1
        else:
            api_stats['failed_predictions'] += 1
        
        # Create prediction log entry
        log_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'filename': image_filename,
            'success': success,
            'processing_time_ms': round(processing_time * 1000, 2),
            'prediction': prediction_result if success else None,
            'model_version': model_metrics.get('model_type', 'unknown')
        }
        
        # Add to history (with size limit)
        analysis_history.append(log_entry)
        if len(analysis_history) > app.config['MAX_HISTORY_SIZE']:
            analysis_history.pop(0)  # Remove oldest entry
        
        logger.info(f"Prediction logged: {log_entry['id']}")
        
    except Exception as e:
        logger.error(f"Prediction logging failed: {e}")


# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'uptime_seconds': (datetime.now() - datetime.fromisoformat(api_stats['start_time'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds()
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with enhanced error handling"""
    start_time = datetime.now()
    image_filename = None
    
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed: {list(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        # Check model availability
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Save uploaded file with cleaner naming
        filename = secure_filename(file.filename)
        # Extract just the file extension
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
        # Create a cleaner filename with timestamp and short UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]  # Use only first 8 chars of UUID
        unique_filename = f"bite_analysis_{timestamp}_{short_uuid}.{file_ext}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(image_path)
        image_filename = unique_filename
        
        # Validate image
        if not validate_image(image_path):
            return jsonify({'error': 'Invalid image file'}), 400
        
        # **CRITICAL FIX**: Preprocess the uploaded image, not a different file
        processed_image = preprocess_image_for_prediction(image_path)
        if processed_image is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Process prediction results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Create detailed prediction result
        prediction_result = {
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'confidence_threshold_met': confidence > 0.8,  # 80% threshold
            'all_probabilities': {
                class_names[i]: round(float(predictions[0][i]) * 100, 2) 
                for i in range(len(class_names))
            },
            'model_info': {
                'version': model_metrics.get('model_type', 'enhanced_cnn'),
                'accuracy': round(model_metrics.get('test_accuracy', 0) * 100, 2)
            }
        }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction
        log_prediction(image_filename, prediction_result, processing_time, True)
        
        # Cleanup uploaded file
        try:
            os.remove(image_path)
        except:
            pass
        
        # Return successful prediction
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'processing_time_ms': round(processing_time * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        # Calculate processing time for failed prediction
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log failed prediction
        if image_filename:
            log_prediction(image_filename, {}, processing_time, False)
        
        # Cleanup uploaded file if it exists
        if image_filename:
            try:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                if os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
        
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'details': str(e),
            'processing_time_ms': round(processing_time * 1000, 2)
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get comprehensive model information"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        model_info_data = {
            'model_loaded': True,
            'model_path': app.config['MODEL_PATH'],
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_parameters': model.count_params(),
            'class_names': class_names,
            'num_classes': len(class_names),
            'metrics': model_metrics,
            'preprocessing_config': preprocessor.config.to_dict() if preprocessor else None,
            'api_version': '2.0.0',
            'tensorflow_version': tf.__version__
        }
        
        return jsonify(model_info_data), 200
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        return jsonify({'error': 'Failed to retrieve model info'}), 500


@app.route('/analytics/stats', methods=['GET'])
def get_analytics():
    """Get API usage analytics"""
    try:
        current_time = datetime.now()
        start_time = datetime.fromisoformat(api_stats['start_time'].replace('Z', '+00:00').replace('+00:00', ''))
        uptime_hours = (current_time - start_time).total_seconds() / 3600
        
        analytics_data = {
            'api_stats': api_stats.copy(),
            'uptime_hours': round(uptime_hours, 2),
            'success_rate': (
                round((api_stats['successful_predictions'] / api_stats['total_predictions']) * 100, 2)
                if api_stats['total_predictions'] > 0 else 0
            ),
            'predictions_per_hour': (
                round(api_stats['total_predictions'] / uptime_hours, 2)
                if uptime_hours > 0 else 0
            ),
            'recent_predictions_count': len(analysis_history),
            'model_performance': {
                'test_accuracy': model_metrics.get('test_accuracy', 0) * 100 if model_metrics else 0,
                'f1_score': model_metrics.get('f1_weighted', 0) if model_metrics else 0
            }
        }
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        return jsonify({'error': 'Failed to retrieve analytics'}), 500


@app.route('/analytics/history/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete a specific analysis from history"""
    try:
        global analysis_history
        
        # Find and remove the analysis by ID
        original_length = len(analysis_history)
        analysis_history = [analysis for analysis in analysis_history if analysis.get('id') != analysis_id]
        
        if len(analysis_history) == original_length:
            return jsonify({'error': 'Analysis not found'}), 404
        
        logger.info(f"Analysis deleted: {analysis_id}")
        return jsonify({
            'message': 'Analysis deleted successfully',
            'deleted_id': analysis_id
        }), 200
        
    except Exception as e:
        logger.error(f"Analysis deletion failed: {e}")
        return jsonify({'error': 'Failed to delete analysis'}), 500


@app.route('/analytics/history', methods=['GET'])
def get_prediction_history():
    """Get recent prediction history"""
    try:
        # Get optional limit parameter
        limit = request.args.get('limit', 50, type=int)
        limit = min(limit, len(analysis_history))  # Don't exceed available data
        
        recent_history = analysis_history[-limit:] if limit > 0 else []
        
        return jsonify({
            'history': recent_history,
            'total_count': len(analysis_history),
            'returned_count': len(recent_history)
        }), 200
        
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        return jsonify({'error': 'Failed to retrieve history'}), 500


# For backward compatibility
@app.route('/stats', methods=['GET'])
def get_stats():
    """Legacy stats endpoint"""
    return get_analytics()


@app.route('/analysis/history', methods=['GET'])
def get_analysis_history():
    """Legacy history endpoint"""
    return get_prediction_history()


# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please try again later'
    }), 500


@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health', '/predict', '/model/info', 
            '/analytics/stats', '/analytics/history'
        ]
    }), 404


# Initialize API on startup
if __name__ == '__main__':
    logger.info("Starting Enhanced BiteMark Classification API...")
    
    if initialize_api():
        logger.info("API ready for requests!")
        logger.info("   Available endpoints:")
        logger.info("   - POST /predict - Make predictions")
        logger.info("   - GET /health - Health check")
        logger.info("   - GET /model/info - Model information")
        logger.info("   - GET /analytics/stats - Usage analytics")
        logger.info("   - GET /analytics/history - Prediction history")
        logger.info("   - DELETE /analytics/history/<id> - Delete analysis")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error("API initialization failed - exiting")
        sys.exit(1)