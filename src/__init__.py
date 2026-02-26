# E-commerce Text Classification
# Вспомогательные модули для проекта классификации текстов электронной коммерции.

from src.data_loader import load_data, preprocess_data, split_data
from src.text_transformers import TextCleaner, TextLengthExtractor, CombinedFeatures
from src.model_utils import (
    evaluate_model_cv,
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_cv_scores,
    get_feature_importance
)
from src.visualization import (
    plot_target_distribution,
    plot_text_length_distribution,
    plot_text_length_stats,
    plot_top_words,
    plot_word_cloud,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_cv_scores,
    plot_feature_importance,
    plot_metrics_history
)
from src.embeddings import SentenceTransformerVectorizer, EmbeddingVectorizer
from src.feature_selection import FeatureSelector, select_features_chi2, select_features_mutual_info
from src.models import get_model, get_ensemble_model, get_default_params

__all__ = [
    # Data
    'load_data',
    'preprocess_data',
    'split_data',
    
    # Text transformers
    'TextCleaner',
    'TextLengthExtractor',
    'CombinedFeatures',
    
    # Model utils
    'evaluate_model_cv',
    'evaluate_model',
    'print_classification_report',
    'plot_confusion_matrix',
    'plot_cv_scores',
    'get_feature_importance',
    
    # Visualization
    'plot_target_distribution',
    'plot_text_length_distribution',
    'plot_text_length_stats',
    'plot_top_words',
    'plot_word_cloud',
    'plot_confusion_matrix',
    'plot_model_comparison',
    'plot_cv_scores',
    'plot_feature_importance',
    'plot_metrics_history',
    
    # Embeddings
    'SentenceTransformerVectorizer',
    'EmbeddingVectorizer',
    
    # Feature selection
    'FeatureSelector',
    'select_features_chi2',
    'select_features_mutual_info',
    
    # Models
    'get_model',
    'get_ensemble_model',
    'get_default_params',
]
