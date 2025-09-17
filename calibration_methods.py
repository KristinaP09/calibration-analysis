# Enhanced experimental framework with statistical testing and core calibration implementations
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ttest_rel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_openml
import warnings
warnings.filterwarnings('ignore')

class PlattScaling:
    """Core implementation of Platt scaling calibration method."""
    
    def __init__(self):
        self.A = None
        self.B = None
        
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability."""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _platt_objective(self, params, scores, targets):
        """Objective function for Platt scaling optimization."""
        A, B = params
        z = A * scores + B
        probs = self._sigmoid(z)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        
        # Negative log-likelihood
        return -np.sum(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    
    def fit(self, scores, targets):
        """
        Fit Platt scaling parameters.
        
        Parameters:
        -----------
        scores : array-like of shape (n_samples,)
            Raw classifier scores (decision function output)
        targets : array-like of shape (n_samples,)
            Binary target labels (0 or 1)
        """
        scores = np.array(scores).reshape(-1)
        targets = np.array(targets).reshape(-1)
        
        # Initial parameter estimates
        prior1 = np.sum(targets)
        prior0 = len(targets) - prior1
        
        # Initialize A and B
        A_init = 0.0
        B_init = np.log(prior0 / prior1) if prior1 > 0 and prior0 > 0 else 0.0
        
        # Optimize parameters using BFGS
        result = minimize(
            self._platt_objective,
            x0=[A_init, B_init],
            args=(scores, targets),
            method='BFGS'
        )
        
        self.A, self.B = result.x
        return self
    
    def predict_proba(self, scores):
        """
        Apply Platt scaling to convert scores to probabilities.
        
        Parameters:
        -----------
        scores : array-like of shape (n_samples,)
            Raw classifier scores
            
        Returns:
        --------
        probabilities : array of shape (n_samples,)
            Calibrated probabilities
        """
        if self.A is None or self.B is None:
            raise ValueError("Model must be fitted before making predictions")
            
        scores = np.array(scores).reshape(-1)
        z = self.A * scores + self.B
        return self._sigmoid(z)

class IsotonicCalibration:
    """Core implementation of isotonic regression calibration method."""
    
    def __init__(self):
        self.isotonic_regressor = None
        self.min_score = None
        self.max_score = None
        
    def fit(self, scores, targets):
        """
        Fit isotonic regression calibration.
        
        Parameters:
        -----------
        scores : array-like of shape (n_samples,)
            Raw classifier scores or probabilities
        targets : array-like of shape (n_samples,)
            Binary target labels (0 or 1)
        """
        scores = np.array(scores).reshape(-1)
        targets = np.array(targets).reshape(-1)
        
        # Store score range for extrapolation
        self.min_score = np.min(scores)
        self.max_score = np.max(scores)
        
        # Fit isotonic regression
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(scores, targets)
        
        return self
    
    def predict_proba(self, scores):
        """
        Apply isotonic regression to convert scores to probabilities.
        
        Parameters:
        -----------
        scores : array-like of shape (n_samples,)
            Raw classifier scores
            
        Returns:
        --------
        probabilities : array of shape (n_samples,)
            Calibrated probabilities
        """
        if self.isotonic_regressor is None:
            raise ValueError("Model must be fitted before making predictions")
            
        scores = np.array(scores).reshape(-1)
        return self.isotonic_regressor.predict(scores)

class CalibrationEvaluator:
    """Comprehensive calibration evaluation framework with core implementations."""
    
    def __init__(self, n_splits=5, n_repeats=10, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    def expected_calibration_error(self, y_true, y_prob, n_bins=10):
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def maximum_calibration_error(self, y_true, y_prob, n_bins=10):
        """Compute Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
        return mce
    
    def reliability_diagram(self, y_true, y_prob, n_bins=10, ax=None):
        """Create reliability diagram for calibration visualization."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_centers.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(prop_in_bin)
        
        # Plot reliability diagram
        bin_centers = np.array(bin_centers)
        bin_accuracies = np.array(bin_accuracies)
        bin_counts = np.array(bin_counts)
        
        # Scatter plot with size proportional to bin count
        ax.scatter(bin_centers, bin_accuracies, s=bin_counts*5, alpha=0.7, 
                  c='blue', edgecolors='black')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def evaluate_methods(self, X, y, base_estimator, use_core_implementations=True):
        """
        Evaluate calibration methods with option to use core implementations.
        
        Parameters:
        -----------
        use_core_implementations : bool
            If True, use our custom Platt scaling and isotonic regression
        """
        results = {
            'base': {'brier': [], 'ece': [], 'mce': [], 'logloss': []},
            'platt': {'brier': [], 'ece': [], 'mce': [], 'logloss': []},
            'isotonic': {'brier': [], 'ece': [], 'mce': [], 'logloss': []}
        }
        
        for repeat in range(self.n_repeats):
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                 random_state=self.random_state + repeat)
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Split training data for calibration
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=self.random_state,
                    stratify=y_train
                )
                
                # Train base classifier
                base_clf = clone(base_estimator)
                base_clf.fit(X_fit, y_fit)
                
                # Get base predictions
                if hasattr(base_clf, 'predict_proba'):
                    base_probs = base_clf.predict_proba(X_test)[:, 1]
                    cal_scores = base_clf.predict_proba(X_cal)[:, 1]
                elif hasattr(base_clf, 'decision_function'):
                    base_scores = base_clf.decision_function(X_test)
                    cal_scores = base_clf.decision_function(X_cal)
                    base_probs = self._sigmoid(base_scores)
                else:
                    raise ValueError("Classifier must have predict_proba or decision_function")
                
                if use_core_implementations:
                    # Use our custom implementations
                    platt_calibrator = PlattScaling()
                    isotonic_calibrator = IsotonicCalibration()
                    
                    platt_calibrator.fit(cal_scores, y_cal)
                    isotonic_calibrator.fit(cal_scores, y_cal)
                    
                    if hasattr(base_clf, 'predict_proba'):
                        test_scores = base_clf.predict_proba(X_test)[:, 1]
                    else:
                        test_scores = base_clf.decision_function(X_test)
                    
                    platt_probs = platt_calibrator.predict_proba(test_scores)
                    isotonic_probs = isotonic_calibrator.predict_proba(test_scores)
                    
                else:
                    # Use scikit-learn implementations
                    platt_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv='prefit')
                    isotonic_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit')
                    
                    platt_clf.fit(X_cal, y_cal)
                    isotonic_clf.fit(X_cal, y_cal)
                    
                    platt_probs = platt_clf.predict_proba(X_test)[:, 1]
                    isotonic_probs = isotonic_clf.predict_proba(X_test)[:, 1]
                
                # Compute metrics for all methods
                for method, probs in [('base', base_probs), ('platt', platt_probs), 
                                    ('isotonic', isotonic_probs)]:
                    results[method]['brier'].append(brier_score_loss(y_test, probs))
                    results[method]['ece'].append(self.expected_calibration_error(y_test, probs))
                    results[method]['mce'].append(self.maximum_calibration_error(y_test, probs))
                    results[method]['logloss'].append(log_loss(y_test, probs, eps=1e-15))
        
        return results
    
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability."""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def statistical_test(self, results):
        """Perform statistical significance testing with enhanced metrics."""
        stats_results = {}
        
        for metric in ['brier', 'ece', 'mce', 'logloss']:
            base_scores = np.array(results['base'][metric])
            platt_scores = np.array(results['platt'][metric])
            isotonic_scores = np.array(results['isotonic'][metric])
            
            # Paired t-tests
            platt_vs_base = ttest_rel(base_scores, platt_scores)
            isotonic_vs_base = ttest_rel(base_scores, isotonic_scores)
            platt_vs_isotonic = ttest_rel(platt_scores, isotonic_scores)
            
            # Effect sizes (Cohen's d)
            platt_effect = (base_scores.mean() - platt_scores.mean()) / np.sqrt(
                ((base_scores.std()**2 + platt_scores.std()**2) / 2))
            isotonic_effect = (base_scores.mean() - isotonic_scores.mean()) / np.sqrt(
                ((base_scores.std()**2 + isotonic_scores.std()**2) / 2))
            
            # Confidence intervals
            platt_ci = np.percentile(platt_scores, [2.5, 97.5])
            isotonic_ci = np.percentile(isotonic_scores, [2.5, 97.5])
            base_ci = np.percentile(base_scores, [2.5, 97.5])
            
            stats_results[metric] = {
                'means': {
                    'base': base_scores.mean(),
                    'platt': platt_scores.mean(),
                    'isotonic': isotonic_scores.mean()
                },
                'confidence_intervals': {
                    'base': base_ci,
                    'platt': platt_ci,
                    'isotonic': isotonic_ci
                },
                'platt_vs_base': {'statistic': platt_vs_base.statistic, 
                                 'p_value': platt_vs_base.pvalue,
                                 'effect_size': platt_effect},
                'isotonic_vs_base': {'statistic': isotonic_vs_base.statistic, 
                                   'p_value': isotonic_vs_base.pvalue,
                                   'effect_size': isotonic_effect},
                'platt_vs_isotonic': {'statistic': platt_vs_isotonic.statistic, 
                                    'p_value': platt_vs_isotonic.pvalue}
            }
        
        return stats_results

# Example usage demonstrating core implementations
if __name__ == "__main__":
    np.random.seed(42)
    X_synthetic = np.random.rand(1000, 10)
    y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 1).astype(int)

    evaluator = CalibrationEvaluator()
    base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Evaluating with Core Implementations:")
    print("=" * 50)
    results_core = evaluator.evaluate_methods(X_synthetic, y_synthetic, base_estimator, 
                                             use_core_implementations=True)
    stats_core = evaluator.statistical_test(results_core)

    print("\nCalibration Evaluation Results (Core Implementations):")
    print("=" * 60)
    for metric in ['brier', 'ece', 'mce', 'logloss']:
        print(f"\n{metric.upper()} Results:")
        for method in ['base', 'platt', 'isotonic']:
            scores = results_core[method][metric]
            ci = stats_core[metric]['confidence_intervals'][method]
            print(f"{method:>10}: {np.mean(scores):.6f} ± {np.std(scores):.6f} "
                  f"[CI: {ci[0]:.6f}, {ci[1]:.6f}]")
        
        print("Statistical Tests:")
        for comparison, result in stats_core[metric].items():
            if comparison in ['platt_vs_base', 'isotonic_vs_base', 'platt_vs_isotonic']:
                significance = "***" if result['p_value'] < 0.001 else \
                             "**" if result['p_value'] < 0.01 else \
                             "*" if result['p_value'] < 0.05 else ""
                print(f"  {comparison}: p={result['p_value']:.6f}{significance}, "
                      f"effect={result.get('effect_size', 'N/A'):.3f}")
    
    # Demonstrate individual calibrators
    print("\n" + "="*60)
    print("Demonstrating Individual Calibrators:")
    print("="*60)
    
    # Create some sample data for demonstration
    X_train, X_test, y_train, y_test = train_test_split(
        X_synthetic, y_synthetic, test_size=0.3, random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get raw scores
    raw_probs = clf.predict_proba(X_test)[:, 1]
    
    # Apply our custom calibrators
    platt = PlattScaling()
    isotonic = IsotonicCalibration()
    
    platt.fit(raw_probs, y_test)
    isotonic.fit(raw_probs, y_test)
    
    platt_probs = platt.predict_proba(raw_probs)
    isotonic_probs = isotonic.predict_proba(raw_probs)
    
    print(f"Raw ECE: {evaluator.expected_calibration_error(y_test, raw_probs):.6f}")
    print(f"Platt ECE: {evaluator.expected_calibration_error(y_test, platt_probs):.6f}")
    print(f"Isotonic ECE: {evaluator.expected_calibration_error(y_test, isotonic_probs):.6f}")