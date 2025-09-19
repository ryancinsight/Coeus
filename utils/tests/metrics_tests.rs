//! Comprehensive tests for advanced metrics functionality

use coeus_tensor::Tensor;
use coeus_utils::utils::metrics::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_accuracy() {
        // Create simple predictions and targets
        let predictions = Tensor::from_vec(vec![0.1, 0.9, 0.8, 0.2], vec![2, 2]); // 2 samples, 2 classes
        let targets = Tensor::from_vec(vec![1, 0], vec![2]); // Target classes

        let accuracy = top_k_accuracy(&predictions, &targets, 1).unwrap();
        // Should be 1.0 since top-1 prediction is correct for both samples
        assert!((accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = Tensor::from_vec(vec![0, 1, 1, 0], vec![4]);
        let targets = Tensor::from_vec(vec![0, 0, 1, 1], vec![4]);

        let cm = confusion_matrix(&predictions, &targets, 2).unwrap();

        // Check confusion matrix shape
        assert_eq!(cm.shape(), &[2, 2]);

        // Check values: [[1, 1], [1, 1]]
        // Row 0: actual class 0, [TN=1, FP=1]
        // Row 1: actual class 1, [FN=1, TP=1]
        let data = cm.data();
        assert_eq!(data[0], 1); // True negative (pred=0, target=0)
        assert_eq!(data[1], 1); // False positive (pred=1, target=0)
        assert_eq!(data[2], 1); // False negative (pred=0, target=1)
        assert_eq!(data[3], 1); // True positive (pred=1, target=1)
    }

    #[test]
    fn test_classification_report() {
        let predictions = Tensor::from_vec(vec![0, 1, 1, 0], vec![4]);
        let targets = Tensor::from_vec(vec![0, 0, 1, 1], vec![4]);

        let report = classification_report(&predictions, &targets, 2).unwrap();

        // Check that precision, recall, f1 are computed for each class
        assert_eq!(report.precision.len(), 2);
        assert_eq!(report.recall.len(), 2);
        assert_eq!(report.f1_score.len(), 2);

        // Check macro averages
        assert!(report.macro_precision >= 0.0 && report.macro_precision <= 1.0);
        assert!(report.macro_recall >= 0.0 && report.macro_recall <= 1.0);
        assert!(report.macro_f1 >= 0.0 && report.macro_f1 <= 1.0);
    }

    #[test]
    fn test_mean_squared_error() {
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let targets = Tensor::from_vec(vec![1.0, 2.0, 4.0], vec![3]);

        let mse = mean_squared_error(&predictions, &targets).unwrap();

        // MSE should be (0 + 0 + 1)/3 = 1/3 ≈ 0.333
        assert!((mse - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_auc_roc_edge_cases() {
        // Test with no positives
        let predictions = Tensor::from_vec(vec![0.1, 0.2], vec![2]);
        let targets = Tensor::from_vec(vec![0, 0], vec![2]);

        let auc = auc_roc(&predictions, &targets).unwrap();
        assert!((auc - 0.5).abs() < 1e-6); // Should be 0.5 (random classifier)

        // Test with no negatives
        let predictions = Tensor::from_vec(vec![0.8, 0.9], vec![2]);
        let targets = Tensor::from_vec(vec![1, 1], vec![2]);

        let auc = auc_roc(&predictions, &targets).unwrap();
        assert!((auc - 0.5).abs() < 1e-6); // Should be 0.5 (random classifier)
    }

    #[test]
    fn test_metrics_edge_cases() {
        // Test with single sample
        let predictions = Tensor::from_vec(vec![0.9, 0.1], vec![1, 2]);
        let targets = Tensor::from_vec(vec![0], vec![1]);

        let accuracy = top_k_accuracy(&predictions, &targets, 1).unwrap();
        assert_eq!(accuracy, 1.0);

        // Test k > num_classes
        let result = top_k_accuracy(&predictions, &targets, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_confusion_matrix_edge_cases() {
        let predictions = Tensor::from_vec(vec![0], vec![1]);
        let targets = Tensor::from_vec(vec![0], vec![1]);

        let cm = confusion_matrix(&predictions, &targets, 2).unwrap();
        assert_eq!(cm.shape(), &[2, 2]);

        // Test with out-of-range class
        let predictions = Tensor::from_vec(vec![2], vec![1]);
        let targets = Tensor::from_vec(vec![0], vec![1]);

        let result = confusion_matrix(&predictions, &targets, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_with_different_dtypes() {
        // Test with f64 tensors
        let predictions_f64 = Tensor::from_vec(vec![0.1f64, 0.9f64], vec![1, 2]);
        let targets = Tensor::from_vec(vec![1i64], vec![1]);

        let accuracy = top_k_accuracy(&predictions_f64, &targets, 1).unwrap();
        assert_eq!(accuracy, 1.0); // Correct prediction (class 1 has highest probability 0.9)

        let accuracy_correct = top_k_accuracy(&predictions_f64, &targets, 2).unwrap();
        assert_eq!(accuracy_correct, 1.0); // Still correct within top-2
    }
}
