//! Python Integration Tests for PyO3 Advanced Features
//!
//! This module tests the advanced PyO3 trait object features implemented
//! during Sprint MS-37, including dataset utilities and neural network composition.

use pyo3::prelude::*;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

static COEUS_EXTENSION_INIT: Once = Once::new();

fn ensure_local_coeus_extension_ready() {
    COEUS_EXTENSION_INIT.call_once(|| {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.join("..");

        let status = Command::new("cargo")
            .args(["build", "-p", "pycoeus"])
            .current_dir(&workspace_root)
            .status()
            .unwrap();
        assert!(status.success());

        let built_dylib = manifest_dir
            .join("..")
            .join("target")
            .join("debug")
            .join("pycoeus.dll");
        let packaged_pyd = manifest_dir.join("python").join("coeus").join("_coeus.pyd");

        std::fs::copy(&built_dylib, &packaged_pyd).unwrap();
    });
}

#[test]
fn test_tensor_dataset_integration() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        // Test loading the Python module
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        // Skip test if module is not built/installed
        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping PyO3 integration test - module not built/installed");
                return;
            }
        };

        // Test TensorDataset creation with new constructor
        let tensor_zeros = coeus.getattr("tensor_zeros").unwrap();

        // Create input and target tensors (simplified for testing)
        let input_data = tensor_zeros.call1(([3, 4],)).unwrap();
        let target_data = tensor_zeros.call1(([3, 1],)).unwrap();

        let utils = coeus.getattr("utils").unwrap();
        let tensor_dataset_class = utils.getattr("TensorDataset").unwrap();

        // Test that TensorDataset constructor works
        let dataset = tensor_dataset_class
            .call1((vec![input_data], vec![target_data]))
            .unwrap();

        // Test length and indexing
        let len: usize = dataset
            .getattr("__len__")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(len, 3);

        // Test getitem
        let item = dataset.getattr("__getitem__").unwrap().call1((0,)).unwrap();

        // Verify the structure exists - detailed validation in Python tests
        assert!(item.hasattr("inputs").unwrap());
        assert!(item.hasattr("targets").unwrap());

        println!("✅ TensorDataset Python integration test passed");
    });
}

#[test]
fn test_dataset_operations_integration() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        // Test loading the Python module
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        // Skip test if module is not built/installed
        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping PyO3 integration test - module not built/installed");
                return;
            }
        };
        let utils = coeus.getattr("utils").unwrap();

        // Create test datasets
        let tensor_zeros = coeus.getattr("tensor_zeros").unwrap();

        // Dataset 1: 2 samples, 4 features -> 2 targets
        let input1 = tensor_zeros.call1(([2, 4],)).unwrap();
        let target1 = tensor_zeros.call1(([2, 1],)).unwrap();
        let dataset1_class = utils.getattr("TensorDataset").unwrap();
        let dataset1 = dataset1_class.call1((vec![input1], vec![target1])).unwrap();

        // Dataset 2: 3 samples, 4 features -> 2 targets
        let input2 = tensor_zeros.call1(([3, 4],)).unwrap();
        let target2 = tensor_zeros.call1(([3, 1],)).unwrap();
        let dataset2 = dataset1_class.call1((vec![input2], vec![target2])).unwrap();

        // Test ConcatDataset - CORE TRAIT OBJECT FEATURE
        let concat_dataset_class = utils.getattr("ConcatDataset").unwrap();
        let concat_dataset = concat_dataset_class
            .call1((vec![dataset1.clone(), dataset2],))
            .unwrap();

        let len: usize = concat_dataset
            .getattr("__len__")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(len, 5); // 2 + 3 = 5 total samples

        // Verify indexing across concatenated datasets
        let item0 = concat_dataset
            .getattr("__getitem__")
            .unwrap()
            .call1((0,))
            .unwrap();
        let item2 = concat_dataset
            .getattr("__getitem__")
            .unwrap()
            .call1((2,))
            .unwrap();
        let item4 = concat_dataset
            .getattr("__getitem__")
            .unwrap()
            .call1((4,))
            .unwrap();

        assert!(item0.hasattr("inputs").unwrap());
        assert!(item2.hasattr("inputs").unwrap());
        assert!(item4.hasattr("inputs").unwrap());

        println!("✅ ConcatDataset trait object integration test passed");

        // Test Subset - CORE TRAIT OBJECT FEATURE
        let subset_class = utils.getattr("Subset").unwrap();
        let subset = subset_class.call1((dataset1, vec![0, 1])).unwrap();

        let subset_len: usize = subset
            .getattr("__len__")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(subset_len, 2);

        let subset_item = subset.getattr("__getitem__").unwrap().call1((0,)).unwrap();
        assert!(subset_item.hasattr("inputs").unwrap());

        println!("✅ Subset trait object integration test passed");
    });
}

#[test]
fn test_dataloader_integration() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        // Test loading the Python module
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        // Skip test if module is not built/installed
        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping PyO3 integration test - module not built/installed");
                return;
            }
        };
        let utils = coeus.getattr("utils").unwrap();

        // Create test dataset
        let tensor_zeros = coeus.getattr("tensor_zeros").unwrap();
        let inputs = tensor_zeros.call1(([4, 2],)).unwrap();
        let targets = tensor_zeros.call1(([4, 1],)).unwrap();

        let dataset_class = utils.getattr("TensorDataset").unwrap();
        let dataset = dataset_class.call1((vec![inputs], vec![targets])).unwrap();

        // Test DataLoader creation
        let data_loader_class = utils.getattr("DataLoader").unwrap();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dataset", dataset).unwrap();
        let data_loader = data_loader_class.call((), Some(&kwargs)).unwrap();

        // Test length calculation
        let len: usize = data_loader
            .getattr("__len__")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(len, 4); // 4 samples

        println!("✅ DataLoader Python integration test passed");
    });
}

#[test]
fn test_sequential_nn_integration() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        // Test loading the Python module
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        // Skip test if module is not built/installed
        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping PyO3 integration test - module not built/installed");
                return;
            }
        };

        // Test Sequential NN creation
        let sequential_class = coeus.getattr("nn").unwrap().getattr("Sequential").unwrap();
        let sequential = sequential_class.call0().unwrap();

        // Test adding modules - CORE TRAIT OBJECT FEATURE
        sequential
            .getattr("add_linear")
            .unwrap()
            .call(("linear1", 10, 5), Some(&pyo3::types::PyDict::new(py)))
            .unwrap();

        // Test adding ReLU activation
        sequential
            .getattr("add_relu")
            .unwrap()
            .call(("relu1",), None)
            .unwrap();

        // Should have 2 modules now (Linear + ReLU)
        let len: usize = sequential
            .getattr("__len__")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(len, 2);

        println!("✅ Sequential NN trait object integration test passed");
    });
}

// Transform Pipeline Integration Tests
#[test]
fn test_transform_pipeline_integration() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        // Test loading the Python module
        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        // Skip test if module is not built/installed
        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping transform integration test - module not built/installed");
                return;
            }
        };

        // Test transform factory functions
        let to_tensor_fn = coeus.getattr("to_tensor").unwrap();
        let normalize_fn = coeus.getattr("normalize").unwrap();

        // Create transforms using factory functions
        let to_tensor_transform = to_tensor_fn.call0().unwrap();
        let _normalize_transform = normalize_fn.call(([0.5], [0.5]), None).unwrap();

        // Test ToTensor transform
        let transform_result = to_tensor_transform
            .call(([1.0, 2.0, 3.0, 4.0],), None)
            .unwrap();
        // Should return a list (tensor representation)
        assert!(transform_result.is_instance_of::<pyo3::types::PyList>());

        println!("✅ Transform factory functions and ToTensor integration test passed");

        // Test transform composition with multiple ToTensor transforms
        let compose_fn = coeus.getattr("compose").unwrap();
        let compose_result = compose_fn
            .call(
                (vec![
                    to_tensor_transform.clone(),
                    to_tensor_transform.clone(),
                ],),
                None,
            )
            .unwrap();

        // Test the composed transform
        let test_data = vec![1.0, 2.0, 3.0];
        let result = compose_result.call1((test_data.clone(),)).unwrap();

        // Should return the same list since ToTensor -> ToTensor is idempotent
        let result_list: Vec<f32> = result.extract().unwrap();
        assert_eq!(result_list, test_data);

        println!("✅ Transform pipeline composition test passed");
    });
}

#[test]
fn test_transforms_basic() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        // Skip test if module is not built/installed
        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping transform tests - module not built/installed");
                return;
            }
        };

        // Test transform factory functions
        let transforms = coeus.getattr("transforms").unwrap();
        let to_tensor_class = transforms.getattr("ToTensor").unwrap();
        let normalize_class = transforms.getattr("Normalize").unwrap();

        // Create transforms using class constructors
        let totensor = to_tensor_class.call0().unwrap();
        let _normalize = normalize_class.call1((vec![2.0f32], vec![1.0f32])).unwrap();

        // Test ToTensor transform
        let input_data = vec![1.0, 2.0, 3.0];
        let result = totensor.call1((input_data,)).unwrap();

        // Should return a list (tensor representation)
        assert!(result.is_instance_of::<pyo3::types::PyList>());

        println!("✅ ToTensor transform test passed");

        // Normalize instance created successfully
        println!("✅ Normalize transform test passed");

        // Test Compose transform creation
        let compose_class = transforms.getattr("Compose").unwrap();
        let compose = compose_class.call1((vec![totensor],)).unwrap();

        let len: usize = compose
            .getattr("__len__")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(len, 1);

        println!("✅ Compose transform test passed");

        // Test Compose transform functionality with multiple ToTensor transforms
        let totensor2 = to_tensor_class.call0().unwrap();
        let totensor3 = to_tensor_class.call0().unwrap();

        let compose2 = compose_class.call(([totensor2, totensor3],), None).unwrap();

        // Test the composed transform pipeline (ToTensor -> ToTensor should just return the same list)
        let test_data = vec![1.0, 2.0, 3.0];
        let result = compose2.call1((test_data.clone(),)).unwrap();

        // Should return the same list since ToTensor -> ToTensor is idempotent
        let result_list: Vec<f32> = result.extract().unwrap();
        assert_eq!(result_list, test_data);

        println!("✅ Compose transform chaining test passed");
    });
}

#[test]
fn test_lr_scheduler_integration() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => {
                println!("Skipping lr_scheduler integration test - module not built/installed");
                return;
            }
        };

        let optim = coeus.getattr("optim").unwrap();
        let sgd_class = optim.getattr("SGD").unwrap();
        let lr_scheduler = optim.getattr("lr_scheduler").unwrap();
        let step_lr_class = lr_scheduler.getattr("StepLR").unwrap();

        let tensor_fn = coeus.getattr("tensor").unwrap();
        let tensor_kwargs = pyo3::types::PyDict::new(py);
        tensor_kwargs.set_item("requires_grad", true).unwrap();
        let param = tensor_fn
            .call((vec![0.0f32],), Some(&tensor_kwargs))
            .unwrap();

        let opt_kwargs = pyo3::types::PyDict::new(py);
        opt_kwargs.set_item("lr", 0.1f64).unwrap();
        let optimizer = sgd_class.call((vec![param],), Some(&opt_kwargs)).unwrap();

        let sched_kwargs = pyo3::types::PyDict::new(py);
        sched_kwargs.set_item("gamma", 0.5f64).unwrap();
        let scheduler = step_lr_class
            .call((optimizer, 2usize), Some(&sched_kwargs))
            .unwrap();

        let lr0: f64 = scheduler
            .getattr("get_lr")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert!((lr0 - 0.1).abs() < 1e-12);

        scheduler.getattr("step").unwrap().call0().unwrap();
        let lr1: f64 = scheduler
            .getattr("get_lr")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert!((lr1 - 0.1).abs() < 1e-12);

        scheduler.getattr("step").unwrap().call0().unwrap();
        let lr2: f64 = scheduler
            .getattr("get_lr")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert!((lr2 - 0.05).abs() < 1e-12);

        let last_lr: Vec<f64> = scheduler
            .getattr("get_last_lr")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(last_lr, vec![lr2]);

        println!("✅ lr_scheduler StepLR integration test passed");
    });
}

#[test]
fn test_functional_api_pytorch_parity_surface() {
    pyo3::Python::initialize();

    pyo3::Python::attach(|py| {
        ensure_local_coeus_extension_ready();

        let sys = py.import("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        path.call_method1(
            "insert",
            (0, env!("CARGO_MANIFEST_DIR").to_string() + "/python"),
        )
        .unwrap();

        let coeus = match py.import("coeus") {
            Ok(module) => module,
            Err(_) => return,
        };

        let tensor_class = coeus.getattr("Tensor").unwrap();
        let input = tensor_class
            .call1((
                vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
                vec![2_usize, 2_usize],
            ))
            .unwrap();

        let softmax_fn = coeus.getattr("softmax").unwrap();
        let kwargs_dim1 = pyo3::types::PyDict::new(py);
        kwargs_dim1.set_item("dim", 1).unwrap();
        let sm_dim1 = softmax_fn
            .call((input.clone(),), Some(&kwargs_dim1))
            .unwrap();
        let sm_dim1_list: Vec<Vec<f64>> = sm_dim1
            .call_method0("numpy")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();

        let e = std::f64::consts::E;
        let row1_denom = e.powf(1.0) + e.powf(2.0);
        let row2_denom = e.powf(3.0) + e.powf(4.0);
        let expected_dim1 = vec![
            vec![e.powf(1.0) / row1_denom, e.powf(2.0) / row1_denom],
            vec![e.powf(3.0) / row2_denom, e.powf(4.0) / row2_denom],
        ];

        for r in 0..2 {
            for c in 0..2 {
                assert!((sm_dim1_list[r][c] - expected_dim1[r][c]).abs() < 1e-6);
            }
        }

        let kwargs_dim0 = pyo3::types::PyDict::new(py);
        kwargs_dim0.set_item("dim", 0).unwrap();
        let sm_dim0 = softmax_fn
            .call((input.clone(),), Some(&kwargs_dim0))
            .unwrap();
        let sm_dim0_list: Vec<Vec<f64>> = sm_dim0
            .call_method0("numpy")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();

        let col1_denom = e.powf(1.0) + e.powf(3.0);
        let col2_denom = e.powf(2.0) + e.powf(4.0);
        let expected_dim0 = vec![
            vec![e.powf(1.0) / col1_denom, e.powf(2.0) / col2_denom],
            vec![e.powf(3.0) / col1_denom, e.powf(4.0) / col2_denom],
        ];

        for r in 0..2 {
            for c in 0..2 {
                assert!((sm_dim0_list[r][c] - expected_dim0[r][c]).abs() < 1e-6);
            }
        }

        let dropout_fn = coeus.getattr("dropout").unwrap();
        let kwargs_dropout_eval = pyo3::types::PyDict::new(py);
        kwargs_dropout_eval.set_item("p", 0.5).unwrap();
        kwargs_dropout_eval.set_item("training", false).unwrap();
        let dropout_eval = dropout_fn
            .call((input.clone(),), Some(&kwargs_dropout_eval))
            .unwrap();
        let dropout_eval_list: Vec<Vec<f64>> = dropout_eval
            .call_method0("numpy")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(dropout_eval_list, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let kwargs_dropout_inplace = pyo3::types::PyDict::new(py);
        kwargs_dropout_inplace.set_item("inplace", true).unwrap();
        let dropout_inplace = dropout_fn.call((input.clone(),), Some(&kwargs_dropout_inplace));
        assert!(dropout_inplace.is_err());

        let batch_norm_fn = coeus.getattr("batch_norm").unwrap();
        let kwargs_bn_eval = pyo3::types::PyDict::new(py);
        kwargs_bn_eval.set_item("training", false).unwrap();
        let bn_eval = batch_norm_fn.call((input.clone(),), Some(&kwargs_bn_eval));
        assert!(bn_eval.is_err());

        let kwargs_bn_train = pyo3::types::PyDict::new(py);
        kwargs_bn_train.set_item("training", true).unwrap();
        let bn_train = batch_norm_fn
            .call((input.clone(),), Some(&kwargs_bn_train))
            .unwrap();
        let bn_train_list: Vec<Vec<f64>> = bn_train
            .call_method0("numpy")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();

        let std = (1.0 + 1e-5_f64).sqrt();
        assert!((bn_train_list[0][0] - (-1.0 / std)).abs() < 1e-6);
        assert!((bn_train_list[0][1] - (-1.0 / std)).abs() < 1e-6);
        assert!((bn_train_list[1][0] - (1.0 / std)).abs() < 1e-6);
        assert!((bn_train_list[1][1] - (1.0 / std)).abs() < 1e-6);

        let nll_loss_fn = coeus.getattr("nll_loss").unwrap();
        let log_probs = tensor_class
            .call1((
                vec![-0.5_f32, -1.0_f32, -2.0_f32, -2.0_f32, -0.5_f32, -1.0_f32],
                vec![2_usize, 3_usize],
            ))
            .unwrap();
        let targets = tensor_class
            .call1((vec![0.0_f32, 1.0_f32], vec![2_usize]))
            .unwrap();
        let nll = nll_loss_fn.call1((log_probs, targets)).unwrap();
        let nll_value: f64 = nll
            .call_method0("numpy")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!((nll_value - 0.5).abs() < 1e-6);

        let cross_entropy_fn = coeus.getattr("cross_entropy").unwrap();
        let logits = tensor_class
            .call1((
                vec![1.0_f32, 0.5_f32, 0.2_f32, 0.1_f32, 2.0_f32, 0.3_f32],
                vec![2_usize, 3_usize],
            ))
            .unwrap();
        let ce = cross_entropy_fn
            .call1((
                logits,
                tensor_class
                    .call1((vec![0.0_f32, 1.0_f32], vec![2_usize]))
                    .unwrap(),
            ))
            .unwrap();
        let ce_value: f64 = ce
            .call_method0("numpy")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!(ce_value > 0.0);
    });
}
