//! Bilinear layer implementation

use crate::core::module::Module;
use crate::core::parameter::Parameter;
use crate::core::error::Result;
use backend::Backend;
use dtype::{DataType, traits::FloatExt};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{Tensor, ops::TensorStorageOps};

/// Bilinear transformation layer
#[derive(Debug, Clone)]
pub struct Bilinear<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub weight: Parameter<B, S, T>,
    pub bias: Option<Parameter<B, S, T>>,
    pub in1_features: usize,
    pub in2_features: usize,
    pub out_features: usize,
}

impl<B, S, T> Bilinear<B, S, T>
where
    B: Backend<Data = T> + Default + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + TensorStorageOps<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + Clone + 'static,
{
    /// Create a new bilinear layer
    pub fn new(
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> Result<Self> {
        // Initialize weights
        let weight_tensor = Tensor::rand(&[out_features, in1_features, in2_features]).map_err(crate::core::error::NNError::from)?;
        let weight = Parameter::new(weight_tensor, "weight".to_string());
        
        let bias_param = if use_bias {
            let bias_tensor = Tensor::zeros(&[out_features]).map_err(crate::core::error::NNError::from)?;
            Some(Parameter::new(bias_tensor, "bias".to_string()))
        } else {
            None
        };
        
        Ok(Self {
            weight,
            bias: bias_param,
            in1_features,
            in2_features,
            out_features,
        })
    }
    
    /// Forward pass with two inputs
    pub fn forward_bilinear(
        &self, 
        input1: &Tensor<B, S, T>, 
        input2: &Tensor<B, S, T>
    ) -> Result<Tensor<B, S, T>> {
        let i1_dense = input1.to_dense_generic().map_err(crate::core::error::NNError::from)?;
        let i2_dense = input2.to_dense_generic().map_err(crate::core::error::NNError::from)?;
        
        // Use .data() to access the tensor from Parameter
        let w_dense = self.weight.data().to_dense_generic().map_err(crate::core::error::NNError::from)?;
        
        let b_dense = if let Some(b) = &self.bias {
            Some(b.data().to_dense_generic().map_err(crate::core::error::NNError::from)?)
        } else {
            None
        };
        
        let result = tensor::ops::linalg::bilinear(&i1_dense, &i2_dense, &w_dense, b_dense.as_ref())
            .map_err(|e| crate::core::error::NNError::ExecutionError { message: e.to_string() })?;
            
        let dense_result = result.to_dense_generic().map_err(crate::core::error::NNError::from)?;
        let shape = dense_result.shape().clone();
        let data = dense_result.as_slice().to_vec();
        
        let storage = S::from_vec(data, shape.dims()).map_err(crate::core::error::NNError::from)?;
        Ok(Tensor::from_storage(storage, input1.backend().clone()))
    }
}

impl<B, S, T> Module<B, S, T> for Bilinear<B, S, T>
where
    B: Backend<Data = T> + Default + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + TensorStorageOps<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + Clone + 'static,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, _input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Bilinear requires two inputs
        Err(crate::core::error::NNError::InvalidInput {
            message: "Bilinear layer requires two inputs. Use forward_bilinear instead.".to_string()
        })
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn name(&self) -> &str {
        "Bilinear"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
