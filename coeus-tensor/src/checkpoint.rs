// ── StateDict checkpointing ──
//
// Fast binary serialization and deserialization of model parameters.
// Uses bytemuck for zero-copy byte transmutation on CPU-addressable storage.

use crate::Tensor;
use coeus_core::{ComputeBackend, Scalar, Storage};
use std::collections::HashMap;
use std::io::{Error, ErrorKind, Read, Result, Write};

/// A dictionary mapping parameter names to their weight/bias tensors.
#[derive(Clone)]
pub struct StateDict<T: Scalar, B: ComputeBackend + Default> {
    pub tensors: HashMap<String, Tensor<T, B>>,
}

impl<T: Scalar, B: ComputeBackend + Default> Default for StateDict<T, B> {
    #[inline]
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }
}

impl<T: Scalar, B: ComputeBackend + Default> StateDict<T, B> {
    /// Create a new empty StateDict.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a tensor into the state dictionary.
    #[inline]
    pub fn insert<S: Into<String>>(&mut self, name: S, tensor: Tensor<T, B>) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Get a reference to a tensor by name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&Tensor<T, B>> {
        self.tensors.get(name)
    }

    /// Get the number of tensors in the dictionary.
    #[inline]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the dictionary is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Serialize the state dictionary to a writer using an optimized binary format.
    ///
    /// Non-contiguous tensors are compacted to contiguous layout before writing.
    /// Uses zero-copy `bytemuck` casts on CPU-addressable buffers.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        // 1. Magic header
        writer.write_all(b"COEU")?;
        // 2. Version
        writer.write_all(&1u32.to_le_bytes())?;
        // 3. Number of tensors
        let num_tensors = self.tensors.len() as u32;
        writer.write_all(&num_tensors.to_le_bytes())?;

        let backend = B::default();

        for (name, tensor) in &self.tensors {
            // Coerce to contiguous for storage format simplicity
            let cont_tensor = tensor.to_contiguous_on(&backend);

            // Key name bytes
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len() as u32;
            writer.write_all(&name_len.to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // Shape metadata
            let shape = cont_tensor.shape();
            let ndim = shape.len() as u32;
            writer.write_all(&ndim.to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Raw data byte length
            let numel = cont_tensor.numel();
            let expected_bytes = numel * std::mem::size_of::<T>();
            writer.write_all(&(expected_bytes as u64).to_le_bytes())?;

            // Zero-copy save if CPU-addressable
            if let Some(slice) = cont_tensor.storage().try_as_slice() {
                let offset = cont_tensor.layout().offset();
                let byte_slice = bytemuck::cast_slice::<T, u8>(&slice[offset..offset + numel]);
                writer.write_all(byte_slice)?;
            } else {
                // Device to host transfer fallback
                let mut host_buf = vec![T::zero(); numel];
                backend.copy_to_host(cont_tensor.storage(), &mut host_buf);
                let byte_slice = bytemuck::cast_slice::<T, u8>(&host_buf);
                writer.write_all(byte_slice)?;
            }
        }
        Ok(())
    }

    /// Deserialize a state dictionary from a reader.
    ///
    /// Performs zero-copy conversions using `bytemuck` casts, falling back to aligned copies
    /// if the source buffer does not satisfy the target type's alignment requirements.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        let backend = B::default();

        // 1. Read magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"COEU" {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid magic header"));
        }

        // 2. Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Unsupported version: {version}"),
            ));
        }

        // 3. Read number of tensors
        let mut num_tensors_bytes = [0u8; 4];
        reader.read_exact(&mut num_tensors_bytes)?;
        let num_tensors = u32::from_le_bytes(num_tensors_bytes) as usize;

        let mut tensors = HashMap::with_capacity(num_tensors);

        for _ in 0..num_tensors {
            // Key name
            let mut name_len_bytes = [0u8; 4];
            reader.read_exact(&mut name_len_bytes)?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;
            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes)?;
            let name =
                String::from_utf8(name_bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

            // Shape
            let mut ndim_bytes = [0u8; 4];
            reader.read_exact(&mut ndim_bytes)?;
            let ndim = u32::from_le_bytes(ndim_bytes) as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                let mut dim_bytes = [0u8; 8];
                reader.read_exact(&mut dim_bytes)?;
                shape.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            // Data byte length
            let mut data_len_bytes_arr = [0u8; 8];
            reader.read_exact(&mut data_len_bytes_arr)?;
            let data_len_bytes = u64::from_le_bytes(data_len_bytes_arr) as usize;

            let numel: usize = shape.iter().product();
            let expected_bytes = numel * std::mem::size_of::<T>();
            if data_len_bytes != expected_bytes {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Data size mismatch: expected {expected_bytes} bytes, got {data_len_bytes}"
                    ),
                ));
            }

            // Read raw bytes
            let mut raw_bytes = vec![0u8; data_len_bytes];
            reader.read_exact(&mut raw_bytes)?;

            // Try zero-copy cast, falling back to aligned copy if alignment fails
            let tensor = match bytemuck::try_cast_slice::<u8, T>(&raw_bytes) {
                Ok(data_slice) => Tensor::from_slice_on(shape, data_slice, &backend),
                Err(_) => {
                    let mut aligned_buf = vec![T::zero(); numel];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            raw_bytes.as_ptr(),
                            aligned_buf.as_mut_ptr() as *mut u8,
                            expected_bytes,
                        );
                    }
                    Tensor::from_slice_on(shape, &aligned_buf, &backend)
                }
            };
            tensors.insert(name, tensor);
        }

        Ok(Self { tensors })
    }
}
