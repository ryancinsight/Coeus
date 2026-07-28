//! Bounded, validated rkyv archives for named tensor state.

use crate::Tensor;
use coeus_core::{ComputeBackend, Scalar, Storage};
use rkyv::{rancor::Error as ArchiveError, Archive, Deserialize, Serialize};
use std::any::type_name;
use std::collections::{HashMap, HashSet};
use std::io::{Error, ErrorKind, Read, Result, Write};

/// Resource bounds applied while saving, validating, and materializing state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateLimits {
    /// Maximum complete archive size.
    pub archive_bytes: usize,
    /// Maximum number of tensors.
    pub tensors: usize,
    /// Maximum UTF-8 bytes in one parameter name.
    pub name_bytes: usize,
    /// Maximum tensor rank.
    pub rank: usize,
    /// Maximum bytes in one tensor payload.
    pub tensor_bytes: usize,
    /// Maximum aggregate tensor payload bytes.
    pub total_tensor_bytes: usize,
}

impl Default for StateLimits {
    fn default() -> Self {
        Self {
            archive_bytes: 1 << 30,
            tensors: 1_000_000,
            name_bytes: 65_536,
            rank: 64,
            tensor_bytes: 1 << 30,
            total_tensor_bytes: 1 << 30,
        }
    }
}

#[derive(Archive, Deserialize, Serialize)]
struct StateArchiveData {
    magic: [u8; 4],
    version: u32,
    scalar: String,
    endianness: u8,
    tensors: Vec<TensorArchiveData>,
}

#[derive(Archive, Deserialize, Serialize)]
struct TensorArchiveData {
    name: String,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// Validated zero-copy view over an archived state dictionary.
pub struct StateArchive<'a> {
    root: &'a ArchivedStateArchiveData,
    limits: StateLimits,
}

impl StateArchive<'_> {
    /// Number of archived tensors without materializing them.
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.tensors.len()
    }

    /// Whether the archive contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root.tensors.is_empty()
    }

    /// Rust scalar type identity recorded by the archive.
    #[must_use]
    pub fn scalar_type(&self) -> &str {
        self.root.scalar.as_str()
    }

    /// Borrow one archived tensor's shape and raw scalar bytes by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<ArchivedTensor<'_>> {
        self.root
            .tensors
            .iter()
            .find(|tensor| tensor.name.as_str() == name)
            .map(|tensor| ArchivedTensor { tensor })
    }

    /// Materialize validated tensors on `B::default()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the archive scalar identity differs from `T`, a
    /// payload byte count does not match `T`, or a tensor violates the
    /// configured resource and shape bounds.
    pub fn materialize<T, B>(&self) -> Result<StateDict<T, B>>
    where
        T: Scalar,
        B: ComputeBackend + Default,
    {
        if self.root.scalar.as_str() != type_name::<T>() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "state scalar mismatch: archive {}, requested {}",
                    self.root.scalar.as_str(),
                    type_name::<T>()
                ),
            ));
        }
        if self.root.endianness != host_endianness() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "state archive scalar byte order differs from this host",
            ));
        }
        let backend = B::default();
        let mut tensors = HashMap::new();
        tensors.try_reserve(self.len()).map_err(Error::other)?;
        for archived in self.root.tensors.iter() {
            let name = archived.name.as_str();
            validate_name(name, self.limits)?;
            ensure_limit("tensor rank", self.limits.rank, archived.shape.len())?;
            let shape = archived
                .shape
                .iter()
                .map(|dimension| {
                    usize::try_from(dimension.to_native()).map_err(|error| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("tensor '{name}' dimension is unrepresentable: {error}"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let elements = validate_tensor(name, &shape, archived.bytes.len(), self.limits)?;
            let expected_bytes =
                elements
                    .checked_mul(std::mem::size_of::<T>())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("tensor '{name}' byte count overflows usize"),
                        )
                    })?;
            if archived.bytes.len() != expected_bytes {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "tensor '{name}' payload size mismatch: expected {expected_bytes}, actual {}",
                        archived.bytes.len()
                    ),
                ));
            }
            let values = bytemuck::pod_collect_to_vec::<u8, T>(archived.bytes.as_slice());
            tensors.insert(
                name.to_owned(),
                Tensor::from_slice_on(shape, &values, &backend).map_err(Error::other)?,
            );
        }
        Ok(StateDict { tensors })
    }
}

/// Borrowed metadata and payload for one archived tensor.
pub struct ArchivedTensor<'a> {
    tensor: &'a ArchivedTensorArchiveData,
}

impl ArchivedTensor<'_> {
    /// Archived dimensions without allocation.
    #[must_use]
    pub fn shape(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.tensor
            .shape
            .iter()
            .map(|dimension| dimension.to_native())
    }

    /// Archived scalar payload bytes without copying.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.tensor.bytes.as_slice()
    }
}

/// A dictionary mapping parameter names to tensors.
#[derive(Clone)]
pub struct StateDict<T: Scalar, B: ComputeBackend + Default> {
    /// Parameter name to tensor map.
    pub tensors: HashMap<String, Tensor<T, B>>,
}

impl<T: Scalar, B: ComputeBackend + Default> Default for StateDict<T, B> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }
}

impl<T: Scalar, B: ComputeBackend + Default> StateDict<T, B> {
    /// Create an empty state dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a named tensor.
    pub fn insert<S: Into<String>>(&mut self, name: S, tensor: Tensor<T, B>) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Get a tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tensor<T, B>> {
        self.tensors.get(name)
    }

    /// Number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the dictionary contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Save a deterministic validated archive using default limits.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.save_with_limits(writer, StateLimits::default())
    }

    /// Save a deterministic validated archive under explicit limits.
    pub fn save_with_limits<W: Write>(&self, writer: &mut W, limits: StateLimits) -> Result<()> {
        ensure_limit("tensor count", limits.tensors, self.len())?;
        let backend = B::default();
        let mut names = self.tensors.keys().collect::<Vec<_>>();
        names.sort_unstable();
        let mut total_bytes = 0usize;
        let mut tensors = Vec::with_capacity(names.len());
        for name in names {
            validate_name(name, limits)?;
            let tensor = self
                .tensors
                .get(name)
                .expect("invariant: name originates from this map");
            let contiguous = tensor.to_contiguous_on(&backend).map_err(Error::other)?;
            let shape = contiguous
                .shape()
                .iter()
                .map(|dimension| u64::try_from(*dimension))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Error::other)?;
            let payload = if let Some(slice) = contiguous.storage().try_as_slice() {
                let offset = contiguous.layout().offset();
                bytemuck::cast_slice(&slice[offset..offset + contiguous.numel()]).to_vec()
            } else {
                let mut host = vec![T::zero(); contiguous.numel()];
                backend
                    .copy_to_host(contiguous.storage(), &mut host)
                    .map_err(Error::other)?;
                bytemuck::cast_slice(&host).to_vec()
            };
            let elements = validate_tensor(name, contiguous.shape(), payload.len(), limits)?;
            let expected_bytes =
                elements
                    .checked_mul(std::mem::size_of::<T>())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidData,
                            format!("tensor '{name}' byte count overflows usize"),
                        )
                    })?;
            if payload.len() != expected_bytes {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "tensor '{name}' payload size mismatch: expected {expected_bytes}, actual {}",
                        payload.len()
                    ),
                ));
            }
            total_bytes = total_bytes.checked_add(payload.len()).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "aggregate tensor bytes overflow usize",
                )
            })?;
            ensure_limit(
                "aggregate tensor bytes",
                limits.total_tensor_bytes,
                total_bytes,
            )?;
            tensors.push(TensorArchiveData {
                name: name.clone(),
                shape,
                bytes: payload,
            });
        }
        let archive = StateArchiveData {
            magic: *b"COEU",
            version: 1,
            scalar: type_name::<T>().to_owned(),
            endianness: host_endianness(),
            tensors,
        };
        let bytes = rkyv::to_bytes::<ArchiveError>(&archive).map_err(Error::other)?;
        ensure_limit("archive bytes", limits.archive_bytes, bytes.len())?;
        writer.write_all(&bytes)
    }

    /// Load and materialize a validated archive using default limits.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        Self::load_with_limits(reader, StateLimits::default())
    }

    /// Load and materialize a validated archive under explicit limits.
    pub fn load_with_limits<R: Read>(reader: &mut R, limits: StateLimits) -> Result<Self> {
        let read_limit = u64::try_from(limits.archive_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let mut bytes = rkyv::util::AlignedVec::<16>::new();
        std::io::copy(&mut reader.take(read_limit), &mut bytes)?;
        ensure_limit("archive bytes", limits.archive_bytes, bytes.len())?;
        Self::archive(&bytes, limits)?.materialize()
    }

    /// Validate an aligned archive and borrow its metadata and payloads.
    pub fn archive(bytes: &[u8], limits: StateLimits) -> Result<StateArchive<'_>> {
        ensure_limit("archive bytes", limits.archive_bytes, bytes.len())?;
        let root =
            rkyv::access::<ArchivedStateArchiveData, ArchiveError>(bytes).map_err(|error| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("invalid state archive: {error}"),
                )
            })?;
        validate_archive(root, limits)?;
        Ok(StateArchive { root, limits })
    }
}

fn validate_archive(root: &ArchivedStateArchiveData, limits: StateLimits) -> Result<()> {
    if root.magic != *b"COEU" {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "invalid state archive magic",
        ));
    }
    if root.version.to_native() != 1 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "unsupported state archive version {}",
                root.version.to_native()
            ),
        ));
    }
    ensure_limit("tensor count", limits.tensors, root.tensors.len())?;
    let mut total_bytes = 0usize;
    let mut names = HashSet::new();
    names
        .try_reserve(root.tensors.len())
        .map_err(Error::other)?;
    for tensor in root.tensors.iter() {
        let name = tensor.name.as_str();
        validate_name(name, limits)?;
        if !names.insert(name) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("duplicate tensor name '{name}'"),
            ));
        }
        ensure_limit("tensor rank", limits.rank, tensor.shape.len())?;
        let shape = tensor
            .shape
            .iter()
            .map(|dimension| usize::try_from(dimension.to_native()).map_err(Error::other))
            .collect::<Result<Vec<_>>>()?;
        validate_tensor(name, &shape, tensor.bytes.len(), limits)?;
        total_bytes = total_bytes.checked_add(tensor.bytes.len()).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                "aggregate tensor bytes overflow usize",
            )
        })?;
        ensure_limit(
            "aggregate tensor bytes",
            limits.total_tensor_bytes,
            total_bytes,
        )?;
    }
    Ok(())
}

const fn host_endianness() -> u8 {
    if cfg!(target_endian = "little") {
        1
    } else {
        2
    }
}

fn validate_name(name: &str, limits: StateLimits) -> Result<()> {
    if name.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "tensor name is empty"));
    }
    ensure_limit("tensor name bytes", limits.name_bytes, name.len())
}

fn validate_tensor(
    name: &str,
    shape: &[usize],
    payload_bytes: usize,
    limits: StateLimits,
) -> Result<usize> {
    ensure_limit("tensor rank", limits.rank, shape.len())?;
    ensure_limit("tensor payload bytes", limits.tensor_bytes, payload_bytes)?;
    shape.iter().try_fold(1usize, |product, dimension| {
        product.checked_mul(*dimension).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("tensor '{name}' shape product overflows usize"),
            )
        })
    })
}

fn ensure_limit(label: &str, limit: usize, actual: usize) -> Result<()> {
    if actual > limit {
        Err(Error::new(
            ErrorKind::InvalidData,
            format!("{label} exceed limit: limit {limit}, actual {actual}"),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_duplicate_tensor_names() {
        let tensor = |value| TensorArchiveData {
            name: "duplicate".to_owned(),
            shape: vec![1],
            bytes: vec![value],
        };
        let state = StateArchiveData {
            magic: *b"COEU",
            version: 1,
            scalar: "u8".to_owned(),
            endianness: host_endianness(),
            tensors: vec![tensor(1), tensor(2)],
        };
        let bytes = rkyv::to_bytes::<ArchiveError>(&state).unwrap();
        let root = rkyv::access::<ArchivedStateArchiveData, ArchiveError>(&bytes).unwrap();
        let error = validate_archive(root, StateLimits::default()).unwrap_err();
        assert_eq!(error.to_string(), "duplicate tensor name 'duplicate'");
    }
}
