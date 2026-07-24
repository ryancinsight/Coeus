use coeus_core::Scalar;

/// A ZST tag representing a reduction operation used in collectives.
///
/// Each reduction (sum, min, max, product) is a zero-sized type implementing this trait,
/// selected as a const generic-style type parameter to [`Communicator::all_reduce`](crate::Communicator::all_reduce)
/// / [`Communicator::reduce`](crate::Communicator::reduce) so the operation is resolved at compile time.
pub trait ReduceOpTag: 'static + Copy + Clone + Send + Sync {
    /// Apply the reduction binary operation to compute the reduced scalar value.
    fn apply<T: Scalar>(a: T, b: T) -> T;
}

/// Sum reduction tag.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{Communicator, LocalCommunicator, Sum};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let comms = LocalCommunicator::create_cluster(2);
/// let mut handles = vec![];
/// for comm in comms {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         let mut tensor =
///             Tensor::from_slice_on([1], &[rank + 1.0], &backend);
///         comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);
///         // 1 + 2 = 3
///         assert_eq!(tensor.as_slice()[0], 3.0);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Sum;
impl ReduceOpTag for Sum {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        a + b
    }
}

/// Max reduction tag.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{Communicator, LocalCommunicator, Max};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let comms = LocalCommunicator::create_cluster(3);
/// let mut handles = vec![];
/// for comm in comms {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         // rank r contributes [r+1, r+2] -> [1,2], [2,3], [3,4]
///         let mut tensor =
///             Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
///         comm.all_reduce::<f32, _, Max>(&mut tensor, &backend);
///         // max across 3 ranks: [3, 4]
///         assert_eq!(tensor.as_slice(), &[3.0, 4.0]);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Max;
impl ReduceOpTag for Max {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        if a > b {
            a
        } else {
            b
        }
    }
}

/// Min reduction tag.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{Communicator, LocalCommunicator, Min};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let comms = LocalCommunicator::create_cluster(3);
/// let mut handles = vec![];
/// for comm in comms {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         // rank r contributes [r+1, r+2] -> [1,2], [2,3], [3,4]
///         let mut tensor =
///             Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
///         comm.all_reduce::<f32, _, Min>(&mut tensor, &backend);
///         // min across 3 ranks: [1, 2]
///         assert_eq!(tensor.as_slice(), &[1.0, 2.0]);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Min;
impl ReduceOpTag for Min {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        if a < b {
            a
        } else {
            b
        }
    }
}

/// Product reduction tag.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{Communicator, LocalCommunicator, Product};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let comms = LocalCommunicator::create_cluster(3);
/// let mut handles = vec![];
/// for comm in comms {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         // rank r contributes [r+1, r+2] -> [1,2], [2,3], [3,4]
///         let mut tensor =
///             Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
///         comm.all_reduce::<f32, _, Product>(&mut tensor, &backend);
///         // product across 3 ranks: [1*2*3, 2*3*4] = [6, 24]
///         assert_eq!(tensor.as_slice(), &[6.0, 24.0]);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Product;
impl ReduceOpTag for Product {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        a * b
    }
}
