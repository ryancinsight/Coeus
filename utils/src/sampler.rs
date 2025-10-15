//! Samplers for controlling data access patterns
//!
//! Samplers determine the order and selection of samples from a dataset.
//! They provide the indices that the DataLoader uses to fetch data.

use std::collections::VecDeque;

/// Trait for controlling data access patterns in datasets
///
/// Samplers provide indices that determine which samples are accessed and in what order.
/// This enables shuffling, stratified sampling, and other data access patterns.
pub trait Sampler {
    /// Returns the next index to sample, or None if exhausted
    fn next(&mut self) -> Option<usize>;

    /// Resets the sampler to its initial state
    fn reset(&mut self);

    /// Returns the total number of samples this sampler will yield
    ///
    /// For samplers that yield infinite sequences (with replacement),
    /// this returns usize::MAX.
    fn len(&self) -> usize;

    /// Returns true if the sampler is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Sequential sampler that yields indices in order: [0, 1, 2, ..., n-1]
pub struct SequentialSampler {
    current: usize,
    length: usize,
}

impl SequentialSampler {
    /// Creates a new sequential sampler for a dataset of the given length
    pub fn new(length: usize) -> Self {
        Self { current: 0, length }
    }
}

impl Sampler for SequentialSampler {
    fn next(&mut self) -> Option<usize> {
        if self.current >= self.length {
            None
        } else {
            let index = self.current;
            self.current += 1;
            Some(index)
        }
    }

    fn reset(&mut self) {
        self.current = 0;
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Random sampler that yields indices in random order
///
/// By default, samples without replacement. Set `replacement=true` for
/// sampling with replacement (infinite sequence).
pub struct RandomSampler {
    indices: VecDeque<usize>,
    length: usize,
    replacement: bool,
}

impl RandomSampler {
    /// Creates a new random sampler
    ///
    /// # Arguments
    /// * `length` - The length of the dataset
    /// * `replacement` - If true, sample with replacement (infinite sequence)
    pub fn new(length: usize, replacement: bool) -> Self {
        let mut sampler = Self {
            indices: VecDeque::new(),
            length,
            replacement,
        };
        sampler.reset();
        sampler
    }

    /// Creates a random sampler without replacement (default behavior)
    pub fn without_replacement(length: usize) -> Self {
        Self::new(length, false)
    }

    /// Creates a random sampler with replacement
    pub fn with_replacement(length: usize) -> Self {
        Self::new(length, true)
    }
}

impl Sampler for RandomSampler {
    fn next(&mut self) -> Option<usize> {
        if self.replacement {
            // Sample with replacement - always return a random index
            Some(rand::random::<usize>() % self.length)
        } else {
            // Sample without replacement
            self.indices.pop_front()
        }
    }

    fn reset(&mut self) {
        if !self.replacement {
            // Generate shuffled indices for sampling without replacement
            let mut indices: Vec<usize> = (0..self.length).collect();
            // Simple Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = rand::random::<usize>() % (i + 1);
                indices.swap(i, j);
            }
            self.indices = indices.into();
        }
    }

    fn len(&self) -> usize {
        if self.replacement {
            usize::MAX // Infinite sequence
        } else {
            self.length
        }
    }
}

/// Batch sampler that groups individual sample indices into batches
///
/// This sampler wraps another sampler and groups its output into batches
/// of the specified size.
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Creates a new batch sampler
    ///
    /// # Arguments
    /// * `sampler` - The underlying sampler to wrap
    /// * `batch_size` - Size of each batch
    /// * `drop_last` - If true, drop the last incomplete batch
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    /// Creates a batch sampler that drops incomplete batches
    pub fn drop_last(sampler: S, batch_size: usize) -> Self {
        Self::new(sampler, batch_size, true)
    }

    /// Creates a batch sampler that keeps incomplete batches
    pub fn keep_last(sampler: S, batch_size: usize) -> Self {
        Self::new(sampler, batch_size, false)
    }
}

impl<S: Sampler> Sampler for BatchSampler<S> {
    fn next(&mut self) -> Option<usize> {
        // BatchSampler doesn't yield individual indices - it yields batch indices
        // This implementation is a placeholder; the actual batching logic
        // is handled in DataLoader
        unimplemented!("BatchSampler should be used with DataLoader's batching logic")
    }

    fn reset(&mut self) {
        self.sampler.reset();
    }

    fn len(&self) -> usize {
        let total_samples = self.sampler.len();
        if total_samples == usize::MAX {
            usize::MAX
        } else {
            let batches = total_samples / self.batch_size;
            if self.drop_last || total_samples % self.batch_size == 0 {
                batches
            } else {
                batches + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler::new(5);

        assert_eq!(sampler.len(), 5);

        // Test sequential access
        assert_eq!(sampler.next(), Some(0));
        assert_eq!(sampler.next(), Some(1));
        assert_eq!(sampler.next(), Some(2));
        assert_eq!(sampler.next(), Some(3));
        assert_eq!(sampler.next(), Some(4));
        assert_eq!(sampler.next(), None);

        // Test reset
        sampler.reset();
        assert_eq!(sampler.next(), Some(0));
    }

    #[test]
    fn test_random_sampler_without_replacement() {
        let mut sampler = RandomSampler::without_replacement(5);

        assert_eq!(sampler.len(), 5);

        // Collect all samples
        let mut samples = Vec::new();
        while let Some(index) = sampler.next() {
            samples.push(index);
        }

        assert_eq!(samples.len(), 5);
        assert_eq!(
            samples
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            5
        );

        // Test reset gives different order
        sampler.reset();
        let mut samples2 = Vec::new();
        while let Some(index) = sampler.next() {
            samples2.push(index);
        }

        // Should be same elements but possibly different order
        assert_eq!(
            samples
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            5
        );
        assert_eq!(
            samples2
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            5
        );
    }

    #[test]
    fn test_random_sampler_with_replacement() {
        let mut sampler = RandomSampler::with_replacement(3);

        assert_eq!(sampler.len(), usize::MAX);

        // Should be able to sample multiple times
        for _ in 0..10 {
            let index = sampler.next().unwrap();
            assert!(index < 3);
        }
    }

    #[test]
    fn test_batch_sampler() {
        let sequential = SequentialSampler::new(7);
        let batch_sampler = BatchSampler::drop_last(sequential, 3);

        // Should have 2 full batches (6 samples), drop last 1
        assert_eq!(batch_sampler.len(), 2);

        let batch_sampler_keep = BatchSampler::keep_last(SequentialSampler::new(7), 3);
        // Should have 3 batches: 2 full + 1 partial
        assert_eq!(batch_sampler_keep.len(), 3);
    }
}
