use super::{RocmBackend, RocmProvider};
use coeus_core::Layout;
use coeus_hephaestus::{
    random_normal, random_uniform, HephaestusBackendError, HephaestusStorage, RandomInitProvider,
};

impl RandomInitProvider<f32> for RocmProvider {
    type Operations = hephaestus_rocm::RocmRandomOps;
}

impl coeus_ops::RandomInitOps<f32> for RocmBackend {
    fn uniform_random(
        &self,
        layout: &Layout,
        low: f32,
        high: f32,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        random_uniform::<RocmProvider, _>(layout, low, high, seed)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("uniform initialization", source))
    }

    fn normal_random(
        &self,
        layout: &Layout,
        mean: f32,
        std_dev: f32,
        seed: u64,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        random_normal::<RocmProvider, _>(layout, mean, std_dev, seed)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("normal initialization", source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::ComputeBackend;
    use coeus_ops::RandomInitOps;

    #[test]
    fn invalid_rank_is_typed_without_device_acquisition() {
        let Err(error) = RocmBackend::new().uniform_random(
            &Layout::new(Vec::<usize>::new().into()),
            -0.75,
            1.25,
            17,
        ) else {
            panic!("rank-zero dispatch must fail before device acquisition");
        };
        assert!(matches!(
            error,
            HephaestusBackendError::Device {
                operation: "uniform initialization",
                source: hephaestus_core::HephaestusError::InvalidConfiguration { .. },
            }
        ));
    }

    #[test]
    fn seeded_values_match_leto_oracle() {
        if hephaestus_rocm::RocmDevice::try_default().is_err() {
            assert_ne!(
                std::env::var("HEPHAESTUS_ROCM_REQUIRE_DEVICE").as_deref(),
                Ok("1"),
                "ROCm CI requires an acquired device"
            );
            return;
        }
        let backend = RocmBackend::new();
        let layout = Layout::new([2, 3].into());
        let expected = coeus_leto::uniform_values(&[2, 3], -0.75_f32, 1.25, 17)
            .expect("Leto must generate a valid uniform oracle");
        let storage = backend
            .uniform_random(&layout, -0.75, 1.25, 17)
            .expect("ROCm provider must initialize a valid layout");
        let mut actual = vec![0.0; expected.len()];
        backend.copy_to_host(&storage, &mut actual);
        assert_eq!(actual, expected);

        let expected = coeus_leto::normal_values(&[2, 3], 0.25_f32, 0.5, 29)
            .expect("Leto must generate a valid normal oracle");
        let storage = backend
            .normal_random(&layout, 0.25, 0.5, 29)
            .expect("ROCm provider must initialize a valid layout");
        let mut actual = vec![0.0; expected.len()];
        backend.copy_to_host(&storage, &mut actual);
        assert_eq!(actual, expected);
    }
}
