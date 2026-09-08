use coeus_core::{ComputeBackend, Scalar};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Write {
    Fill,
    FillZero,
    CopyToDevice,
}

pub(crate) fn preserves_cloned_storage<B: ComputeBackend>(backend: &B, write: Write) {
    scalar_writes(backend, write, 1_i8);
    scalar_writes(backend, write, 1_i16);
    scalar_writes(backend, write, 1_i32);
    scalar_writes(backend, write, 1_i64);
    scalar_writes(backend, write, 1_u8);
    scalar_writes(backend, write, 1_u16);
    scalar_writes(backend, write, 1_u32);
    scalar_writes(backend, write, 1_u64);
    scalar_writes(backend, write, 1.0_f32);
    scalar_writes(backend, write, 1.0_f64);
    scalar_writes(backend, write, eunomia::F16::from_bits(0x3c00));
    scalar_writes(backend, write, eunomia::Bf16::from_bits(0x3f80));
}

fn scalar_writes<T: Scalar, B: ComputeBackend>(backend: &B, write: Write, one: T) {
    let two = one + one;
    let three = two + one;
    for len in [0, 4, 1, 2, 3, 5, 8] {
        let expected_original: Vec<_> =
            (0..len).map(|index| [one, two, three][index % 3]).collect();
        let uploaded: Vec<_> = (0..len).map(|index| [three, one, two][index % 3]).collect();
        let mut original = backend.allocate(len);
        backend.copy_to_device(&expected_original, &mut original);
        let mut modified = original.clone();
        let expected_modified = match write {
            Write::Fill => {
                backend.fill(&mut modified, two);
                vec![two; len]
            }
            Write::FillZero => {
                backend.fill_zero(&mut modified);
                vec![T::zero(); len]
            }
            Write::CopyToDevice => {
                backend.copy_to_device(&uploaded, &mut modified);
                uploaded
            }
        };
        let mut original_values = vec![T::zero(); len];
        let mut modified_values = vec![T::zero(); len];
        backend.copy_to_host(&original, &mut original_values);
        backend.copy_to_host(&modified, &mut modified_values);
        assert_eq!(
            original_values,
            expected_original,
            "{} {write:?} changed shared {} storage of length {len}",
            core::any::type_name::<B>(),
            core::any::type_name::<T>()
        );
        assert_eq!(
            modified_values,
            expected_modified,
            "{} {write:?} produced incorrect {} storage of length {len}",
            core::any::type_name::<B>(),
            core::any::type_name::<T>()
        );
    }
}
