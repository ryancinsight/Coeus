use super::device_outputs::{
    preserves_output_clones, rejects_invalid_output_write, scans_preserve_output_clones, Add,
    Negate, Product, Square, Sum,
};
use coeus_core::{MoiraiBackend, SequentialBackend};

#[test]
fn cpu_negate_preserves_output_clones() {
    preserves_output_clones(&SequentialBackend::new(), Negate, 1_i8);
    preserves_output_clones(&SequentialBackend::new(), Negate, 1_i16);
    preserves_output_clones(&SequentialBackend::new(), Negate, 1_i32);
    preserves_output_clones(&SequentialBackend::new(), Negate, 1_i64);
    preserves_output_clones(&SequentialBackend::new(), Negate, 1.0_f32);
    preserves_output_clones(&SequentialBackend::new(), Negate, 1.0_f64);
    preserves_output_clones(
        &SequentialBackend::new(),
        Negate,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &SequentialBackend::new(),
        Negate,
        eunomia::Bf16::from_bits(0x3f80),
    );
    preserves_output_clones(&MoiraiBackend::new(), Negate, 1_i8);
    preserves_output_clones(&MoiraiBackend::new(), Negate, 1_i16);
    preserves_output_clones(&MoiraiBackend::new(), Negate, 1_i32);
    preserves_output_clones(&MoiraiBackend::new(), Negate, 1_i64);
    preserves_output_clones(&MoiraiBackend::new(), Negate, 1.0_f32);
    preserves_output_clones(&MoiraiBackend::new(), Negate, 1.0_f64);
    preserves_output_clones(
        &MoiraiBackend::new(),
        Negate,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &MoiraiBackend::new(),
        Negate,
        eunomia::Bf16::from_bits(0x3f80),
    );
}

#[test]
fn cpu_add_preserves_output_clones() {
    preserves_output_clones(&SequentialBackend::new(), Add, 1_i8);
    preserves_output_clones(&SequentialBackend::new(), Add, 1_i16);
    preserves_output_clones(&SequentialBackend::new(), Add, 1_i32);
    preserves_output_clones(&SequentialBackend::new(), Add, 1_i64);
    preserves_output_clones(&SequentialBackend::new(), Add, 1.0_f32);
    preserves_output_clones(&SequentialBackend::new(), Add, 1.0_f64);
    preserves_output_clones(
        &SequentialBackend::new(),
        Add,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &SequentialBackend::new(),
        Add,
        eunomia::Bf16::from_bits(0x3f80),
    );
    preserves_output_clones(&SequentialBackend::new(), Add, 1_u8);
    preserves_output_clones(&SequentialBackend::new(), Add, 1_u16);
    preserves_output_clones(&SequentialBackend::new(), Add, 1_u32);
    preserves_output_clones(&SequentialBackend::new(), Add, 1_u64);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_i8);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_i16);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_i32);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_i64);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1.0_f32);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1.0_f64);
    preserves_output_clones(&MoiraiBackend::new(), Add, eunomia::F16::from_bits(0x3c00));
    preserves_output_clones(&MoiraiBackend::new(), Add, eunomia::Bf16::from_bits(0x3f80));
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_u8);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_u16);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_u32);
    preserves_output_clones(&MoiraiBackend::new(), Add, 1_u64);
}

#[test]
fn cpu_sum_preserves_output_clones() {
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_i8);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_i16);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_i32);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_i64);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1.0_f32);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1.0_f64);
    preserves_output_clones(
        &SequentialBackend::new(),
        Sum,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &SequentialBackend::new(),
        Sum,
        eunomia::Bf16::from_bits(0x3f80),
    );
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_u8);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_u16);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_u32);
    preserves_output_clones(&SequentialBackend::new(), Sum, 1_u64);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_i8);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_i16);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_i32);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_i64);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1.0_f32);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1.0_f64);
    preserves_output_clones(&MoiraiBackend::new(), Sum, eunomia::F16::from_bits(0x3c00));
    preserves_output_clones(&MoiraiBackend::new(), Sum, eunomia::Bf16::from_bits(0x3f80));
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_u8);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_u16);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_u32);
    preserves_output_clones(&MoiraiBackend::new(), Sum, 1_u64);
}

#[test]
fn cpu_product_preserves_output_clones() {
    preserves_output_clones(&SequentialBackend::new(), Product, 1_i8);
    preserves_output_clones(&SequentialBackend::new(), Product, 1_i16);
    preserves_output_clones(&SequentialBackend::new(), Product, 1_i32);
    preserves_output_clones(&SequentialBackend::new(), Product, 1_i64);
    preserves_output_clones(&SequentialBackend::new(), Product, 1.0_f32);
    preserves_output_clones(&SequentialBackend::new(), Product, 1.0_f64);
    preserves_output_clones(
        &SequentialBackend::new(),
        Product,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &SequentialBackend::new(),
        Product,
        eunomia::Bf16::from_bits(0x3f80),
    );
    preserves_output_clones(&SequentialBackend::new(), Product, 1_u8);
    preserves_output_clones(&SequentialBackend::new(), Product, 1_u16);
    preserves_output_clones(&SequentialBackend::new(), Product, 1_u32);
    preserves_output_clones(&SequentialBackend::new(), Product, 1_u64);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_i8);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_i16);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_i32);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_i64);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1.0_f32);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1.0_f64);
    preserves_output_clones(
        &MoiraiBackend::new(),
        Product,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &MoiraiBackend::new(),
        Product,
        eunomia::Bf16::from_bits(0x3f80),
    );
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_u8);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_u16);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_u32);
    preserves_output_clones(&MoiraiBackend::new(), Product, 1_u64);
}

#[test]
fn cpu_invalid_output_requests_preserve_values() {
    rejects_invalid_output_write(&SequentialBackend::new(), Negate, 1_i8);
    rejects_invalid_output_write(&SequentialBackend::new(), Negate, 1_i16);
    rejects_invalid_output_write(&SequentialBackend::new(), Negate, 1_i32);
    rejects_invalid_output_write(&SequentialBackend::new(), Negate, 1_i64);
    rejects_invalid_output_write(&SequentialBackend::new(), Negate, 1.0_f32);
    rejects_invalid_output_write(&SequentialBackend::new(), Negate, 1.0_f64);
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Negate,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Negate,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(&MoiraiBackend::new(), Negate, 1_i8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Negate, 1_i16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Negate, 1_i32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Negate, 1_i64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Negate, 1.0_f32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Negate, 1.0_f64);
    rejects_invalid_output_write(
        &MoiraiBackend::new(),
        Negate,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &MoiraiBackend::new(),
        Negate,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_i8);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_i16);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_i32);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_i64);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1.0_f32);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1.0_f64);
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Add,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Add,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_u8);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_u16);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_u32);
    rejects_invalid_output_write(&SequentialBackend::new(), Add, 1_u64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_i8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_i16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_i32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_i64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1.0_f32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1.0_f64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, eunomia::F16::from_bits(0x3c00));
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, eunomia::Bf16::from_bits(0x3f80));
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_u8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_u16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_u32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Add, 1_u64);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_i8);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_i16);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_i32);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_i64);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1.0_f32);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1.0_f64);
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Sum,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Sum,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_u8);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_u16);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_u32);
    rejects_invalid_output_write(&SequentialBackend::new(), Sum, 1_u64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_i8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_i16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_i32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_i64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1.0_f32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1.0_f64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, eunomia::F16::from_bits(0x3c00));
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, eunomia::Bf16::from_bits(0x3f80));
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_u8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_u16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_u32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Sum, 1_u64);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_i8);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_i16);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_i32);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_i64);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1.0_f32);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1.0_f64);
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Product,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Product,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_u8);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_u16);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_u32);
    rejects_invalid_output_write(&SequentialBackend::new(), Product, 1_u64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_i8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_i16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_i32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_i64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1.0_f32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1.0_f64);
    rejects_invalid_output_write(
        &MoiraiBackend::new(),
        Product,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &MoiraiBackend::new(),
        Product,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_u8);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_u16);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_u32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Product, 1_u64);
}

#[test]
fn cpu_square_preserves_output_clones() {
    preserves_output_clones(&SequentialBackend::new(), Square, 1.0_f32);
    rejects_invalid_output_write(&SequentialBackend::new(), Square, 1.0_f32);
    preserves_output_clones(&SequentialBackend::new(), Square, 1.0_f64);
    rejects_invalid_output_write(&SequentialBackend::new(), Square, 1.0_f64);
    preserves_output_clones(
        &SequentialBackend::new(),
        Square,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Square,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &SequentialBackend::new(),
        Square,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(
        &SequentialBackend::new(),
        Square,
        eunomia::Bf16::from_bits(0x3f80),
    );
    preserves_output_clones(&MoiraiBackend::new(), Square, 1.0_f32);
    rejects_invalid_output_write(&MoiraiBackend::new(), Square, 1.0_f32);
    preserves_output_clones(&MoiraiBackend::new(), Square, 1.0_f64);
    rejects_invalid_output_write(&MoiraiBackend::new(), Square, 1.0_f64);
    preserves_output_clones(
        &MoiraiBackend::new(),
        Square,
        eunomia::F16::from_bits(0x3c00),
    );
    rejects_invalid_output_write(
        &MoiraiBackend::new(),
        Square,
        eunomia::F16::from_bits(0x3c00),
    );
    preserves_output_clones(
        &MoiraiBackend::new(),
        Square,
        eunomia::Bf16::from_bits(0x3f80),
    );
    rejects_invalid_output_write(
        &MoiraiBackend::new(),
        Square,
        eunomia::Bf16::from_bits(0x3f80),
    );
}

#[test]
fn cpu_scans_preserves_output_clones() {
    scans_preserve_output_clones(&SequentialBackend::new(), 1_i8);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_i16);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_i32);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_i64);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_u8);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_u16);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_u32);
    scans_preserve_output_clones(&SequentialBackend::new(), 1_u64);
    scans_preserve_output_clones(&SequentialBackend::new(), 1.0_f32);
    scans_preserve_output_clones(&SequentialBackend::new(), 1.0_f64);
    scans_preserve_output_clones(&SequentialBackend::new(), eunomia::F16::from_bits(0x3c00));
    scans_preserve_output_clones(&SequentialBackend::new(), eunomia::Bf16::from_bits(0x3f80));
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_i8);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_i16);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_i32);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_i64);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_u8);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_u16);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_u32);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1_u64);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1.0_f32);
    scans_preserve_output_clones(&MoiraiBackend::new(), 1.0_f64);
    scans_preserve_output_clones(&MoiraiBackend::new(), eunomia::F16::from_bits(0x3c00));
    scans_preserve_output_clones(&MoiraiBackend::new(), eunomia::Bf16::from_bits(0x3f80));
}
