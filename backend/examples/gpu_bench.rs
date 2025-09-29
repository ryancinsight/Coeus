use coeus_backend::Backend;
use coeus_backend::cpu::CpuBackend;

fn main() {
    let backend = CpuBackend;  // Fallback to CPU for bench
    // Simple bench stub
    println!("GPU bench stub - using CPU fallback");
}
