use libloading::Library;
use std::sync::{Arc, OnceLock};

pub type CUdevice = i32;
pub type CUcontext = *mut std::ffi::c_void;
pub type CUdeviceptr = u64;
pub type CUresult = i32;
pub type CUmodule = *mut std::ffi::c_void;
pub type CUfunction = *mut std::ffi::c_void;
pub type CUstream = *mut std::ffi::c_void;

/// Dynamically loaded CUDA driver function pointers.
pub struct CudaDriver {
    _lib: Library,
    pub cu_init: unsafe extern "C" fn(flags: u32) -> CUresult,
    pub cu_device_get: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cu_ctx_create:
        unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult,
    pub cu_mem_alloc: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cu_mem_free: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cu_memcpy_htod: unsafe extern "C" fn(
        dst_device: CUdeviceptr,
        src_host: *const std::ffi::c_void,
        byte_count: usize,
    ) -> CUresult,
    pub cu_memcpy_dtoh: unsafe extern "C" fn(
        dst_host: *mut std::ffi::c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult,
    pub cu_memcpy_dtod: unsafe extern "C" fn(
        dst_device: CUdeviceptr,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult,
    pub cu_module_load_data:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const std::ffi::c_void) -> CUresult,
    pub cu_module_get_function: unsafe extern "C" fn(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const std::ffi::c_char,
    ) -> CUresult,
    pub cu_module_unload: unsafe extern "C" fn(hmod: CUmodule) -> CUresult,
    #[allow(non_snake_case)]
    pub cu_launch_kernel: unsafe extern "C" fn(
        f: CUfunction,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: CUstream,
        kernelParams: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CUresult,
}

/// Thread-safe wrapper around raw CUDA context pointers.
#[derive(Debug, Clone, Copy)]
pub struct CUcontextWrapper(pub CUcontext);
unsafe impl Send for CUcontextWrapper {}
unsafe impl Sync for CUcontextWrapper {}

static DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();
static CONTEXT: OnceLock<Option<CUcontextWrapper>> = OnceLock::new();

unsafe extern "C" fn local_cu_init(flags: u32) -> CUresult {
    cuda_core::sys::cuInit(flags) as CUresult
}

unsafe extern "C" fn local_cu_device_get(device: *mut CUdevice, ordinal: i32) -> CUresult {
    let mut dev = 0;
    let res = cuda_core::sys::cuDeviceGet(&mut dev, ordinal);
    if res == 0 {
        *device = dev;
    }
    res as CUresult
}

unsafe extern "C" fn local_cu_ctx_create(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult {
    cuda_core::sys::cuCtxCreate_v4(
        pctx as *mut *mut _,
        std::ptr::null_mut(),
        flags,
        dev,
    ) as CUresult
}

unsafe extern "C" fn local_cu_mem_alloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    cuda_core::sys::cuMemAlloc_v2(dptr as *mut _, bytesize) as CUresult
}

unsafe extern "C" fn local_cu_mem_free(dptr: CUdeviceptr) -> CUresult {
    cuda_core::sys::cuMemFree_v2(dptr) as CUresult
}

unsafe extern "C" fn local_cu_memcpy_htod(dst: CUdeviceptr, src: *const std::ffi::c_void, bytesize: usize) -> CUresult {
    cuda_core::sys::cuMemcpyHtoD_v2(dst, src, bytesize) as CUresult
}

unsafe extern "C" fn local_cu_memcpy_dtoh(dst: *mut std::ffi::c_void, src: CUdeviceptr, bytesize: usize) -> CUresult {
    cuda_core::sys::cuMemcpyDtoH_v2(dst, src, bytesize) as CUresult
}

unsafe extern "C" fn local_cu_memcpy_dtod(dst: CUdeviceptr, src: CUdeviceptr, bytesize: usize) -> CUresult {
    cuda_core::sys::cuMemcpyDtoD_v2(dst, src, bytesize) as CUresult
}

unsafe extern "C" fn local_cu_module_load_data(module: *mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    cuda_core::sys::cuModuleLoadData(module as *mut *mut _, image) as CUresult
}

unsafe extern "C" fn local_cu_module_get_function(hfunc: *mut CUfunction, hmod: CUmodule, name: *const std::ffi::c_char) -> CUresult {
    cuda_core::sys::cuModuleGetFunction(hfunc as *mut *mut _, hmod as *mut _, name) as CUresult
}

unsafe extern "C" fn local_cu_module_unload(hmod: CUmodule) -> CUresult {
    cuda_core::sys::cuModuleUnload(hmod as *mut _) as CUresult
}

unsafe extern "C" fn local_cu_launch_kernel(
    f: CUfunction,
    gridDimX: u32,
    gridDimY: u32,
    gridDimZ: u32,
    blockDimX: u32,
    blockDimY: u32,
    blockDimZ: u32,
    sharedMemBytes: u32,
    hStream: CUstream,
    kernelParams: *mut *mut std::ffi::c_void,
    extra: *mut *mut std::ffi::c_void,
) -> CUresult {
    cuda_core::sys::cuLaunchKernel(
        f as *mut _,
        gridDimX,
        gridDimY,
        gridDimZ,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        hStream as *mut _,
        kernelParams as *mut *mut _,
        extra as *mut *mut _,
    ) as CUresult
}

impl CudaDriver {
    /// Retrieve a reference to the dynamically loaded driver if available.
    pub fn get() -> Option<&'static Self> {
        DRIVER
            .get_or_init(|| unsafe {
                let lib_name = if cfg!(windows) {
                    "nvcuda.dll"
                } else {
                    "libcuda.so"
                };
                let lib = Library::new(lib_name).ok()?;

                Some(Self {
                    _lib: lib,
                    cu_init: local_cu_init,
                    cu_device_get: local_cu_device_get,
                    cu_ctx_create: local_cu_ctx_create,
                    cu_mem_alloc: local_cu_mem_alloc,
                    cu_mem_free: local_cu_mem_free,
                    cu_memcpy_htod: local_cu_memcpy_htod,
                    cu_memcpy_dtoh: local_cu_memcpy_dtoh,
                    cu_memcpy_dtod: local_cu_memcpy_dtod,
                    cu_module_load_data: local_cu_module_load_data,
                    cu_module_get_function: local_cu_module_get_function,
                    cu_module_unload: local_cu_module_unload,
                    cu_launch_kernel: local_cu_launch_kernel,
                })
            })
            .as_ref()
    }
}

/// Retrieve a reference to the active CUDA driver context.
pub fn get_cuda_context() -> Option<CUcontext> {
    let ctx = CONTEXT
        .get_or_init(|| {
            cuda_async::device_context::with_device(0, |device| {
                CUcontextWrapper(device.cu_ctx() as CUcontext)
            })
            .ok()
        })
        .as_ref()?
        .0;

    if let Some(device) = get_borrowed_device() {
        let cu_ctx = device.cu_ctx();
        if cu_ctx.is_null() {
            panic!("device.cu_ctx() is NULL!");
        }
        if let Err(e) = device.bind_to_thread() {
            panic!("device.bind_to_thread() failed: {:?}", e);
        }
    }

    Some(ctx)
}

static BORROWED_DEVICE: OnceLock<Option<Arc<cuda_core::Device>>> = OnceLock::new();
static BORROWED_STREAM: OnceLock<Option<Arc<cuda_core::Stream>>> = OnceLock::new();

/// Retrieve the borrowed cutile Device wrapper.
pub fn get_borrowed_device() -> Option<Arc<cuda_core::Device>> {
    BORROWED_DEVICE
        .get_or_init(|| {
            cuda_async::device_context::with_device(0, |device| {
                device.clone()
            })
            .ok()
        })
        .clone()
}

/// Retrieve the borrowed cutile Stream wrapper.
pub fn get_borrowed_stream() -> Option<Arc<cuda_core::Stream>> {
    BORROWED_STREAM
        .get_or_init(|| {
            cuda_async::device_context::with_default_device_policy(|policy| {
                policy.next_stream().ok()
            })
            .ok()
            .flatten()
        })
        .clone()
}

#[allow(non_camel_case_types)]
pub type nvrtcProgram = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type nvrtcResult = i32;

#[allow(non_snake_case)]
pub struct NvrtcDriver {
    _lib: Library,
    pub nvrtcCreateProgram: unsafe extern "C" fn(
        prog: *mut nvrtcProgram,
        src: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        numHeaders: std::ffi::c_int,
        headers: *const *const std::ffi::c_char,
        includeNames: *const *const std::ffi::c_char,
    ) -> nvrtcResult,
    pub nvrtcCompileProgram: unsafe extern "C" fn(
        prog: nvrtcProgram,
        numOptions: std::ffi::c_int,
        options: *const *const std::ffi::c_char,
    ) -> nvrtcResult,
    pub nvrtcGetPTXSize:
        unsafe extern "C" fn(prog: nvrtcProgram, ptxSize: *mut usize) -> nvrtcResult,
    pub nvrtcGetPTX:
        unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut std::ffi::c_char) -> nvrtcResult,
    pub nvrtcGetProgramLogSize:
        unsafe extern "C" fn(prog: nvrtcProgram, logSize: *mut usize) -> nvrtcResult,
    pub nvrtcGetProgramLog:
        unsafe extern "C" fn(prog: nvrtcProgram, log: *mut std::ffi::c_char) -> nvrtcResult,
    pub nvrtcDestroyProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult,
    pub nvrtcGetErrorString: unsafe extern "C" fn(result: nvrtcResult) -> *const std::ffi::c_char,
}

static NVRTC_DRIVER: OnceLock<Option<NvrtcDriver>> = OnceLock::new();

impl NvrtcDriver {
    #[allow(non_snake_case)]
    pub fn get() -> Option<&'static Self> {
        NVRTC_DRIVER
            .get_or_init(|| {
                let lib = find_nvrtc_library()?;
                unsafe {
                    let nvrtcCreateProgram = *lib.get(b"nvrtcCreateProgram\0").ok()?;
                    let nvrtcCompileProgram = *lib.get(b"nvrtcCompileProgram\0").ok()?;
                    let nvrtcGetPTXSize = *lib.get(b"nvrtcGetPTXSize\0").ok()?;
                    let nvrtcGetPTX = *lib.get(b"nvrtcGetPTX\0").ok()?;
                    let nvrtcGetProgramLogSize = *lib.get(b"nvrtcGetProgramLogSize\0").ok()?;
                    let nvrtcGetProgramLog = *lib.get(b"nvrtcGetProgramLog\0").ok()?;
                    let nvrtcDestroyProgram = *lib.get(b"nvrtcDestroyProgram\0").ok()?;
                    let nvrtcGetErrorString = *lib.get(b"nvrtcGetErrorString\0").ok()?;

                    Some(Self {
                        _lib: lib,
                        nvrtcCreateProgram,
                        nvrtcCompileProgram,
                        nvrtcGetPTXSize,
                        nvrtcGetPTX,
                        nvrtcGetProgramLogSize,
                        nvrtcGetProgramLog,
                        nvrtcDestroyProgram,
                        nvrtcGetErrorString,
                    })
                }
            })
            .as_ref()
    }
}

fn find_nvrtc_library() -> Option<Library> {
    if let Ok(lib) = unsafe { Library::new("nvrtc") } {
        return Some(lib);
    }
    if let Ok(lib) = unsafe { Library::new("nvrtc64") } {
        return Some(lib);
    }

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let paths = vec![
            format!("{}/bin/x64", cuda_path),
            format!("{}/bin", cuda_path),
            format!("{}/lib64", cuda_path),
            format!("{}/lib", cuda_path),
        ];
        for dir in paths {
            let cleaned_dir = dir.replace('\\', "/");
            if let Ok(entries) = std::fs::read_dir(&cleaned_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                            let matches = if cfg!(windows) {
                                filename.starts_with("nvrtc") && !filename.contains("builtins") && filename.ends_with(".dll")
                            } else {
                                filename.starts_with("libnvrtc")
                                    && (filename.ends_with(".so") || filename.contains(".so."))
                            };
                            if matches {
                                if let Ok(lib) = unsafe { Library::new(&path) } {
                                    return Some(lib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let fallback_names = if cfg!(windows) {
        vec![
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
        ]
    } else {
        vec!["libnvrtc.so"]
    };
    for name in fallback_names {
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Some(lib);
        }
    }
    None
}
