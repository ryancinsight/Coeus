use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, RwLock};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineCacheKey {
    device_addr: usize,
    key: String,
    entry_point: String,
    source_hash: u64,
}

#[inline]
fn shader_source_hash(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

#[inline]
fn device_addr(device: &wgpu::Device) -> usize {
    (device as *const wgpu::Device) as usize
}

/// Pipeline cache to avoid recompiling compute shaders.
pub struct PipelineCache {
    pipelines: RwLock<HashMap<PipelineCacheKey, Arc<wgpu::ComputePipeline>>>,
}

impl PipelineCache {
    /// Retrieve a compute pipeline from cache or compile it.
    pub fn get_or_create(
        &self,
        key: &str,
        device: &wgpu::Device,
        source: &str,
        entry_point: &str,
    ) -> Arc<wgpu::ComputePipeline> {
        let cache_key = PipelineCacheKey {
            device_addr: device_addr(device),
            key: key.to_string(),
            entry_point: entry_point.to_string(),
            source_hash: shader_source_hash(source),
        };

        if let Some(pipeline) = self.pipelines.read().unwrap().get(&cache_key) {
            return pipeline.clone();
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_arc = Arc::new(pipeline);
        let mut cache = self.pipelines.write().unwrap();
        if let Some(existing) = cache.get(&cache_key) {
            existing.clone()
        } else {
            cache.insert(cache_key, pipeline_arc.clone());
            pipeline_arc
        }
    }
}

/// Global compute pipeline cache.
pub static PIPELINE_CACHE: LazyLock<PipelineCache> = LazyLock::new(|| PipelineCache {
    pipelines: RwLock::new(HashMap::new()),
});
