use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

/// Pipeline cache to avoid recompiling compute shaders.
pub struct PipelineCache {
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
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
        let mut cache = self.pipelines.lock().unwrap();
        if let Some(pipeline) = cache.get(key) {
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
        cache.insert(key.to_string(), pipeline_arc.clone());
        pipeline_arc
    }
}

/// Global compute pipeline cache.
pub static PIPELINE_CACHE: LazyLock<PipelineCache> = LazyLock::new(|| PipelineCache {
    pipelines: Mutex::new(HashMap::new()),
});
