# PyTorch API Parity Gap Analysis

This report categorizes missing PyTorch functionality by priority and implementability.

## Summary Statistics

**Total Missing Items:** 1393

**Critical:** 355
  - Implementable: 355
  - Architectural: 0
  - Internal: 0
**Important:** 307
  - Implementable: 233
  - Architectural: 74
  - Internal: 0
**Optional:** 731
  - Implementable: 493
  - Architectural: 97
  - Internal: 141

---

## Critical Priority

### Implementable (355 items)

#### Automatic differentiation functionality

- `Gradient`
- `autograd`
- `batch_norm_backward_elemt`
- `batch_norm_backward_reduce`
- `enable_grad`
- `gradient`
- `is_grad_enabled`
- `set_grad_enabled`

#### CUDA backend support needed

- `cuda`
- `profiler_allow_cudagraph_cupti_lazy_reinit_cuda12`

#### Core neural network layer

- `nn.AlphaDropout`
- `nn.BatchNorm1d`
- `nn.Conv1d`
- `nn.Conv2d`
- `nn.Conv3d`
- `nn.Dropout1d`
- `nn.Dropout2d`
- `nn.Dropout3d`
- `nn.EmbeddingBag`
- `nn.FeatureAlphaDropout`
- `nn.GRU`
- `nn.GRUCell`
- `nn.LSTM`
- `nn.LSTMCell`
- `nn.LayerNorm`
- `nn.LazyBatchNorm1d`
- `nn.LazyBatchNorm2d`
- `nn.LazyConv1d`
- `nn.LazyConv2d`
- `nn.LazyConv3d`
- `nn.LazyLinear`
- `nn.Module`
- `nn.ModuleDict`
- `nn.ModuleList`
- `nn.Parameter`
- `nn.ParameterDict`
- `nn.ParameterList`
- `nn.RNN`
- `nn.RNNBase`
- `nn.RNNCell`
- `nn.RNNCellBase`
- `nn.UninitializedParameter`

#### Core optimizer

- `optim.ASGD`
- `optim.NAdam`
- `optim.Optimizer`
- `optim.RAdam`
- `optim.lr_scheduler.Optimizer`

#### Core tensor operation

- `DisableTorchFunctionSubclass`
- `SUM`
- `abs`
- `abs_`
- `absolute`
- `adaptive_max_pool1d`
- `add`
- `addbmm`
- `addcdiv`
- `addcmul`
- `addmm`
- `addmv`
- `addmv_`
- `addr`
- `amax`
- `amin`
- `aminmax`
- `arcsin`
- `arcsin_`
- `arcsinh`
- `arcsinh_`
- `arctan`
- `arctan2`
- `arctan_`
- `arctanh`
- `arctanh_`
- `are_deterministic_algorithms_enabled`
- `as_strided_scatter`
- `asin`
- `asin_`
- `asinh`
- `asinh_`
- `atan`
- `atan2`
- `atan_`
- `atanh`
- `atanh_`
- `baddbmm`
- `bmm`
- `chain_matmul`
- `chunk`
- `clamp_max`
- `clamp_max_`
- `clamp_min`
- `clamp_min_`
- `column_stack`
- `concat`
- `concatenate`
- `constant_pad_nd`
- `cummax`
- ... and 232 more

#### Tensor creation function

- `align_tensors`
- `as_tensor`
- `broadcast_tensors`
- `empty_like`
- `empty_permuted`
- `empty_strided`
- `eye`
- `from_numpy`
- `full_like`
- `hash_tensor`
- `is_tensor`
- `ones_like`
- `per_tensor_affine`
- `quasirandom`
- `rand`
- `rand_like`
- `randint`
- `randint_like`
- `randn`
- `randn_like`
- `random`
- `randperm`
- `scalar_tensor`
- `set_default_tensor_type`
- `tensordot`
- `utils.swap_tensors`

---

## Important Priority

### Implementable (233 items)

#### Additional optimizer

- `optim.Adadelta`
- `optim.Adafactor`
- `optim.LBFGS`
- `optim.Muon`
- `optim.Rprop`
- `optim.swa_utils`

#### Automatic mixed precision training

- `GradScaler`
- `amp`
- `autocast`
- `autocast_decrement_nesting`
- `autocast_increment_nesting`
- `clamp`
- `clamp_`
- `clear_autocast_cache`
- `get_autocast_cpu_dtype`
- `get_autocast_dtype`
- `get_autocast_gpu_dtype`
- `grid_sampler`
- `grid_sampler_2d`
- `grid_sampler_3d`
- `is_autocast_cache_enabled`
- `is_autocast_cpu_enabled`
- `is_autocast_enabled`
- `set_autocast_cache_enabled`
- `set_autocast_cpu_dtype`
- `set_autocast_cpu_enabled`
- `set_autocast_dtype`
- `set_autocast_enabled`
- `set_autocast_gpu_dtype`

#### Commonly used neural network layer

- `nn.AdaptiveAvgPool1d`
- `nn.AdaptiveAvgPool2d`
- `nn.AdaptiveAvgPool3d`
- `nn.AvgPool1d`
- `nn.AvgPool2d`
- `nn.AvgPool3d`
- `nn.CELU`
- `nn.ELU`
- `nn.GELU`
- `nn.GroupNorm`
- `nn.SELU`
- `nn.SiLU`
- `nn.Transformer`
- `nn.TransformerDecoder`
- `nn.TransformerDecoderLayer`
- `nn.TransformerEncoder`
- `nn.TransformerEncoderLayer`
- `nn.Upsample`

#### Fast Fourier Transform operation

- `fft`

#### Functional API for neural network operations

- `nn.functional.BroadcastingList1`
- `nn.functional.BroadcastingList2`
- `nn.functional.BroadcastingList3`
- `nn.functional.Callable`
- `nn.functional.GRID_SAMPLE_INTERPOLATION_MODES`
- `nn.functional.Optional`
- `nn.functional.Tensor`
- `nn.functional.Union`
- `nn.functional.adaptive_avg_pool1d`
- `nn.functional.adaptive_avg_pool2d`
- `nn.functional.adaptive_avg_pool3d`
- `nn.functional.affine_grid`
- `nn.functional.alpha_dropout`
- `nn.functional.assert_int_or_pair`
- `nn.functional.avg_pool1d`
- `nn.functional.avg_pool3d`
- `nn.functional.batch_norm`
- `nn.functional.bilinear`
- `nn.functional.boolean_dispatch`
- `nn.functional.celu`
- `nn.functional.celu_`
- `nn.functional.channel_shuffle`
- `nn.functional.conv1d`
- `nn.functional.conv2d`
- `nn.functional.conv3d`
- `nn.functional.conv_tbc`
- `nn.functional.dropout1d`
- `nn.functional.dropout2d`
- `nn.functional.dropout3d`
- `nn.functional.elu_`
- `nn.functional.embedding`
- `nn.functional.embedding_bag`
- `nn.functional.feature_alpha_dropout`
- `nn.functional.fold`
- `nn.functional.glu`
- `nn.functional.grad`
- `nn.functional.grid_sample`
- `nn.functional.group_norm`
- `nn.functional.handle_torch_function`
- `nn.functional.hardshrink`
- `nn.functional.hardswish`
- `nn.functional.has_torch_function`
- `nn.functional.has_torch_function_unary`
- `nn.functional.interpolate`
- `nn.functional.linear`
- `nn.functional.local_response_norm`
- `nn.functional.lp_pool1d`
- `nn.functional.lp_pool2d`
- `nn.functional.lp_pool3d`
- `nn.functional.mish`
- ... and 24 more

#### Learning rate scheduler

- `optim.lr_scheduler.Any`
- `optim.lr_scheduler.ChainedScheduler`
- `optim.lr_scheduler.Counter`
- `optim.lr_scheduler.CyclicLR`
- `optim.lr_scheduler.LRScheduler`
- `optim.lr_scheduler.LambdaLR`
- `optim.lr_scheduler.LinearLR`
- `optim.lr_scheduler.Literal`
- `optim.lr_scheduler.OneCycleLR`
- `optim.lr_scheduler.Optional`
- `optim.lr_scheduler.PolynomialLR`
- `optim.lr_scheduler.ReduceLROnPlateau`
- `optim.lr_scheduler.Self`
- `optim.lr_scheduler.SequentialLR`
- `optim.lr_scheduler.StepLR`
- `optim.lr_scheduler.SupportsFloat`
- `optim.lr_scheduler.Tensor`
- `optim.lr_scheduler.Union`
- `optim.lr_scheduler.annotations`
- `optim.lr_scheduler.bisect_right`
- `optim.lr_scheduler.cast`
- `optim.lr_scheduler.inf`
- `optim.lr_scheduler.override`
- `optim.lr_scheduler.partial`
- `optim.lr_scheduler.ref`
- `optim.lr_scheduler.wraps`

#### Linear algebra operation

- `batch_norm`
- `batch_norm_elemt`
- `batch_norm_gather_stats`
- `batch_norm_gather_stats_with_counts`
- `batch_norm_stats`
- `batch_norm_update_stats`
- `celu`
- `celu_`
- `cholesky`
- `cholesky_inverse`
- `cholesky_solve`
- `convolution`
- `det`
- `detach`
- `detach_`
- `detach_copy`
- `eig`
- `embedding_renorm_`
- `frobenius_norm`
- `geqrf`
- `group_norm`
- `inverse`
- `kthvalue`
- `linalg`
- `lu`
- `lu_solve`
- `lu_unpack`
- `native_batch_norm`
- `native_group_norm`
- `native_layer_norm`
- `native_norm`
- `norm`
- `norm_except_dim`
- `normal`
- `nuclear_norm`
- `orgqr`
- `ormqr`
- `pinverse`
- `qr`
- `renorm`
- `rms_norm`
- `selu`
- `selu_`
- `set_flush_denormal`
- `slice_inverse`
- `svd`
- `svd_lowrank`
- `symeig`

#### Neural network utilities

- `nn.UninitializedBuffer`
- `nn.init`
- `nn.modules`
- `nn.parameter`
- `nn.utils`

#### Sparse tensor operations - storage abstraction exists

- `ccol_indices_copy`
- `col_indices_copy`
- `crow_indices_copy`
- `indices_copy`
- `max_pool1d_with_indices`
- `nn.functional.adaptive_max_pool1d_with_indices`
- `nn.functional.adaptive_max_pool2d_with_indices`
- `nn.functional.adaptive_max_pool3d_with_indices`
- `nn.functional.fractional_max_pool2d_with_indices`
- `nn.functional.fractional_max_pool3d_with_indices`
- `nn.functional.max_pool1d_with_indices`
- `nn.functional.max_pool2d_with_indices`
- `nn.functional.max_pool3d_with_indices`
- `nn.functional.sparse_support_notes`
- `optim.SparseAdam`
- `resize_as_sparse_`
- `row_indices_copy`
- `sparse`
- `sparse_bsc`
- `sparse_bsc_tensor`
- `sparse_bsr`
- `sparse_bsr_tensor`
- `sparse_compressed_tensor`
- `sparse_coo`
- `sparse_coo_tensor`
- `sparse_csc`
- `sparse_csc_tensor`
- `sparse_csr`
- `sparse_csr_tensor`
- `tril_indices`
- `triu_indices`
- `values_copy`

### Architectural (74 items)

#### Requires JIT compilation infrastructure

- `Block`
- `FunctionSchema`
- `JITException`
- `LiteScriptModule`
- `ScriptClass`
- `ScriptClassFunction`
- `ScriptDict`
- `ScriptDictIterator`
- `ScriptDictKeyIterator`
- `ScriptFunction`
- `ScriptList`
- `ScriptListIterator`
- `ScriptMethod`
- `ScriptModule`
- `ScriptModuleSerializer`
- `ScriptObject`
- `ScriptObjectProperty`
- `compile`
- `compiled_with_cxx11_abi`
- `compiler`
- `jit`

#### Requires distributed training infrastructure

- `RRefType`
- `distributed`
- `futures`
- `is_distributed`
- `multiprocessing`
- `nn.DataParallel`
- `nn.parallel`
- `prepare_multiprocessing_environment`

#### Requires quantization infrastructure

- `QInt32Storage`
- `QInt8Storage`
- `QUInt2x4Storage`
- `QUInt4x2Storage`
- `QUInt8Storage`
- `dequantize`
- `empty_quantized`
- `fake_quantize_per_channel_affine`
- `fake_quantize_per_tensor_affine`
- `fbgemm_linear_fp16_weight`
- `fbgemm_linear_fp16_weight_fp32_activation`
- `fbgemm_linear_int8_weight`
- `fbgemm_linear_int8_weight_fp32_activation`
- `fbgemm_linear_quantize_weight`
- `fbgemm_pack_gemm_matrix_fp16`
- `fbgemm_pack_quantized_matrix`
- `int_repr`
- `nn.qat`
- `nn.quantizable`
- `nn.quantized`
- `q_per_channel_axis`
- `q_per_channel_scales`
- `q_per_channel_zero_points`
- `q_scale`
- `q_zero_point`
- `qint32`
- `qint8`
- `qscheme`
- `quantization`
- `quantize_per_channel`
- `quantize_per_tensor`
- `quantize_per_tensor_dynamic`
- `quantized_batch_norm`
- `quantized_gru`
- `quantized_gru_cell`
- `quantized_lstm`
- `quantized_lstm_cell`
- `quantized_max_pool1d`
- `quantized_max_pool2d`
- `quantized_max_pool3d`
- `quantized_rnn_relu_cell`
- `quantized_rnn_tanh_cell`
- `quint2x4`
- `quint4x2`
- `quint8`

---

## Optional Priority

### Implementable (493 items)

#### Additional functionality

- `AVG`
- `AcceleratorError`
- `AliasDb`
- `Argument`
- `ArgumentSpec`
- `BFloat16Tensor`
- `BoolTensor`
- `BufferDict`
- `ByteTensor`
- `Capsule`
- `CharTensor`
- `CompleteArgumentSpec`
- `DisableTorchFunction`
- `DoubleTensor`
- `Event`
- `ExecutionPlan`
- `FloatTensor`
- `Future`
- `Generator`
- `HalfTensor`
- `IODescriptor`
- `IntTensor`
- `LongTensor`
- `ModuleDict`
- `OperatorInfo`
- `OutOfMemoryError`
- `ParameterDict`
- `PyTorchFileReader`
- `PyTorchFileWriter`
- `ShortTensor`
- `Size`
- `StaticModule`
- `Stream`
- `Tag`
- `Use`
- `adaptive_avg_pool1d`
- `adjoint`
- `affine_grid_generator`
- `alias_copy`
- `all`
- `alpha_dropout`
- `alpha_dropout_`
- `angle`
- `any`
- `ao`
- `argsort`
- `argwhere`
- `as_strided`
- `as_strided_`
- `as_strided_copy`
- ... and 365 more

#### Probability distribution

- `distributions`

#### Profiling and benchmarking tools

- `BenchmarkConfig`
- `BenchmarkExecutionStats`
- `ThroughputBenchmark`
- `profiler`
- `utils.ThroughputBenchmark`
- `utils.throughput_benchmark`

#### Signal processing operation

- `istft`
- `signal`
- `stft`

#### Special mathematical function

- `erf`
- `erf_`
- `erfc`
- `erfc_`
- `erfinv`
- `i0`
- `i0_`
- `special`

#### Specialized neural network layer

- `nn.BatchNorm3d`
- `nn.Bilinear`
- `nn.Buffer`
- `nn.ChannelShuffle`
- `nn.CircularPad1d`
- `nn.CircularPad2d`
- `nn.CircularPad3d`
- `nn.Container`
- `nn.Flatten`
- `nn.Fold`
- `nn.GLU`
- `nn.Hardshrink`
- `nn.Hardswish`
- `nn.Identity`
- `nn.LPPool1d`
- `nn.LPPool2d`
- `nn.LPPool3d`
- `nn.LazyBatchNorm3d`
- `nn.LocalResponseNorm`
- `nn.Mish`
- `nn.PixelShuffle`
- `nn.PixelUnshuffle`
- `nn.RMSNorm`
- `nn.ReflectionPad1d`
- `nn.ReflectionPad2d`
- `nn.ReflectionPad3d`
- `nn.Softplus`
- `nn.Softshrink`
- `nn.Softsign`
- `nn.SyncBatchNorm`
- `nn.Threshold`
- `nn.Unflatten`
- `nn.Unfold`
- `nn.UpsamplingBilinear2d`
- `nn.UpsamplingNearest2d`
- `nn.ZeroPad1d`
- `nn.ZeroPad2d`
- `nn.ZeroPad3d`
- `nn.attention`
- `nn.factory_kwargs`
- `nn.grad`
- `nn.intrinsic`

#### Testing utilities

- `testing`

#### Utility functions

- `utils.backcompat`
- `utils.backend_registration`
- `utils.checkpoint`
- `utils.cmake_prefix_path`
- `utils.collect_env`
- `utils.copyreg`
- `utils.cpp_backtrace`
- `utils.data`
- `utils.dlpack`
- `utils.generate_methods_for_privateuse1_backend`
- `utils.get_cpp_backtrace`
- `utils.hooks`
- `utils.rename_privateuse1_backend`
- `utils.set_module`
- `utils.torch`
- `utils.weak`
- `utils.weakref`

### Architectural (97 items)

#### Backend-specific functionality

- `accelerator`
- `cudnn_affine_grid_generator`
- `cudnn_batch_norm`
- `cudnn_convolution`
- `cudnn_convolution_add_relu`
- `cudnn_convolution_relu`
- `cudnn_grid_sampler`
- `cudnn_is_acceptable`
- `flipud`
- `get_autocast_ipu_dtype`
- `get_autocast_xla_dtype`
- `is_autocast_ipu_enabled`
- `is_autocast_xla_enabled`
- `is_vulkan_available`
- `miopen_batch_norm`
- `miopen_convolution`
- `miopen_convolution_add_relu`
- `miopen_convolution_relu`
- `miopen_depthwise_convolution`
- `miopen_rnn`
- `mkldnn_adaptive_avg_pool2d`
- `mkldnn_convolution`
- `mkldnn_linear_backward_weights`
- `mkldnn_max_pool2d`
- `mkldnn_max_pool3d`
- `mkldnn_rnn_layer`
- `mps`
- `mtia`
- `set_autocast_ipu_dtype`
- `set_autocast_ipu_enabled`
- `set_autocast_xla_dtype`
- `set_autocast_xla_enabled`
- `xpu`

#### Model hub functionality

- `hub`

#### Storage abstraction already exists in Coeus

- `BFloat16Storage`
- `BoolStorage`
- `ByteStorage`
- `CharStorage`
- `ComplexDoubleStorage`
- `ComplexFloatStorage`
- `DoubleStorage`
- `FloatStorage`
- `HalfStorage`
- `IntStorage`
- `LongStorage`
- `ShortStorage`
- `Storage`
- `StorageBase`
- `TypedStorage`
- `UntypedStorage`

#### Symbolic shape inference for compilation

- `SymBool`
- `SymFloat`
- `SymInt`
- `sym_constrain_range`
- `sym_constrain_range_for_size`
- `sym_float`
- `sym_fresh_size`
- `sym_int`
- `sym_ite`
- `sym_max`
- `sym_min`
- `sym_not`
- `sym_sqrt`
- `sym_sum`

#### Type system for JIT/tracing

- `AggregationType`
- `AnyType`
- `AwaitType`
- `BoolType`
- `ClassType`
- `ComplexType`
- `ConcreteModuleType`
- `ConcreteModuleTypeBuilder`
- `DeviceObjType`
- `DictType`
- `EnumType`
- `FloatType`
- `FutureType`
- `InferredType`
- `IntType`
- `InterfaceType`
- `ListType`
- `NoneType`
- `NumberType`
- `OptionalType`
- `PyObjectType`
- `StreamObjType`
- `StringType`
- `SymBoolType`
- `SymIntType`
- `TensorType`
- `TupleType`
- `Type`
- `UnionType`
- `nn.functional.DType`
- `nn.functional.ScalingType`
- `nn.functional.SwizzleType`
- `optim.lr_scheduler.TypedDict`

### Internal (141 items)

#### Internal API not needed for user-facing functionality

- `CallStack`
- `Code`
- `CompilationUnit`
- `DeepCopyMemoTable`
- `DeserializationStorageContext`
- `DispatchKey`
- `DispatchKeySet`
- `ErrorReport`
- `ExcludeDispatchKeyGuard`
- `FatalError`
- `FileCheck`
- `Graph`
- `GraphExecutorState`
- `LockingLogger`
- `LoggerBase`
- `Node`
- `NoopLogger`
- `PRIVATE_OPS`
- `SerializationStorageContext`
- `TYPE_CHECKING`
- `TracingState`
- `USE_GLOBAL_DEPS`
- `USE_RTLD_GLOBAL_WITH_LIBTORCH`
- `Value`
- `acos`
- `acos_`
- `acosh`
- `acosh_`
- `allclose`
- `arccos`
- `arccos_`
- `arccosh`
- `arccosh_`
- `binary_cross_entropy_with_logits`
- `builtins`
- `choose_qparams_optimized`
- `classproperty`
- `conv_transpose1d`
- `conv_transpose2d`
- `conv_transpose3d`
- `cos`
- `cos_`
- `cosh`
- `cosh_`
- `cosine_embedding_loss`
- `cosine_similarity`
- `cross`
- `ctc_loss`
- `ctypes`
- `cudnn_convolution_transpose`
- ... and 91 more

---

