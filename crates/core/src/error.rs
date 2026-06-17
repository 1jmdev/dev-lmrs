use crate::{DType, Shape};
use cudarc::driver::DriverError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cuda driver error: {0:?}")]
    Cuda(DriverError),
    #[error("shape element count overflow for dimensions {0:?}")]
    ElementCountOverflow(Shape),
    #[error("shape {shape:?} requires {expected} values, got {actual}")]
    ElementCountMismatch {
        shape: Shape,
        expected: usize,
        actual: usize,
    },
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Shape, actual: Shape },
    #[error("rank mismatch: expected rank {expected}, got {actual} for shape {shape:?}")]
    RankMismatch {
        expected: usize,
        actual: usize,
        shape: Shape,
    },
    #[error("dtype mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("device mismatch: expected cuda:{expected}, got cuda:{actual}")]
    DeviceMismatch { expected: usize, actual: usize },
    #[error("invalid dimension {dim}: {reason}")]
    InvalidDim { dim: usize, reason: &'static str },
    #[error("operation requires contiguous row-major tensors")]
    NonContiguous,
    #[error("tensor has {actual} elements, maximum supported by current CUDA kernel ABI is {max}")]
    KernelElementLimit { actual: usize, max: usize },
    #[error("unsupported operation for dtype {dtype:?}")]
    UnsupportedDType { dtype: DType },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<DriverError> for Error {
    fn from(value: DriverError) -> Self {
        Self::Cuda(value)
    }
}
