use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    F16,
    F32,
}

pub trait TensorElement: DeviceRepr + ValidAsZeroBits + Copy + Send + Sync + 'static {
    const DTYPE: DType;
}

impl TensorElement for half::f16 {
    const DTYPE: DType = DType::F16;
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;
}
