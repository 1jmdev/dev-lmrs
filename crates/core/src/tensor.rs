use std::sync::Arc;

use crate::device::{CudaDevice, DeviceInner};
use crate::{DType, Error, Result, Shape, Strides, TensorElement};
use lmrs_kernels::{Runtime, Tensor as LmrsTensor};

pub struct Tensor<T: TensorElement> {
    storage: LmrsTensor<T>,
    shape: Shape,
    strides: Strides,
    device: Arc<DeviceInner>,
}

impl<T: TensorElement> Tensor<T> {
    pub fn from_slice(device: &CudaDevice, data: &[T], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.elem_count()?;
        if expected != data.len() {
            return Err(Error::ElementCountMismatch {
                shape,
                expected,
                actual: data.len(),
            });
        }
        let storage = device.inner().runtime().upload(data)?;
        Ok(Self::from_storage(device.inner().clone(), storage, shape))
    }

    pub fn zeros(device: &CudaDevice, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let len = shape.elem_count()?;
        Self::check_kernel_len(len)?;
        let storage = device.inner().runtime().zeros::<T>(len)?;
        Ok(Self::from_storage(device.inner().clone(), storage, shape))
    }

    pub fn download(&self) -> Result<Vec<T>> {
        Ok(self.device.runtime().download(&self.storage)?)
    }

    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        if self.len() != data.len() {
            return Err(Error::ElementCountMismatch {
                shape: self.shape.clone(),
                expected: self.len(),
                actual: data.len(),
            });
        }
        self.device
            .runtime()
            .copy_from_slice(data, &mut self.storage)?;
        Ok(())
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn device(&self) -> CudaDevice {
        CudaDevice::from_inner(self.device.clone())
    }

    pub fn device_ordinal(&self) -> usize {
        self.device.ordinal()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.contiguous_strides()
    }

    pub fn reshape(mut self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.elem_count()?;
        if expected != self.len() {
            return Err(Error::ElementCountMismatch {
                shape,
                expected,
                actual: self.len(),
            });
        }
        self.strides = shape.contiguous_strides();
        self.shape = shape;
        Ok(self)
    }

    pub fn same_shape(&self, other: &Self) -> Result<()> {
        self.shape.require_same(&other.shape)
    }

    pub(crate) fn raw(&self) -> &LmrsTensor<T> {
        &self.storage
    }

    pub(crate) fn raw_mut(&mut self) -> &mut LmrsTensor<T> {
        &mut self.storage
    }

    pub fn as_kernel_tensor(&self) -> &LmrsTensor<T> {
        &self.storage
    }

    pub fn as_kernel_tensor_mut(&mut self) -> &mut LmrsTensor<T> {
        &mut self.storage
    }

    pub(crate) fn runtime(&self) -> &Runtime {
        self.device.runtime()
    }

    pub(crate) fn device_inner(&self) -> Arc<DeviceInner> {
        self.device.clone()
    }

    pub(crate) fn ensure_compatible(&self, other: &Self) -> Result<()> {
        self.same_shape(other)?;
        self.ensure_same_device(other)?;
        self.ensure_contiguous()?;
        other.ensure_contiguous()?;
        Ok(())
    }

    pub(crate) fn ensure_same_device<U: TensorElement>(&self, other: &Tensor<U>) -> Result<()> {
        let expected = self.device_ordinal();
        let actual = other.device_ordinal();
        if expected == actual {
            Ok(())
        } else {
            Err(Error::DeviceMismatch { expected, actual })
        }
    }

    pub(crate) fn ensure_contiguous(&self) -> Result<()> {
        if self.is_contiguous() {
            Ok(())
        } else {
            Err(Error::NonContiguous)
        }
    }

    pub(crate) fn empty_like(&self) -> Result<Self> {
        Self::zeros(&self.device(), self.shape.clone())
    }

    pub(crate) fn from_storage(
        device: Arc<DeviceInner>,
        storage: LmrsTensor<T>,
        shape: Shape,
    ) -> Self {
        let strides = shape.contiguous_strides();
        Self {
            storage,
            shape,
            strides,
            device,
        }
    }

    pub(crate) fn check_kernel_len(len: usize) -> Result<()> {
        if len <= i32::MAX as usize {
            Ok(())
        } else {
            Err(Error::KernelElementLimit {
                actual: len,
                max: i32::MAX as usize,
            })
        }
    }
}

impl<T: TensorElement> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("device", &format_args!("cuda:{}", self.device_ordinal()))
            .finish_non_exhaustive()
    }
}
