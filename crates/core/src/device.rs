use std::sync::Arc;

use cudarc::driver::{CudaGraph as RawCudaGraph, DriverError, sys};

use crate::{Result, Tensor, TensorElement};
use lmrs_kernels::Runtime;

pub type CudaGraph = RawCudaGraph;

#[derive(Clone)]
pub struct CudaDevice {
    inner: Arc<DeviceInner>,
}

pub(crate) struct DeviceInner {
    ordinal: usize,
    runtime: Runtime,
}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(DeviceInner {
                ordinal,
                runtime: Runtime::new(ordinal)?,
            }),
        })
    }

    pub fn default() -> Result<Self> {
        Self::new(0)
    }

    pub fn ordinal(&self) -> usize {
        self.inner.ordinal
    }

    pub fn synchronize(&self) -> Result<()> {
        self.inner.runtime.synchronize()?;
        Ok(())
    }

    pub fn kernels(&self) -> &Runtime {
        self.inner.runtime()
    }

    pub fn upload<T: TensorElement>(
        &self,
        data: &[T],
        shape: impl Into<crate::Shape>,
    ) -> Result<Tensor<T>> {
        Tensor::from_slice(self, data, shape)
    }

    pub fn zeros<T: TensorElement>(&self, shape: impl Into<crate::Shape>) -> Result<Tensor<T>> {
        Tensor::zeros(self, shape)
    }

    pub fn capture_graph<F>(&self, f: F) -> Result<CudaGraph>
    where
        F: FnOnce() -> Result<()>,
    {
        let mut err = None;
        let graph = self.inner.runtime.capture_graph(|_| match f() {
            Ok(()) => Ok(()),
            Err(e) => {
                err = Some(e);
                Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))
            }
        });
        match (graph, err) {
            (Ok(graph), None) => Ok(graph),
            (Err(err), None) => Err(err.into()),
            (_, Some(err)) => Err(err),
        }
    }

    pub(crate) fn inner(&self) -> &Arc<DeviceInner> {
        &self.inner
    }

    pub(crate) fn from_inner(inner: Arc<DeviceInner>) -> Self {
        Self { inner }
    }
}

impl DeviceInner {
    pub(crate) fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.ordinal
    }
}
