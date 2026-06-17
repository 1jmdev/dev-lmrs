use half::f16;

use cudarc::driver::DriverError;
use lmrs_kernels::{Runtime, Tensor as LmrsTensor};
use std::result;

use crate::{CudaDevice, Error, Result, Shape, Tensor};

impl Tensor<f16> {
    pub fn full(device: &CudaDevice, shape: impl Into<Shape>, value: f32) -> Result<Self> {
        let mut out = Self::zeros(device, shape)?;
        out.fill_(value)?;
        Ok(out)
    }

    pub fn fill_(&mut self, value: f32) -> Result<()> {
        self.ensure_contiguous()?;
        Self::check_kernel_len(self.len())?;
        let device = self.device_inner();
        device.runtime().fp16_fill(self.raw_mut(), value)?;
        Ok(())
    }

    pub fn copy(&self) -> Result<Self> {
        self.ensure_contiguous()?;
        let mut out = self.empty_like()?;
        self.copy_into(&mut out)?;
        Ok(out)
    }

    pub fn copy_into(&self, out: &mut Self) -> Result<()> {
        self.ensure_output(out)?;
        self.runtime().fp16_copy(out.raw_mut(), self.raw())?;
        Ok(())
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, |rt, out, a, b| rt.fp16_add(out, a, b))
    }

    pub fn add_into(&self, rhs: &Self, out: &mut Self) -> Result<()> {
        self.binary_into(rhs, out, |rt, out, a, b| rt.fp16_add(out, a, b))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, |rt, out, a, b| rt.fp16_sub(out, a, b))
    }

    pub fn sub_into(&self, rhs: &Self, out: &mut Self) -> Result<()> {
        self.binary_into(rhs, out, |rt, out, a, b| rt.fp16_sub(out, a, b))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, |rt, out, a, b| rt.fp16_mul(out, a, b))
    }

    pub fn mul_into(&self, rhs: &Self, out: &mut Self) -> Result<()> {
        self.binary_into(rhs, out, |rt, out, a, b| rt.fp16_mul(out, a, b))
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, |rt, out, a, b| rt.fp16_div(out, a, b))
    }

    pub fn div_into(&self, rhs: &Self, out: &mut Self) -> Result<()> {
        self.binary_into(rhs, out, |rt, out, a, b| rt.fp16_div(out, a, b))
    }

    pub fn add_scalar(&self, value: f32) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_add_scalar(out, x, value))
    }

    pub fn add_scalar_into(&self, value: f32, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_add_scalar(out, x, value))
    }

    pub fn mul_scalar(&self, value: f32) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_mul_scalar(out, x, value))
    }

    pub fn mul_scalar_into(&self, value: f32, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_mul_scalar(out, x, value))
    }

    pub fn relu(&self) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_relu(out, x))
    }

    pub fn relu_into(&self, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_relu(out, x))
    }

    pub fn silu(&self) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_silu(out, x))
    }

    pub fn silu_into(&self, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_silu(out, x))
    }

    pub fn gelu(&self) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_gelu(out, x))
    }

    pub fn gelu_into(&self, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_gelu(out, x))
    }

    pub fn exp(&self) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_exp(out, x))
    }

    pub fn exp_into(&self, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_exp(out, x))
    }

    pub fn tanh(&self) -> Result<Self> {
        self.unary(|rt, out, x| rt.fp16_tanh(out, x))
    }

    pub fn tanh_into(&self, out: &mut Self) -> Result<()> {
        self.unary_into(out, |rt, out, x| rt.fp16_tanh(out, x))
    }

    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let mut out = self.empty_like()?;
        self.rms_norm_into(weight, eps, &mut out)?;
        Ok(out)
    }

    pub fn rms_norm_into(&self, weight: &Self, eps: f32, out: &mut Self) -> Result<()> {
        self.ensure_same_device(weight)?;
        self.ensure_output(out)?;
        weight.ensure_contiguous()?;
        let width = self.shape().last_dim()?;
        if weight.len() != width {
            return Err(Error::ElementCountMismatch {
                shape: weight.shape().clone(),
                expected: width,
                actual: weight.len(),
            });
        }
        let width = checked_i32(width, "normalization width")?;
        self.runtime()
            .fp16_rms_norm(out.raw_mut(), self.raw(), weight.raw(), width, eps)?;
        Ok(())
    }

    pub fn layer_norm(&self, weight: &Self, bias: &Self, eps: f32) -> Result<Self> {
        let mut out = self.empty_like()?;
        self.layer_norm_into(weight, bias, eps, &mut out)?;
        Ok(out)
    }

    pub fn layer_norm_into(
        &self,
        weight: &Self,
        bias: &Self,
        eps: f32,
        out: &mut Self,
    ) -> Result<()> {
        self.ensure_same_device(weight)?;
        self.ensure_same_device(bias)?;
        self.ensure_output(out)?;
        weight.ensure_contiguous()?;
        bias.ensure_contiguous()?;
        weight.same_shape(bias)?;
        let width = self.shape().last_dim()?;
        if weight.len() != width {
            return Err(Error::ElementCountMismatch {
                shape: weight.shape().clone(),
                expected: width,
                actual: weight.len(),
            });
        }
        let width = checked_i32(width, "normalization width")?;
        self.runtime().fp16_layer_norm(
            out.raw_mut(),
            self.raw(),
            weight.raw(),
            bias.raw(),
            width,
            eps,
        )?;
        Ok(())
    }

    pub fn softmax_last_dim(&self, scale: f32) -> Result<Self> {
        let mut out = self.empty_like()?;
        self.softmax_last_dim_into(scale, &mut out)?;
        Ok(out)
    }

    pub fn softmax_last_dim_into(&self, scale: f32, out: &mut Self) -> Result<()> {
        self.ensure_output(out)?;
        let width = checked_i32(self.shape().last_dim()?, "softmax width")?;
        self.runtime()
            .fp16_softmax_rows(out.raw_mut(), self.raw(), width, scale)?;
        Ok(())
    }

    pub fn causal_mask(&self) -> Result<Self> {
        let mut out = self.empty_like()?;
        self.causal_mask_into(&mut out)?;
        Ok(out)
    }

    pub fn causal_mask_into(&self, out: &mut Self) -> Result<()> {
        self.ensure_output(out)?;
        if self.shape().rank() < 2 {
            return Err(Error::RankMismatch {
                expected: 2,
                actual: self.shape().rank(),
                shape: self.shape().clone(),
            });
        }
        let dims = self.shape().dims();
        let query_len = checked_i32(dims[dims.len() - 2], "query length")?;
        let key_len = checked_i32(dims[dims.len() - 1], "key length")?;
        self.runtime()
            .fp16_causal_mask(out.raw_mut(), self.raw(), query_len, key_len)?;
        Ok(())
    }

    pub fn rope(&self, seq_len: usize, head_dim: usize, theta: f32) -> Result<Self> {
        self.ensure_contiguous()?;
        if self.len() % head_dim != 0 {
            return Err(Error::InvalidDim {
                dim: head_dim,
                reason: "head_dim must divide tensor length",
            });
        }
        if head_dim & 1 != 0 {
            return Err(Error::InvalidDim {
                dim: head_dim,
                reason: "head_dim must be even for rotary pairs",
            });
        }
        let seq_len = checked_i32(seq_len, "sequence length")?;
        let head_dim = checked_i32(head_dim, "head dimension")?;
        let mut out = self.empty_like()?;
        self.runtime()
            .fp16_rope(out.raw_mut(), self.raw(), seq_len, head_dim, theta)?;
        Ok(out)
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        self.matmul_impl(rhs, false)
    }

    pub fn matmul_rhs_t(&self, rhs: &Self) -> Result<Self> {
        self.matmul_impl(rhs, true)
    }

    pub fn matmul_into(&self, rhs: &Self, out: &mut Self) -> Result<()> {
        self.matmul_into_impl(rhs, false, out)
    }

    pub fn matmul_rhs_t_into(&self, rhs: &Self, out: &mut Self) -> Result<()> {
        self.matmul_into_impl(rhs, true, out)
    }

    fn binary<F>(&self, rhs: &Self, f: F) -> Result<Self>
    where
        F: FnOnce(
            &Runtime,
            &mut LmrsTensor<f16>,
            &LmrsTensor<f16>,
            &LmrsTensor<f16>,
        ) -> result::Result<(), DriverError>,
    {
        self.ensure_compatible(rhs)?;
        Self::check_kernel_len(self.len())?;
        let mut out = self.empty_like()?;
        self.binary_into(rhs, &mut out, f)?;
        Ok(out)
    }

    fn binary_into<F>(&self, rhs: &Self, out: &mut Self, f: F) -> Result<()>
    where
        F: FnOnce(
            &Runtime,
            &mut LmrsTensor<f16>,
            &LmrsTensor<f16>,
            &LmrsTensor<f16>,
        ) -> result::Result<(), DriverError>,
    {
        self.ensure_compatible(rhs)?;
        self.ensure_output(out)?;
        f(self.runtime(), out.raw_mut(), self.raw(), rhs.raw())?;
        Ok(())
    }

    fn unary<F>(&self, f: F) -> Result<Self>
    where
        F: FnOnce(
            &Runtime,
            &mut LmrsTensor<f16>,
            &LmrsTensor<f16>,
        ) -> result::Result<(), DriverError>,
    {
        self.ensure_contiguous()?;
        Self::check_kernel_len(self.len())?;
        let mut out = self.empty_like()?;
        self.unary_into(&mut out, f)?;
        Ok(out)
    }

    fn unary_into<F>(&self, out: &mut Self, f: F) -> Result<()>
    where
        F: FnOnce(
            &Runtime,
            &mut LmrsTensor<f16>,
            &LmrsTensor<f16>,
        ) -> result::Result<(), DriverError>,
    {
        self.ensure_output(out)?;
        f(self.runtime(), out.raw_mut(), self.raw())?;
        Ok(())
    }

    fn matmul_impl(&self, rhs: &Self, rhs_t: bool) -> Result<Self> {
        let (rows, cols, _) = self.matmul_dims(rhs, rhs_t)?;
        let mut out = Self::zeros(&self.device(), Shape::from([rows, cols]))?;
        self.matmul_into_impl(rhs, rhs_t, &mut out)?;
        Ok(out)
    }

    fn matmul_into_impl(&self, rhs: &Self, rhs_t: bool, out: &mut Self) -> Result<()> {
        self.ensure_same_device(rhs)?;
        self.ensure_contiguous()?;
        rhs.ensure_contiguous()?;
        out.ensure_contiguous()?;
        self.ensure_same_device(out)?;
        let (rows, cols, inner) = self.matmul_dims(rhs, rhs_t)?;
        out.shape().require_same(&Shape::from([rows, cols]))?;
        let rows = checked_i32(rows, "matmul rows")?;
        let cols = checked_i32(cols, "matmul cols")?;
        let inner = checked_i32(inner, "matmul inner")?;
        if rhs_t {
            self.runtime().fp16_matmul_nt(
                out.raw_mut(),
                self.raw(),
                rhs.raw(),
                rows,
                cols,
                inner,
            )?;
        } else {
            self.runtime().fp16_matmul_nn(
                out.raw_mut(),
                self.raw(),
                rhs.raw(),
                rows,
                cols,
                inner,
            )?;
        }
        Ok(())
    }

    fn matmul_dims(&self, rhs: &Self, rhs_t: bool) -> Result<(usize, usize, usize)> {
        self.shape().require_rank(2)?;
        rhs.shape().require_rank(2)?;
        let rows = self.shape().dim(0)?;
        let inner = self.shape().dim(1)?;
        let rhs_inner = if rhs_t {
            rhs.shape().dim(1)?
        } else {
            rhs.shape().dim(0)?
        };
        if inner != rhs_inner {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([inner]),
                actual: Shape::from([rhs_inner]),
            });
        }
        let cols = if rhs_t {
            rhs.shape().dim(0)?
        } else {
            rhs.shape().dim(1)?
        };
        Ok((rows, cols, inner))
    }

    fn ensure_output(&self, out: &Self) -> Result<()> {
        self.ensure_same_device(out)?;
        self.shape().require_same(out.shape())?;
        self.ensure_contiguous()?;
        out.ensure_contiguous()?;
        Self::check_kernel_len(self.len())?;
        Ok(())
    }
}

impl Tensor<f32> {
    pub fn full(device: &CudaDevice, shape: impl Into<Shape>, value: f32) -> Result<Self> {
        let mut out = Self::zeros(device, shape)?;
        out.fill_(value)?;
        Ok(out)
    }

    pub fn fill_(&mut self, value: f32) -> Result<()> {
        self.ensure_contiguous()?;
        Self::check_kernel_len(self.len())?;
        let device = self.device_inner();
        device.runtime().fp32_fill(self.raw_mut(), value)?;
        Ok(())
    }
}

fn checked_i32(value: usize, name: &'static str) -> Result<i32> {
    if value <= i32::MAX as usize {
        Ok(value as i32)
    } else {
        Err(Error::InvalidDim {
            dim: value,
            reason: name,
        })
    }
}
