use smallvec::{SmallVec, smallvec};

use crate::{Error, Result};

pub type Strides = SmallVec<[usize; 6]>;

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Shape {
    dims: SmallVec<[usize; 6]>,
}

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self {
            dims: dims.into_iter().collect(),
        }
    }

    pub fn scalar() -> Self {
        Self { dims: smallvec![] }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dim(&self, dim: usize) -> Result<usize> {
        self.dims.get(dim).copied().ok_or(Error::InvalidDim {
            dim,
            reason: "dimension is out of bounds",
        })
    }

    pub fn last_dim(&self) -> Result<usize> {
        self.dims.last().copied().ok_or(Error::InvalidDim {
            dim: 0,
            reason: "scalar shape has no last dimension",
        })
    }

    pub fn elem_count(&self) -> Result<usize> {
        self.dims.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::ElementCountOverflow(self.clone()))
        })
    }

    pub fn contiguous_strides(&self) -> Strides {
        let mut strides = SmallVec::with_capacity(self.rank());
        let mut stride = 1usize;
        for &dim in self.dims.iter().rev() {
            strides.push(stride);
            stride = stride.saturating_mul(dim);
        }
        strides.reverse();
        strides
    }

    pub fn require_rank(&self, expected: usize) -> Result<()> {
        let actual = self.rank();
        if actual == expected {
            Ok(())
        } else {
            Err(Error::RankMismatch {
                expected,
                actual,
                shape: self.clone(),
            })
        }
    }

    pub fn require_same(&self, other: &Self) -> Result<()> {
        if self == other {
            Ok(())
        } else {
            Err(Error::ShapeMismatch {
                expected: self.clone(),
                actual: other.clone(),
            })
        }
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.dims.iter()).finish()
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self::new(value)
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self::new(value.iter().copied())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self::new(value)
    }
}
