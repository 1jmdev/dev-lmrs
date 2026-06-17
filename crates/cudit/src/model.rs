#[derive(Debug)]
pub(crate) struct Param {
    pub(crate) name: String,
    pub(crate) rust_ty: String,
    pub(crate) is_ptr: bool,
    pub(crate) is_const: bool,
    pub(crate) base_ty: String,
}

#[derive(Debug)]
pub(crate) struct Kernel {
    pub(crate) name: String,
    pub(crate) params: Vec<Param>,
}

#[derive(Debug)]
pub(crate) struct KernelFile {
    pub(crate) stem: String,
    pub(crate) ptx_name: String,
    pub(crate) kernels: Vec<Kernel>,
}

#[derive(Debug)]
pub(crate) struct NativeFile {
    pub(crate) functions: Vec<Kernel>,
}
