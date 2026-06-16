//! Auto-discovers `kernels/*.cu`, compiles each to PTX via `nvcc`, parses the
//! `extern "C" __global__` signatures, and generates safe host-slice functions.
//!
//! Env knobs:
//!   NVCC              path to nvcc (default: "nvcc")
//!   CUDA_COMPUTE_CAP  e.g. "75" -> adds `-arch=compute_75`
//!
//! NOTE: kernels MUST be declared `extern "C"` so their names aren't mangled.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug)]
struct Param {
    name: String,
    rust_ty: String,
    is_ptr: bool,
    is_const: bool,
    base_ty: String,
}

#[derive(Debug)]
struct Kernel {
    name: String,
    params: Vec<Param>,
}

#[derive(Debug)]
struct KernelFile {
    stem: String,
    ptx_name: String,
    kernels: Vec<Kernel>,
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");

    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let mut cu_files: Vec<PathBuf> = Vec::new();
    if kernels_dir.is_dir() {
        for entry in fs::read_dir(&kernels_dir).expect("failed to read kernels/ dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("cu") {
                cu_files.push(path);
            }
        }
    }
    cu_files.sort();

    let mut files: Vec<KernelFile> = Vec::new();

    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());

        let orig_stem = cu.file_stem().unwrap().to_str().unwrap().to_string();
        let ptx_name = format!("{orig_stem}.ptx");
        let ptx_path = out_dir.join(&ptx_name);

        let mut cmd = Command::new(&nvcc);
        cmd.arg("--ptx").arg(cu).arg("-o").arg(&ptx_path);
        if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
            cmd.arg(format!("-arch=compute_{cap}"));
        }
        let status = cmd.status().unwrap_or_else(|e| {
            panic!("failed to run `{nvcc}` (is the CUDA toolkit on PATH?): {e}");
        });
        if !status.success() {
            panic!("nvcc failed to compile {}", cu.display());
        }

        let src = fs::read_to_string(cu).unwrap();
        let mut kernels = Vec::new();
        for (name, params_str) in find_kernels(&src) {
            match parse_params(&params_str) {
                Ok(params) => kernels.push(Kernel { name, params }),
                Err(why) => {
                    println!(
                        "cargo:warning=skipping kernel `{name}` in {}: {why}",
                        cu.display()
                    );
                }
            }
        }

        files.push(KernelFile {
            stem: sanitize(&orig_stem),
            ptx_name,
            kernels,
        });
    }

    let generated = generate(&files);
    fs::write(out_dir.join("gen_kernels.rs"), generated).unwrap();
}

fn strip_comments(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn matching_paren(s: &str, open: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn find_kernels(src: &str) -> Vec<(String, String)> {
    let src = strip_comments(src);
    let mut res = Vec::new();
    let mut from = 0;

    while let Some(rel) = src[from..].find("__global__") {
        let after = from + rel + "__global__".len();
        if let Some(prel) = src[after..].find('(') {
            let open = after + prel;
            let before = src[after..open].trim_end();
            let name: String = before
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            if let Some(close) = matching_paren(&src, open) {
                if !name.is_empty() {
                    res.push((name, src[open + 1..close].to_string()));
                }
                from = close + 1;
                continue;
            }
        }
        from = after;
    }
    res
}

fn split_params(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            '>' | ')' | ']' | '}' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                parts.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_string());
    }
    parts
}

fn is_type_keyword(t: &str) -> bool {
    matches!(
        t,
        "void"
            | "float"
            | "double"
            | "int"
            | "unsigned"
            | "signed"
            | "long"
            | "short"
            | "char"
            | "size_t"
            | "bool"
            | "half"
            | "const"
            | "volatile"
    )
}

fn map_base_type(base: &str) -> Option<&'static str> {
    let norm = base.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(match norm.as_str() {
        "float" => "f32",
        "double" => "f64",
        "int" | "signed int" | "signed" => "i32",
        "unsigned int" | "unsigned" => "u32",
        "long" | "long int" | "signed long" => "i64",
        "unsigned long" | "unsigned long int" => "u64",
        "long long" | "long long int" => "i64",
        "unsigned long long" | "unsigned long long int" => "u64",
        "short" | "short int" => "i16",
        "unsigned short" | "unsigned short int" => "u16",
        "char" | "signed char" => "i8",
        "unsigned char" => "u8",
        "size_t" => "usize",
        "bool" => "bool",
        _ => return None,
    })
}

fn parse_params(params_str: &str) -> Result<Vec<Param>, String> {
    let trimmed = params_str.trim();
    if trimmed.is_empty() || trimmed == "void" {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for (idx, raw) in split_params(trimmed).into_iter().enumerate() {
        let mut s = raw.clone();
        for kw in ["__restrict__", "__restrict", "restrict", "volatile"] {
            s = s.replace(kw, " ");
        }

        let is_array = s.contains('[');
        if let Some(b) = s.find('[') {
            s.truncate(b);
        }

        let ptr_count = s.matches('*').count();
        let is_ptr = ptr_count > 0 || is_array;
        let is_const = s.split_whitespace().any(|t| t == "const");

        let cleaned = s.replace('*', " ");
        let mut tokens: Vec<String> = cleaned
            .split_whitespace()
            .filter(|t| *t != "const")
            .map(|t| t.to_string())
            .collect();

        if tokens.is_empty() {
            return Err(format!("could not parse parameter `{raw}`"));
        }

        let last_is_name = tokens.len() >= 2 && !is_type_keyword(&tokens[tokens.len() - 1]);
        let name = if last_is_name {
            tokens.pop().unwrap()
        } else {
            format!("arg{idx}")
        };

        let base = tokens.join(" ");
        let rust_base =
            map_base_type(&base).ok_or_else(|| format!("unsupported C type `{base}`"))?;

        let rust_ty = rust_base.to_string();

        out.push(Param {
            name,
            rust_ty,
            is_ptr,
            is_const,
            base_ty: rust_base.to_string(),
        });
    }
    Ok(out)
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn generate(files: &[KernelFile]) -> String {
    let mut s = String::new();
    s.push_str("// @generated by build.rs - do not edit.\n");
    s.push_str(
        "#[allow(dead_code, unused_imports, non_snake_case, clippy::too_many_arguments)]\n\n",
    );
    s.push_str("use std::cell::RefCell;\n");
    s.push_str("use std::sync::Arc;\n");
    s.push_str("use cudarc::driver::{CudaContext, CudaStream, CudaFunction, LaunchConfig, PushKernelArg, DriverError};\n");
    s.push_str("use cudarc::nvrtc::Ptx;\n\n");

    s.push_str("struct Runtime {\n");
    s.push_str("    _ctx: Arc<CudaContext>,\n");
    s.push_str("    pub stream: Arc<CudaStream>,\n");
    for f in files {
        for k in &f.kernels {
            s.push_str(&format!("    func_{}: CudaFunction,\n", k.name));
        }
    }
    s.push_str("}\n\n");

    s.push_str("impl Runtime {\n");
    s.push_str("    fn new(ordinal: usize) -> Result<Self, DriverError> {\n");
    s.push_str("        let ctx = CudaContext::new(ordinal)?;\n");
    s.push_str("        let stream = ctx.default_stream();\n\n");
    for f in files {
        if f.kernels.is_empty() {
            continue;
        }
        s.push_str(&format!(
            "        let module_{m} = ctx.load_module(Ptx::from_src(include_str!(concat!(env!(\"OUT_DIR\"), \"/{ptx}\"))))?;\n",
            m = f.stem,
            ptx = f.ptx_name
        ));
        for k in &f.kernels {
            s.push_str(&format!(
                "        let func_{name} = module_{m}.load_function(\"{name}\")?;\n",
                name = k.name,
                m = f.stem
            ));
        }
        s.push('\n');
    }
    s.push_str("        Ok(Self {\n");
    s.push_str("            _ctx: ctx,\n");
    s.push_str("            stream,\n");
    for f in files {
        for k in &f.kernels {
            s.push_str(&format!("            func_{},\n", k.name));
        }
    }
    s.push_str("        })\n");
    s.push_str("    }\n\n");
    s.push_str("}\n\n");

    s.push_str("thread_local! {\n");
    s.push_str("    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };\n");
    s.push_str("}\n\n");

    s.push_str("fn with_runtime<T>(f: impl FnOnce(&Runtime) -> Result<T, DriverError>) -> Result<T, DriverError> {\n");
    s.push_str("    RUNTIME.with(|cell| {\n");
    s.push_str("        if cell.borrow().is_none() {\n");
    s.push_str("            *cell.borrow_mut() = Some(Runtime::new(0)?);\n");
    s.push_str("        }\n");
    s.push_str("        let rt = cell.borrow();\n");
    s.push_str("        f(rt.as_ref().expect(\"runtime initialized\"))\n");
    s.push_str("    })\n");
    s.push_str("}\n\n");

    s.push_str("pub fn kernel_names() -> &'static [&'static str] {\n    &[");
    let names: Vec<String> = files
        .iter()
        .flat_map(|f| f.kernels.iter().map(|k| format!("\"{}\"", k.name)))
        .collect();
    s.push_str(&names.join(", "));
    s.push_str("]\n}\n\n");

    for f in files {
        for k in &f.kernels {
            let output_params: Vec<&Param> = k
                .params
                .iter()
                .filter(|p| p.is_ptr && !p.is_const && matches!(p.name.as_str(), "out" | "output"))
                .collect();
            let returns_output = output_params.len() == 1;
            let has_n = k.params.iter().any(|p| !p.is_ptr && p.name == "n");

            let mut sig = format!("pub fn {}(", k.name);
            let mut args = Vec::new();
            for p in &k.params {
                if returns_output && p.name == output_params[0].name {
                    continue;
                }
                if p.is_ptr {
                    if p.is_const {
                        args.push(format!("{}: &[{}]", p.name, p.base_ty));
                    } else {
                        args.push(format!("{}: &mut [{}]", p.name, p.base_ty));
                    }
                } else {
                    args.push(format!("{}: {}", p.name, p.rust_ty));
                }
            }
            sig.push_str(&args.join(", "));
            if returns_output {
                sig.push_str(&format!(
                    ") -> Result<Vec<{}>, DriverError> {{\n",
                    output_params[0].base_ty
                ));
            } else {
                sig.push_str(") -> Result<(), DriverError> {\n");
            }
            s.push_str(&sig);

            s.push_str("    with_runtime(|rt| {\n");
            if has_n {
                s.push_str("        let n = n as usize;\n");
                for p in &k.params {
                    if p.is_ptr && !(returns_output && p.name == output_params[0].name) {
                        s.push_str(&format!(
                            "        assert!({}.len() >= n, \"{} length is smaller than n\");\n",
                            p.name, p.name
                        ));
                    }
                }
            }
            for p in &k.params {
                if p.is_ptr {
                    if returns_output && p.name == output_params[0].name {
                        s.push_str(&format!(
                            "        let mut dev_{} = rt.stream.alloc_zeros::<{}>(n)?;\n",
                            p.name, p.base_ty
                        ));
                    } else if p.is_const {
                        s.push_str(&format!(
                            "        let dev_{} = rt.stream.clone_htod({})?;\n",
                            p.name, p.name
                        ));
                    } else {
                        s.push_str(&format!(
                            "        let mut dev_{} = rt.stream.clone_htod({})?;\n",
                            p.name, p.name
                        ));
                    }
                }
            }
            if has_n {
                s.push_str("        let cfg = LaunchConfig::for_num_elems(n as u32);\n");
            } else {
                s.push_str("        let cfg = LaunchConfig::for_num_elems(1);\n");
            }
            s.push_str(&format!(
                "        let mut builder = rt.stream.launch_builder(&rt.func_{});\n",
                k.name
            ));
            for p in &k.params {
                if p.is_ptr {
                    if p.is_const {
                        s.push_str(&format!("        builder.arg(&dev_{});\n", p.name));
                    } else {
                        s.push_str(&format!("        builder.arg(&mut dev_{});\n", p.name));
                    }
                } else {
                    s.push_str(&format!("        builder.arg(&{});\n", p.name));
                }
            }
            s.push_str("        unsafe { builder.launch(cfg) }?;\n");
            if returns_output {
                s.push_str(&format!(
                    "        rt.stream.clone_dtoh(&dev_{})\n",
                    output_params[0].name
                ));
            } else {
                for p in &k.params {
                    if p.is_ptr && !p.is_const {
                        s.push_str(&format!(
                            "        let host_{} = rt.stream.clone_dtoh(&dev_{})?;\n",
                            p.name, p.name
                        ));
                        s.push_str(&format!(
                            "        {}.copy_from_slice(&host_{});\n",
                            p.name, p.name
                        ));
                    }
                }
                s.push_str("        Ok(())\n");
            }
            s.push_str("    })\n");
            s.push_str("}\n\n");
        }
    }
    s
}
