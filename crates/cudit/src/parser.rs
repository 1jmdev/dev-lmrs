use crate::model::{Kernel, Param};

pub(crate) fn parse_kernels(src: &str) -> Result<Vec<Kernel>, (String, String)> {
    let mut kernels = Vec::new();
    for (name, params_str) in find_kernels(src) {
        match parse_params(&params_str) {
            Ok(params) => kernels.push(Kernel { name, params }),
            Err(message) => return Err((name, message)),
        }
    }
    Ok(kernels)
}

pub(crate) fn parse_native_functions(src: &str) -> Result<Vec<Kernel>, (String, String)> {
    let mut functions = Vec::new();
    for (name, params_str) in find_native_functions(src) {
        match parse_params(&params_str) {
            Ok(params) => functions.push(Kernel { name, params }),
            Err(message) => return Err((name, message)),
        }
    }
    Ok(functions)
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

fn find_native_functions(src: &str) -> Vec<(String, String)> {
    let src = strip_comments(src);
    let mut res = Vec::new();
    let mut from = 0;

    while let Some(rel) = src[from..].find("extern \"C\"") {
        let after_extern = from + rel + "extern \"C\"".len();
        let rest = src[after_extern..].trim_start();
        if rest.starts_with("__global__") {
            from = after_extern;
            continue;
        }
        if !(rest.starts_with("int ") || rest.starts_with("void ")) {
            from = after_extern;
            continue;
        }
        if let Some(prel) = src[after_extern..].find('(') {
            let open = after_extern + prel;
            let before = src[after_extern..open].trim_end();
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
        from = after_extern;
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
            | "cudaStream_t"
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
        "half" => "half::f16",
        "void" => "std::ffi::c_void",
        "cudaStream_t" => "std::ffi::c_void",
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

        out.push(Param {
            name,
            rust_ty: rust_base.to_string(),
            is_ptr,
            is_const,
            base_ty: rust_base.to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_global_kernel() {
        let kernels = parse_kernels(
            r#"extern "C" __global__ void add(const float* a, float* out, int n) {}"#,
        )
        .unwrap();
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "add");
        assert_eq!(kernels[0].params.len(), 3);
    }
}
