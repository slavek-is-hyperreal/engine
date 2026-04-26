// src/loader/gguf.rs
// Minimalny loader GGUF. Czyste Rust std, zero zewnętrznych deps.
// Obsługuje F32 i F16. Dla kwantyzowanych modeli (Q4_K_M):
// tensor dtype będzie > 1 → zwróci UnsupportedDType.
// W takim przypadku pobierz wersję F16: tinyllama...F16.gguf

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    TensorNotFound(String),
    UnsupportedDType(u32),
}
impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self { GgufError::Io(e) }
}
impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "IO: {}", e),
            GgufError::InvalidMagic => write!(f, "Brak nagłówka GGUF"),
            GgufError::UnsupportedVersion(v) => write!(f, "Wersja GGUF {} nieobsługiwana", v),
            GgufError::TensorNotFound(n) => write!(f, "Tensor '{}' nie znaleziony", n),
            GgufError::UnsupportedDType(t) =>
                write!(f, "Typ danych {} nieobsługiwany (użyj modelu F16)", t),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GgufDType { F32, F16 }

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDType,
    pub offset: u64,
}
impl TensorInfo {
    pub fn n_elements(&self) -> u64 { self.shape.iter().product() }
}

pub struct LayerWeights {
    pub layer_idx: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub rms_att_weight: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
}

pub struct GgufLoader {
    file: File,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: u64,
}

impl GgufLoader {
    pub fn open(path: &str) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" { return Err(GgufError::InvalidMagic); }
        let version = read_u32(&mut file)?;
        if version < 2 || version > 3 { return Err(GgufError::UnsupportedVersion(version)); }
        let n_tensors = read_u64(&mut file)?;
        let n_kv = read_u64(&mut file)?;
        for _ in 0..n_kv { skip_kv(&mut file)?; }
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = read_str(&mut file)?;
            let n_dims = read_u32(&mut file)?;
            let shape: Vec<u64> = (0..n_dims).map(|_| read_u64(&mut file)).collect::<Result<_,_>>()?;
            let dtype_raw = read_u32(&mut file)?;
            let dtype = match dtype_raw {
                0 => GgufDType::F32,
                1 => GgufDType::F16,
                t => return Err(GgufError::UnsupportedDType(t)),
            };
            let offset = read_u64(&mut file)?;
            tensors.insert(name.clone(), TensorInfo { name, shape, dtype, offset });
        }
        let pos = file.seek(SeekFrom::Current(0))?;
        let data_offset = (pos + 31) & !31;
        Ok(GgufLoader { file, tensors, data_offset })
    }

    pub fn load_f32(&mut self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?.clone();
        self.file.seek(SeekFrom::Start(self.data_offset + info.offset))?;
        let n = info.n_elements() as usize;
        match info.dtype {
            GgufDType::F32 => {
                let mut buf = vec![0u8; n * 4];
                self.file.read_exact(&mut buf)?;
                Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
            }
            GgufDType::F16 => {
                let mut buf = vec![0u8; n * 2];
                self.file.read_exact(&mut buf)?;
                Ok(buf.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect())
            }
        }
    }

    pub fn load_layer(&mut self, l: usize) -> Result<LayerWeights, GgufError> {
        Ok(LayerWeights {
            layer_idx: l,
            w_q:            self.load_f32(&format!("blk.{}.attn_q.weight", l))?,
            w_k:            self.load_f32(&format!("blk.{}.attn_k.weight", l))?,
            w_v:            self.load_f32(&format!("blk.{}.attn_v.weight", l))?,
            w_o:            self.load_f32(&format!("blk.{}.attn_output.weight", l))?,
            w_gate:         self.load_f32(&format!("blk.{}.ffn_gate.weight", l))?,
            w_up:           self.load_f32(&format!("blk.{}.ffn_up.weight", l))?,
            w_down:         self.load_f32(&format!("blk.{}.ffn_down.weight", l))?,
            rms_att_weight: self.load_f32(&format!("blk.{}.attn_norm.weight", l))?,
            rms_ffn_weight: self.load_f32(&format!("blk.{}.ffn_norm.weight", l))?,
        })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

fn read_u32(f: &mut File) -> Result<u32, std::io::Error> {
    let mut b = [0u8;4]; f.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64(f: &mut File) -> Result<u64, std::io::Error> {
    let mut b = [0u8;8]; f.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn read_str(f: &mut File) -> Result<String, GgufError> {
    let len = read_u64(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidMagic)
}
fn skip_kv(f: &mut File) -> Result<(), GgufError> {
    let l = read_u64(f)? as i64; f.seek(SeekFrom::Current(l))?;
    let t = read_u32(f)?; skip_val(f, t)
}
fn skip_val(f: &mut File, t: u32) -> Result<(), GgufError> {
    match t {
        0|1|7 => { f.seek(SeekFrom::Current(1))?; }
        2|3   => { f.seek(SeekFrom::Current(2))?; }
        4|5|6 => { f.seek(SeekFrom::Current(4))?; }
        10|11|12 => { f.seek(SeekFrom::Current(8))?; }
        8 => { let l=read_u64(f)? as i64; f.seek(SeekFrom::Current(l))?; }
        9 => { let at=read_u32(f)?; let al=read_u64(f)?;
               for _ in 0..al { skip_val(f, at)?; } }
        _ => {}
    }
    Ok(())
}
fn f16_to_f32(b: u16) -> f32 {
    let s=((b as u32)>>15)<<31; let e=((b>>10)&0x1f) as u32; let m=(b&0x3ff) as u32;
    f32::from_bits(if e==0 { if m==0 {s} else {let z=m.leading_zeros()-22;
        s|((127-14-z)<<23)|((m<<(z+1))&0x7fffff)} }
    else if e==31 {s|0x7f800000|(m<<13)} else {s|((e+127-15)<<23)|(m<<13)})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_f16_to_f32() {
        assert!((f16_to_f32(0x3C00)-1.0f32).abs()<1e-6);
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert!((f16_to_f32(0xBC00)+1.0f32).abs()<1e-6);
        assert!((f16_to_f32(0x4000)-2.0f32).abs()<1e-6);
    }
}
