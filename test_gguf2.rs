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

fn read_u32(f: &mut File) -> Result<u32, std::io::Error> {
    let mut b = [0u8;4]; f.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64(f: &mut File) -> Result<u64, std::io::Error> {
    let mut b = [0u8;8]; f.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn read_str(f: &mut File) -> Result<String, GgufError> {
    let len = read_u64(f).unwrap() as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).unwrap();
    String::from_utf8(buf).map_err(|_| GgufError::InvalidMagic)
}
fn skip_kv(f: &mut File) -> Result<(), GgufError> {
    let name = read_str(f)?;
    let t = read_u32(f).unwrap(); 
    println!("KV {} typ {}", name, t);
    skip_val(f, t)
}
fn skip_val(f: &mut File, t: u32) -> Result<(), GgufError> {
    match t {
        0|1|7 => { f.seek(SeekFrom::Current(1)).unwrap(); }
        2|12  => { f.seek(SeekFrom::Current(2)).unwrap(); }
        3|4|6 => { f.seek(SeekFrom::Current(4)).unwrap(); }
        5|10|11=>{ f.seek(SeekFrom::Current(8)).unwrap(); }
        8 => { let l=read_u64(f).unwrap() as i64; f.seek(SeekFrom::Current(l)).unwrap(); }
        9 => { let at=read_u32(f).unwrap(); let al=read_u64(f).unwrap();
               for _ in 0..al { skip_val(f, at)?; } }
        _ => { println!("UNKNOWN TYPE: {}", t); }
    }
    Ok(())
}

fn main() {
    let mut file = File::open("models/tinyllama-f16.gguf").unwrap();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).unwrap();
    let version = read_u32(&mut file).unwrap();
    let n_tensors = read_u64(&mut file).unwrap();
    let n_kv = read_u64(&mut file).unwrap();
    for i in 0..n_kv { 
        println!("Skipping KV {}", i);
        skip_kv(&mut file).unwrap(); 
    }
    println!("OK");
}
