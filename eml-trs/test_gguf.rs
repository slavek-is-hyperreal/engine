use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let mut file = File::open("models/tinyllama-f16.gguf").unwrap();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).unwrap();
    println!("Magic: {:?}", std::str::from_utf8(&magic));
    let mut b = [0u8; 4]; file.read_exact(&mut b).unwrap();
    let version = u32::from_le_bytes(b);
    println!("Version: {}", version);
    
    let mut b8 = [0u8; 8]; file.read_exact(&mut b8).unwrap();
    let n_tensors = u64::from_le_bytes(b8);
    println!("Tensors: {}", n_tensors);
    
    file.read_exact(&mut b8).unwrap();
    let n_kv = u64::from_le_bytes(b8);
    println!("KV: {}", n_kv);
}
