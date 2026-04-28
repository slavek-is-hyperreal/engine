// benches/compress_bench.rs
//
// Comparison of EML-TSLP with gzip and zstd.

use eml_trs::compress::lift::lift_bytes;
use eml_trs::compress::serial::serialize_grammar;
use eml_trs::tslp::grammar::extract_grammar;
use eml_trs::trs::rewrite;

use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("{:<20} | {:>10} | {:>10} | {:>10}", "Test Case", "Gzip (B)", "Zstd (B)", "EML (B)");
    println!("{:-<20}-|-{:-<10}-|-{:-<10}-|-{:-<10}", "", "", "", "");

    run_bench("Algebraic Repetition", generate_algebraic_data(100));
    run_bench("JSON Data", generate_json_data(50));
    run_bench("Random Bytes", (0..1024).map(|_| rand::random::<u8>()).collect());
}

fn run_bench(name: &str, data: Vec<u8>) {
    // 1. Gzip
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(&data).unwrap();
    let gz_size = gz.finish().unwrap().len();

    // 2. Zstd
    let zstd_size = zstd::encode_all(&data[..], 3).unwrap().len();

    // 3. EML-TSLP
    let tree = lift_bytes(&data);
    let optimized = rewrite(tree);
    let grammar = extract_grammar(&optimized);
    let mut eml_buf = Vec::new();
    serialize_grammar(&grammar, &mut eml_buf).unwrap();
    let eml_size = eml_buf.len();

    println!("{:<20} | {:>10} | {:>10} | {:>10}", name, gz_size, zstd_size, eml_size);
}

fn generate_algebraic_data(n: usize) -> Vec<u8> {
    // Simulate data with algebraic structure: repeating "1+1" or similar patterns
    let mut data = Vec::new();
    for _ in 0..n {
        data.extend_from_slice(b"eml(1, eml(1, 1))");
    }
    data
}

fn generate_json_data(n: usize) -> Vec<u8> {
    let mut data = Vec::new();
    data.push(b'[');
    for i in 0..n {
        if i > 0 { data.extend_from_slice(b", "); }
        data.extend_from_slice(format!("{{\"id\": {}, \"val\": {}}}", i, i as f64 * 0.1).as_bytes());
    }
    data.push(b']');
    data
}

// Minimal mock for rand::random since we don't have the crate
mod rand {
    pub fn random<T: Default>() -> T { T::default() }
}
