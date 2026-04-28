// src/bin/eml_compress.rs
//
// CLI tool for EML-TSLP compression and decompression.

use eml_trs::compress::lift::{lift_bytes, unlift_bytes};
use eml_trs::compress::serial::{serialize_grammar, deserialize_grammar};
use eml_trs::compress::decompress::rebuild_tree;
use eml_trs::tslp::grammar::extract_grammar;
use eml_trs::trs::rewrite;
use eml_trs::ast::EmlNode;
use std::sync::Arc;
use std::env;
use std::fs::{File, read};
use std::io::{BufWriter, BufReader, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        print_usage();
        return;
    }

    let command = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    match command.as_str() {
        "compress" => compress(input_path, output_path),
        "compress-json" => compress_json(input_path, output_path),
        "decompress" => decompress(input_path, output_path),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("EML-TSLP Compressor");
    println!("Usage:");
    println!("  eml_compress compress <input> <output.eml>");
    println!("  eml_compress compress-json <input.json> <output.eml>");
    println!("  eml_compress decompress <input.eml> <output>");
}

fn compress_json(input_path: &str, output_path: &str) {
    println!("Compressing JSON Graph: {} -> {}", input_path, output_path);
    let start = Instant::now();

    // 1. Read input JSON
    let json_str = std::fs::read_to_string(input_path).expect("Failed to read JSON file");
    
    // 2. Load to EML
    let roots = eml_trs::compress::json_loader::load_eml_json(&json_str);
    
    // Count unique nodes across all roots
    let input_node_count = {
        let mut visited = std::collections::HashSet::new();
        fn count_all(node: &Arc<EmlNode>, visited: &mut std::collections::HashSet<usize>) -> usize {
            let ptr = Arc::as_ptr(node) as usize;
            if visited.contains(&ptr) { return 0; }
            visited.insert(ptr);
            match node.as_ref() {
                EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
                EmlNode::Eml(l, r) => 1 + count_all(l, visited) + count_all(r, visited),
            }
        }
        roots.iter().map(|r| count_all(r, &mut visited)).sum::<usize>()
    };

    // 3. TRS Normalize all roots
    let optimized_roots: Vec<Arc<EmlNode>> = roots.into_iter().map(|r| rewrite(r)).collect();
    
    let optimized_node_count = {
        let mut visited = std::collections::HashSet::new();
        fn count_all(node: &Arc<EmlNode>, visited: &mut std::collections::HashSet<usize>) -> usize {
            let ptr = Arc::as_ptr(node) as usize;
            if visited.contains(&ptr) { return 0; }
            visited.insert(ptr);
            match node.as_ref() {
                EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
                EmlNode::Eml(l, r) => 1 + count_all(l, visited) + count_all(r, visited),
            }
        }
        optimized_roots.iter().map(|r| count_all(r, &mut visited)).sum::<usize>()
    };
    
    // 4. Extract Grammar for the first root (for now)
    let grammar = extract_grammar(&optimized_roots[0]);
    
    // 5. Serialize
    {
        let file = File::create(output_path).expect("Failed to create output file");
        let mut writer = BufWriter::new(file);
        eml_trs::compress::serial::serialize_grammar(&grammar, &mut writer).expect("Failed to serialize grammar");
        writer.flush().expect("Failed to flush writer");
    }
    
    let duration = start.elapsed();
    let output_size = std::fs::metadata(output_path).unwrap().len();
    
    println!("Done in {:?}", duration);
    println!("Initial nodes:   {}", input_node_count);
    println!("Optimized nodes: {}", optimized_node_count);
    println!("Output size:     {} bytes", output_size);
    println!("Reduction:       {:.2}x nodes", input_node_count as f64 / optimized_node_count as f64);
}

fn compress(input_path: &str, output_path: &str) {
    println!("Compressing: {} -> {}", input_path, output_path);
    let start = Instant::now();

    // 1. Read input
    let data = read(input_path).expect("Failed to read input file");
    let input_size = data.len();

    // 2. Lift to EML
    let tree = lift_bytes(&data);
    
    // 3. TRS Normalize
    let optimized = rewrite(tree);
    
    // 4. Extract Grammar
    let grammar = extract_grammar(&optimized);
    
    // 5. Serialize
    {
        let file = File::create(output_path).expect("Failed to create output file");
        let mut writer = BufWriter::new(file);
        serialize_grammar(&grammar, &mut writer).expect("Failed to serialize grammar");
        writer.flush().expect("Failed to flush writer");
    }
    
    let duration = start.elapsed();
    let output_size = std::fs::metadata(output_path).unwrap().len();
    
    println!("Done in {:?}", duration);
    println!("Input size:  {} bytes", input_size);
    println!("Output size: {} bytes", output_size);
    println!("Ratio:       {:.2}x", input_size as f64 / output_size as f64);
}

fn decompress(input_path: &str, output_path: &str) {
    println!("Decompressing: {} -> {}", input_path, output_path);
    let start = Instant::now();

    // 1. Open input
    let file = File::open(input_path).expect("Failed to open input file");
    let mut reader = BufReader::new(file);
    
    // 2. Deserialize Grammar
    let grammar = deserialize_grammar(&mut reader).expect("Failed to deserialize grammar");
    
    // 3. Rebuild Tree
    let tree = rebuild_tree(&grammar);
    
    // 4. Unlift to bytes
    let data = unlift_bytes(&tree);
    
    // 5. Write output
    std::fs::write(output_path, data).expect("Failed to write output file");
    
    let duration = start.elapsed();
    println!("Done in {:?}", duration);
}
