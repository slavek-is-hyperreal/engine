use eml_trs::ast::*;
use eml_trs::dag_mmap::*;
use std::sync::Arc;
use std::collections::HashMap;

fn main() {
    println!("=== Testing MmapDag V3 ===");
    let path = "/vectorlegis_ssd_pool/eml_working/test_dag.bin";
    let mut dag = MmapDag::create(path, 1000).expect("Failed to create test dag");

    // Test 1: Basic deduplication
    println!("Test 1: Basic deduplication...");
    let x = var("x");
    let mut local1 = HashMap::new();
    let id1 = add_tree_to_mmap_dag(&mut dag, &x, &mut local1);
    
    let mut local2 = HashMap::new();
    let id2 = add_tree_to_mmap_dag(&mut dag, &x, &mut local2);
    
    assert_eq!(id1, id2, "x should have same ID across different local caches");
    assert_eq!(dag.unique_node_count(), 1, "Should have exactly 1 node");
    println!("  [OK]");

    // Test 2: Complex tree deduplication
    println!("Test 2: Complex tree deduplication...");
    let y = var("y");
    let tree1 = eml(x.clone(), y.clone());
    let tree2 = eml(x.clone(), y.clone());
    
    let mut local3 = HashMap::new();
    add_tree_to_mmap_dag(&mut dag, &tree1, &mut local3);
    
    let mut local4 = HashMap::new();
    add_tree_to_mmap_dag(&mut dag, &tree2, &mut local4);
    
    // Unique nodes: x, y, eml(x,y) = 3
    assert_eq!(dag.unique_node_count(), 3);
    println!("  [OK]");

    // Test 3: x_vars cross-row sharing (the "Claude Test")
    println!("Test 3: x_vars cross-row sharing...");
    let x_vars: Vec<Arc<EmlNode>> = (0..10).map(|i| var(&format!("x{}", i))).collect();
    
    // Row 1
    {
        let mut local = HashMap::new();
        for x_i in &x_vars {
            add_tree_to_mmap_dag(&mut dag, x_i, &mut local);
        }
    }
    let count_after_row1 = dag.unique_node_count();
    
    // Row 2 (identical x_vars)
    {
        let mut local = HashMap::new();
        for x_i in &x_vars {
            add_tree_to_mmap_dag(&mut dag, x_i, &mut local);
        }
    }
    assert_eq!(dag.unique_node_count(), count_after_row1, "x_vars should NOT increase node count in Row 2");
    println!("  [OK]");

    println!("\nALL TESTS PASSED! MmapDag is scientifically sound.");
}
