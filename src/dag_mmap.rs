use std::collections::HashMap;
use std::fs::OpenOptions;
use memmap2::MmapMut;
use crate::ast::EmlNode;
use std::sync::Arc;

/// 24-byte node stored on disk via mmap.
/// Compact and aligned for efficient access.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CompactNode {
    pub tag: u8,          // 0=One, 1=Var, 2=Const, 3=Eml
    pub _pad: [u8; 3],
    pub left: u32,        // Eml: left child index; Var: var_id; Const: const_id
    pub right: u32,       // Eml: right child index; else: 0
    pub hash: u64,        // structural hash for global deduplication
    pub ref_count: u32,   // global sharing counter
}

/// Out-of-core DAG engine.
/// - Nodes: Stored on ZFS SSD via mmap (O(1) access).
/// - Hash Index: Stored in RAM (O(1) lookup).
pub struct MmapDag {
    _file: std::fs::File,
    mmap: MmapMut,
    capacity: u64,
    
    // --- RAM State ---
    pub node_count: u32,
    hash_index: HashMap<u64, u32>,   // hash -> node_id
    var_names: Vec<String>,           // var_id -> name
    var_index: HashMap<String, u32>,  // name -> var_id
    const_values: Vec<f64>,           // const_id -> value
    const_index: HashMap<u64, u32>,   // bits -> const_id
}

impl MmapDag {
    /// Creates or opens a MmapDag.
    /// Initial capacity is specified in number of nodes.
    pub fn create(path: &str, initial_capacity: u64) -> std::io::Result<Self> {
        let file_size = 8 + (initial_capacity * 24); // 8B magic/header + nodes
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        
        file.set_len(file_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(MmapDag {
            _file: file,
            mmap,
            capacity: initial_capacity,
            node_count: 0,
            hash_index: HashMap::with_capacity(initial_capacity as usize), 
            var_names: Vec::new(),
            var_index: HashMap::new(),
            const_values: Vec::new(),
            const_index: HashMap::new(),
        })
    }

    /// Gets a node from the mmap by ID.
    #[inline(always)]
    pub fn get_node(&self, id: u32) -> &CompactNode {
        let offset = 8 + (id as usize * 24);
        unsafe {
            let ptr = self.mmap.as_ptr().add(offset) as *const CompactNode;
            &*ptr
        }
    }

    /// Mutably gets a node from the mmap (internal use for ref_count).
    #[inline(always)]
    fn get_node_mut(&mut self, id: u32) -> &mut CompactNode {
        let offset = 8 + (id as usize * 24);
        unsafe {
            let ptr = self.mmap.as_mut_ptr().add(offset) as *mut CompactNode;
            &mut *ptr
        }
    }

    /// Pushes a new node to the storage.
    fn push_node(&mut self, node: CompactNode) -> u32 {
        if self.node_count as u64 >= self.capacity {
            panic!("MmapDag capacity exceeded ({})", self.capacity);
        }
        let id = self.node_count;
        let offset = 8 + (id as usize * 24);
        
        unsafe {
            let ptr = self.mmap.as_mut_ptr().add(offset) as *mut CompactNode;
            std::ptr::write(ptr, node);
        }
        
        self.node_count += 1;
        id
    }

    pub fn intern_var(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.var_index.get(name) {
            return id;
        }
        let id = self.var_names.len() as u32;
        self.var_names.push(name.to_string());
        self.var_index.insert(name.to_string(), id);
        id
    }

    pub fn intern_const(&mut self, v: f64) -> u32 {
        let bits = v.to_bits();
        if let Some(&id) = self.const_index.get(&bits) {
            return id;
        }
        let id = self.const_values.len() as u32;
        self.const_values.push(v);
        self.const_index.insert(bits, id);
        id
    }

    /// Adds a node to the DAG or returns existing ID.
    pub fn add_eml_node(&mut self, left: u32, right: u32, hash: u64) -> u32 {
        if let Some(&id) = self.hash_index.get(&hash) {
            self.get_node_mut(id).ref_count += 1;
            return id;
        }
        
        let node = CompactNode {
            tag: 3,
            _pad: [0; 3],
            left,
            right,
            hash,
            ref_count: 1,
        };
        
        let id = self.push_node(node);
        self.hash_index.insert(hash, id);
        id
    }

    /// Special cases for leaf node creation in the DAG
    pub fn add_leaf_node(&mut self, tag: u8, val_id: u32, hash: u64) -> u32 {
        if let Some(&id) = self.hash_index.get(&hash) {
            self.get_node_mut(id).ref_count += 1;
            return id;
        }
        
        let node = CompactNode {
            tag,
            _pad: [0; 3],
            left: val_id,
            right: 0,
            hash,
            ref_count: 1,
        };
        
        let id = self.push_node(node);
        self.hash_index.insert(hash, id);
        id
    }

    pub fn unique_node_count(&self) -> u32 {
        self.node_count
    }

    pub fn sharing_savings(&self) -> u64 {
        let mut savings = 0u64;
        for id in 0..self.node_count {
            let node = self.get_node(id);
            savings += (node.ref_count.saturating_sub(1)) as u64;
        }
        savings
    }
}

// --- Hashing Helpers (matching the logic in full_layer_unified) ---

pub fn hash_one() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    "one".hash(&mut h);
    h.finish()
}

pub fn hash_var(var_id: u32) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    "var".hash(&mut h);
    var_id.hash(&mut h);
    h.finish()
}

pub fn hash_const(bits: u64) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    "const".hash(&mut h);
    bits.hash(&mut h);
    h.finish()
}

pub fn hash_eml(left: u32, right: u32) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    left.hash(&mut h);
    right.hash(&mut h);
    h.finish()
}

/// Adapter to add an existing Arc<EmlNode> tree to the MmapDag.
/// Uses a local cache per-call to handle sharing within the tree efficiently.
pub fn add_tree_to_mmap_dag(
    dag: &mut MmapDag,
    node: &Arc<EmlNode>,
    local_cache: &mut HashMap<usize, u32>,
) -> u32 {
    let ptr = Arc::as_ptr(node) as usize;
    if let Some(&id) = local_cache.get(&ptr) {
        return id;
    }

    let id = match node.as_ref() {
        EmlNode::One => {
            let hash = hash_one();
            dag.add_leaf_node(0, 0, hash)
        }
        EmlNode::Var(name) => {
            let var_id = dag.intern_var(name);
            let hash = hash_var(var_id);
            dag.add_leaf_node(1, var_id, hash)
        }
        EmlNode::Const(v) => {
            let const_id = dag.intern_const(*v);
            let hash = hash_const(v.to_bits());
            dag.add_leaf_node(2, const_id, hash)
        }
        EmlNode::Eml(l, r) => {
            let l_id = add_tree_to_mmap_dag(dag, l, local_cache);
            let r_id = add_tree_to_mmap_dag(dag, r, local_cache);
            let hash = hash_eml(l_id, r_id);
            dag.add_eml_node(l_id, r_id, hash)
        }
    };

    local_cache.insert(ptr, id);
    id
}
