// src/compress/serial.rs
//
// Binary serialization for EML-TSLP grammars.
// Format: EMLT (EML TSLP)
//
// This module provides persistence for the extracted grammars,
// allowing them to be stored and transmitted in a compact form.

use crate::tslp::grammar::{TslpGrammar, TslpRhs, LeafKind, GrammarNodeId};
use std::collections::HashMap;
use std::io::{Read, Write, Result};

const MAGIC: &[u8; 4] = b"EMLT";
const VERSION: u32 = 1;

/// Serializes a TSLP grammar to a binary bitstream.
pub fn serialize_grammar<W: Write>(grammar: &TslpGrammar, writer: &mut W) -> Result<()> {
    // 1. Header
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    
    // 2. Metadata
    let n_productions = grammar.productions.len() as u32;
    writer.write_all(&n_productions.to_le_bytes())?;
    
    // Create a mapping from GrammarNodeId (pointer-based) to sequential u32 IDs
    let mut id_map: HashMap<GrammarNodeId, u32> = HashMap::new();
    for (i, (old_id, _)) in grammar.productions.iter().enumerate() {
        id_map.insert(*old_id, i as u32);
    }
    
    let start_idx = *id_map.get(&grammar.start).unwrap_or(&0);
    writer.write_all(&start_idx.to_le_bytes())?;
    
    // 3. Productions
    for (_, rhs) in &grammar.productions {
        match rhs {
            TslpRhs::Leaf(LeafKind::One) => {
                writer.write_all(&[0u8])?;
            }
            TslpRhs::Leaf(LeafKind::Var(name)) => {
                writer.write_all(&[1u8])?;
                let bytes = name.as_bytes();
                writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
                writer.write_all(bytes)?;
            }
            TslpRhs::Leaf(LeafKind::Const(v)) => {
                writer.write_all(&[2u8])?;
                writer.write_all(&v.to_le_bytes())?;
            }
            TslpRhs::Eml(l, r) => {
                writer.write_all(&[3u8])?;
                let l_idx = *id_map.get(l).expect("Left child ID not found in map");
                let r_idx = *id_map.get(r).expect("Right child ID not found in map");
                writer.write_all(&l_idx.to_le_bytes())?;
                writer.write_all(&r_idx.to_le_bytes())?;
            }
        }
    }
    
    Ok(())
}

/// Deserializes a TSLP grammar from a binary bitstream.
pub fn deserialize_grammar<R: Read>(reader: &mut R) -> Result<TslpGrammar> {
    // 1. Header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid magic"));
    }
    
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let _version = u32::from_le_bytes(version_bytes);
    
    // 2. Metadata
    let mut n_bytes = [0u8; 4];
    reader.read_exact(&mut n_bytes)?;
    let n_productions = u32::from_le_bytes(n_bytes);
    
    let mut start_bytes = [0u8; 4];
    reader.read_exact(&mut start_bytes)?;
    let start_idx = u32::from_le_bytes(start_bytes) as usize;
    
    // 3. Productions
    let mut productions = Vec::new();
    
    for i in 0..n_productions {
        let mut kind_byte = [0u8; 1];
        reader.read_exact(&mut kind_byte)?;
        
        let rhs = match kind_byte[0] {
            0 => TslpRhs::Leaf(LeafKind::One),
            1 => {
                let mut len_bytes = [0u8; 4];
                reader.read_exact(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                let mut name_bytes = vec![0u8; len];
                reader.read_exact(&mut name_bytes)?;
                let name = String::from_utf8(name_bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                TslpRhs::Leaf(LeafKind::Var(name))
            }
            2 => {
                let mut v_bytes = [0u8; 8];
                reader.read_exact(&mut v_bytes)?;
                let v = f64::from_le_bytes(v_bytes);
                TslpRhs::Leaf(LeafKind::Const(v))
            }
            3 => {
                let mut l_bytes = [0u8; 4];
                reader.read_exact(&mut l_bytes)?;
                let l = u32::from_le_bytes(l_bytes) as usize;
                
                let mut r_bytes = [0u8; 4];
                reader.read_exact(&mut r_bytes)?;
                let r = u32::from_le_bytes(r_bytes) as usize;
                
                TslpRhs::Eml(l, r)
            }
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid production type")),
        };
        
        productions.push((i as usize, rhs));
    }
    
    // Recompute depths (simple topological pass)
    let mut depths = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (id, rhs) in &productions {
            if depths.contains_key(id) { continue; }
            match rhs {
                TslpRhs::Leaf(_) => {
                    depths.insert(*id, 0);
                    changed = true;
                }
                TslpRhs::Eml(l, r) => {
                    if let (Some(&dl), Some(&dr)) = (depths.get(l), depths.get(r)) {
                        depths.insert(*id, 1 + dl.max(dr));
                        changed = true;
                    }
                }
            }
        }
    }
    
    Ok(TslpGrammar {
        productions,
        start: start_idx,
        depths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tslp::grammar::extract_grammar;
    use crate::ast::*;

    #[test]
    fn test_serialization_roundtrip() {
        let x = var("x");
        let tree = eml(ln_node(x.clone()), exp_node(konst(3.14)));
        let grammar = extract_grammar(&tree);
        
        let mut buffer = Vec::new();
        serialize_grammar(&grammar, &mut buffer).unwrap();
        
        let mut reader = &buffer[..];
        let restored = deserialize_grammar(&mut reader).unwrap();
        
        assert_eq!(grammar.productions.len(), restored.productions.len());
        assert_eq!(grammar.max_depth(), restored.max_depth());
    }
}
