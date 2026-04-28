// src/compress/lift.rs
//
// Symbol lifting: mapping raw data into the EML algebraic domain.
//
// Theorem C3 (PAPER.md) requires data to be represented as an EML tree
// before TRS normalization and grammar extraction can occur.
//
// Strategy:
// 1. Naive lifting: right-spine tree where each leaf is a byte/char value.
// 2. Structural lifting: higher-level lifting that preserves known 
//    algebraic patterns (JSON, CSV, Protobuf).

use crate::ast::*;
use std::sync::Arc;

/// Maps a byte sequence to a right-spine EML tree.
/// [b0, b1, b2] -> eml(b0, eml(b1, b2))
///
/// Time: O(N), Space: O(N)
pub fn lift_bytes(data: &[u8]) -> Arc<EmlNode> {
    if data.is_empty() {
        return one();
    }
    
    // Start from the last byte
    let mut res = konst(data[data.len() - 1] as f64);
    
    // Build the spine upwards
    for i in (0..data.len() - 1).rev() {
        res = eml(konst(data[i] as f64), res);
    }
    
    res
}

/// Reconstructs a byte sequence from an EML tree.
/// Traverses the right-spine and extracts constant values.
pub fn unlift_bytes(root: &Arc<EmlNode>) -> Vec<u8> {
    let mut res = Vec::new();
    let mut current = root;
    
    loop {
        match current.as_ref() {
            EmlNode::Const(v) => {
                res.push(*v as u8);
                break;
            }
            EmlNode::Eml(l, r) => {
                // Left child is the byte value
                if let EmlNode::Const(v) = l.as_ref() {
                    res.push(*v as u8);
                } else if let EmlNode::One = l.as_ref() {
                    res.push(1);
                }
                // Continue to the right
                current = r;
            }
            EmlNode::One => {
                res.push(1);
                break;
            }
            EmlNode::Var(_) => {
                // Variables cannot be unlifted to bytes easily
                break;
            }
        }
    }
    
    res
}

/// Maps UTF-8 text to an EML tree using Unicode code points.
pub fn lift_text(text: &str) -> Arc<EmlNode> {
    let chars: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    if chars.is_empty() {
        return one();
    }
    
    let mut res = konst(chars[chars.len() - 1]);
    for i in (0..chars.len() - 1).rev() {
        res = eml(konst(chars[i]), res);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift_unlift_roundtrip() {
        let data = b"Hello EML!";
        let tree = lift_bytes(data);
        let restored = unlift_bytes(&tree);
        assert_eq!(data.to_vec(), restored);
    }

    #[test]
    fn test_lift_text() {
        let text = "Algebraic Compression";
        let tree = lift_text(text);
        let restored = unlift_bytes(&tree); // reuse unlift_bytes for code points
        let restored_text: String = restored.iter()
            .map(|&b| b as char)
            .collect();
        assert_eq!(text, restored_text);
    }

    #[test]
    fn test_empty_lifting() {
        let tree = lift_bytes(&[]);
        assert!(matches!(tree.as_ref(), EmlNode::One));
    }
}
