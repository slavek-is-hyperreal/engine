// src/compress/json_loader.rs
use crate::ast::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum JsonEmlNode {
    #[serde(rename = "one")]
    One,
    #[serde(rename = "var")]
    Var { name: String },
    #[serde(rename = "konst")]
    Konst { value: f64 },
    #[serde(rename = "eml")]
    Eml { l: usize, r: usize },
}

#[derive(Deserialize, Serialize)]
pub struct JsonEmlGraph {
    pub nodes: Vec<JsonEmlNode>,
    pub outputs: HashMap<String, usize>,
}

pub fn load_eml_json(json_str: &str) -> Vec<Arc<EmlNode>> {
    let graph: JsonEmlGraph = serde_json::from_str(json_str).expect("Failed to parse EML JSON");
    let mut node_cache: HashMap<usize, Arc<EmlNode>> = HashMap::new();
    
    for (id, json_node) in graph.nodes.iter().enumerate() {
        let node = match json_node {
            JsonEmlNode::One => one(),
            JsonEmlNode::Var { name } => var(name),
            JsonEmlNode::Konst { value } => konst(*value),
            JsonEmlNode::Eml { l, r } => {
                let left = node_cache.get(l).expect("Left child not found").clone();
                let right = node_cache.get(r).expect("Right child not found").clone();
                eml(left, right)
            }
        };
        node_cache.insert(id, node);
    }
    
    graph.outputs.values()
        .map(|&id| node_cache.get(&id).expect("Output node not found").clone())
        .collect()
}
