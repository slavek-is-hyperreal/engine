use eml_trs::cost_model::CostModel;

fn main() {
    println!("TinyLlama Layer Reduction: {:.4}%", CostModel::tinyllama_layer_reduction());
}
