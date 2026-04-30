import gguf
import numpy as np
import json
import os

# Konwerter zsynchronizowany z architekturą projektu (nn_layer.rs i asis.rs)

class EMLGraph:
    def __init__(self):
        self.nodes = []
        self.cache = {}

    def add_node(self, op, *args):
        # Strukturę opieramy na wektorach [op, arg1, arg2, ...] dla łatwego ładowania w Rust
        key = (op, args)
        if key in self.cache: return self.cache[key]
        idx = len(self.nodes)
        self.nodes.append([op] + list(args))
        self.cache[key] = idx
        return idx

    def one(self): return self.add_node("One")
    def konst(self, val): return self.add_node("Const", float(val))
    def var(self, name): return self.add_node("Var", name)
    def eml(self, l, r): return self.add_node("Eml", l, r)
    
    # Podstawowe tożsamości z PAPER.md i ast.rs
    def exp_node(self, x): return self.eml(x, self.one())
    
    def ln_node(self, x):
        # ln(x) = eml(1, eml(eml(1, x), 1))
        r1 = self.eml(self.one(), x)
        r2 = self.eml(r1, self.one())
        return self.eml(self.one(), r2)
    
    def mul_cf(self, x, w):
        # Optymalizacja 5-węzłowa z nn_layer.rs:31
        # w*x = eml(eml(ln(ln(x)), konst(1/w)), one())
        if abs(w) < 1e-15: return self.konst(0.0)
        ln_ln_x = self.ln_node(self.ln_node(x))
        return self.eml(self.eml(ln_ln_x, self.konst(1.0 / w)), self.one())

    def sub_eml(self, x, y):
        # x - y = eml(ln(x), exp(y))
        return self.eml(self.ln_node(x), self.exp_node(y))

    def add_eml(self, x, y):
        # x + y = x - (0 - y)
        zero = self.konst(0.0)
        neg_y = self.sub_eml(zero, y)
        return self.sub_eml(x, neg_y)

    def dot_product(self, inputs, weights):
        # Balanced Tree Reduction z nn_layer.rs:48
        terms = []
        for x, w in zip(inputs, weights):
            terms.append(self.mul_cf(x, w))
            
        while len(terms) > 1:
            next_t = []
            for i in range(0, len(terms), 2):
                if i+1 < len(terms):
                    next_t.append(self.add_eml(terms[i], terms[i+1]))
                else:
                    next_t.append(terms[i])
            terms = next_t
        return terms[0]

def export_full_model(gguf_path, out_json, h_limit=16):
    print(f"Reading {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)
    graph = EMLGraph()
    
    # Przykład: Warstwa Attention Projection (Q)
    l = 0
    wq_tensor = next(t for t in reader.tensors if t.name == f"blk.{l}.attn_q.weight")
    # TinyLlama ma ujemne wagi, co przetestuje stabilność ln(0)
    wq = wq_tensor.data[:h_limit, :h_limit]
    
    # Wejścia > 1.0 (wymagane dla mul_cf z nn_layer.rs)
    inputs = [graph.add_eml(graph.var(f"x.{i}"), graph.konst(2.0)) for i in range(h_limit)]
    
    outputs = []
    for i in range(h_limit):
        outputs.append(graph.dot_product(inputs, wq[i]))
        
    model_data = {
        "metadata": {"layers": 1, "hidden": h_limit},
        "graph": {"nodes": graph.nodes, "outputs": outputs}
    }
    
    with open(out_json, "w") as f:
        json.dump(model_data, f)
    print(f"Exported {len(graph.nodes)} nodes to {out_json}")

if __name__ == "__main__":
    export_full_model("models/tinyllama-f16.gguf", "research/model_full.oxieml.json")
