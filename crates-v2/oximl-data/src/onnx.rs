use oximl_core::TensorResult;
use oximl_autodiff::{Variable, graph::Op};
use std::path::Path;
use std::fs::File;
use std::io::Write;

/// High-performance ONNX serialization stubs.
pub struct OnnxExporter;

impl OnnxExporter {
    /// Serialize an OxidizeML v2 Computation Graph into a JSON-based ONNX structural stub.
    /// A true ONNX exporter requires massive Protobuf definition files. We mimic the logic by writing a JSON graph.
    pub fn export_model<P: AsRef<Path>>(root: &Variable, path: P) -> TensorResult<()> {
        let mut file = File::create(path).map_err(|e| oximl_core::TensorError::InvalidOperation(format!("File create error: {}", e)))?;
        
        writeln!(file, "{{\n  \"model\": \"OxidizeML_v2_Graph\",\n  \"nodes\": [").map_err(|e| oximl_core::TensorError::InvalidOperation(e.to_string()))?;
        
        let graph = &root.graph;
        let num_nodes = graph.len();
        
        for i in 0..num_nodes {
            let node = graph.get_node(i).unwrap();
            
            let op_name = match node.op {
                Op::Leaf => "Parameter/Input",
                Op::Add(_, _) => "Add",
                Op::Mul(_, _) => "Mul",
                Op::MatMul(_, _) => "MatMul",
                Op::ScalarMul(_, _) => "ConstantOfShape", 
                Op::Transpose(_) => "Transpose",
                Op::Reshape(_) => "Reshape",
                Op::Softmax(_) => "Softmax",
                Op::Relu(_) => "Relu",
                Op::Exp(_) => "Exp",
                Op::Ln(_) => "Log",
                Op::Div(_, _) => "Div",
            };
            
            let inputs = match node.op {
                Op::Leaf => vec![],
                Op::Add(a, b) | Op::Mul(a, b) | Op::MatMul(a, b) | Op::Div(a, b) => vec![a, b],
                Op::ScalarMul(a, _) | Op::Transpose(a) | Op::Reshape(a) | Op::Softmax(a) | Op::Relu(a) | Op::Exp(a) | Op::Ln(a) => vec![a],
            };
            
            let is_last = if i == num_nodes - 1 { "" } else { "," };
            writeln!(file, "    {{\"id\": {}, \"op\": \"{}\", \"inputs\": {:?}}}{}", node.id, op_name, inputs, is_last)
                .map_err(|e| oximl_core::TensorError::InvalidOperation(e.to_string()))?;
        }
        
        writeln!(file, "  ]\n}}").map_err(|e| oximl_core::TensorError::InvalidOperation(e.to_string()))?;
        
        Ok(())
    }
}
