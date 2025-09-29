use prost::Message;
use coeus_tensor::Tensor;
use onnx_proto::{model_proto::ModelProto, graph_proto::GraphProto, node_proto::NodeProto};

pub trait SerializeOps<D: Dtype> {
    fn export_onnx(&self, tensors: &[Tensor<D, Self>]) -> Vec<u8>;
}

impl<D: Dtype + Float, B: Backend<D = D>> SerializeOps<D> for B {
    fn export_onnx(&self, tensors: &[Tensor<D, Self>]) -> Vec<u8> {
        let mut graph = GraphProto::default();
        for t in tensors {
            let node = NodeProto {
                op_type: "Const".to_string(),
                input: vec![],
                output: vec![t.name().unwrap_or_default()],
                attribute: vec![/* tensor attrs */],
                ..Default::default()
            };
            graph.node.push(node);
        }
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        model.encode_to_vec()
    }
}

// ...async stream export...
