use tosa::builder::*;
use tosa::serialize::{deserialize, serialize};
use tosa::types::*;

// ---------------------------------------------------------------------------
// 1. Minimal graph: one region, one block, one operator (Op::Add)
// ---------------------------------------------------------------------------

#[test]
fn minimal_graph_one_add_operator() {
    let graph = TosaGraphBuilder::new(1, 0, 0)
        .add_region("main")
        .add_block("entry")
        .add_tensor(TosaTensor::new("a", vec![2, 3], DType::Fp32))
        .add_tensor(TosaTensor::new("b", vec![2, 3], DType::Fp32))
        .add_tensor(TosaTensor::new("c", vec![2, 3], DType::Fp32))
        .add_input("a")
        .add_input("b")
        .add_output("c")
        .add_operator(Op::Add, Attribute::Add(AddAttribute {}), &["a", "b"], &["c"])
        .finish_block()
        .finish_region()
        .build();

    assert_eq!(graph.regions.len(), 1);
    assert_eq!(graph.regions[0].blocks.len(), 1);

    let block = &graph.regions[0].blocks[0];
    assert_eq!(block.operators.len(), 1);
    assert_eq!(block.operators[0].op, Op::Add);
    assert!(matches!(block.operators[0].attribute, Attribute::Add(_)));
    assert_eq!(block.operators[0].inputs, vec!["a", "b"]);
    assert_eq!(block.operators[0].outputs, vec!["c"]);
}

// ---------------------------------------------------------------------------
// 2. Multiple regions
// ---------------------------------------------------------------------------

#[test]
fn graph_with_multiple_regions() {
    let graph = TosaGraphBuilder::new(1, 0, 0)
        .add_region("region_0")
        .add_block("block_0")
        .add_tensor(TosaTensor::new("x", vec![4], DType::Int32))
        .add_input("x")
        .add_output("x")
        .add_operator(Op::Identity, Attribute::Identity(IdentityAttribute {}), &["x"], &["x"])
        .finish_block()
        .finish_region()
        .add_region("region_1")
        .add_block("block_1")
        .add_tensor(TosaTensor::new("y", vec![4], DType::Int32))
        .add_input("y")
        .add_output("y")
        .add_operator(Op::Abs, Attribute::Abs(AbsAttribute {}), &["y"], &["y"])
        .finish_block()
        .finish_region()
        .build();

    assert_eq!(graph.regions.len(), 2);
    assert_eq!(graph.regions[0].name, "region_0");
    assert_eq!(graph.regions[1].name, "region_1");
    assert_eq!(graph.regions[0].blocks[0].operators[0].op, Op::Identity);
    assert_eq!(graph.regions[1].blocks[0].operators[0].op, Op::Abs);
}

// ---------------------------------------------------------------------------
// 3. Multiple blocks in one region
// ---------------------------------------------------------------------------

#[test]
fn multiple_blocks_in_one_region() {
    let graph = TosaGraphBuilder::new(1, 0, 0)
        .add_region("main")
        .add_block("block_a")
        .add_tensor(TosaTensor::new("t0", vec![1], DType::Fp32))
        .add_input("t0")
        .add_output("t0")
        .finish_block()
        .add_block("block_b")
        .add_tensor(TosaTensor::new("t1", vec![2], DType::Int32))
        .add_input("t1")
        .add_output("t1")
        .finish_block()
        .add_block("block_c")
        .add_tensor(TosaTensor::new("t2", vec![3], DType::Int8))
        .add_input("t2")
        .add_output("t2")
        .finish_block()
        .finish_region()
        .build();

    let region = &graph.regions[0];
    assert_eq!(region.blocks.len(), 3);
    assert_eq!(region.blocks[0].name, "block_a");
    assert_eq!(region.blocks[1].name, "block_b");
    assert_eq!(region.blocks[2].name, "block_c");
}

// ---------------------------------------------------------------------------
// 4. TosaTensor constructors: new, with_data, variable
// ---------------------------------------------------------------------------

#[test]
fn tensor_new() {
    let t = TosaTensor::new("input", vec![1, 3, 224, 224], DType::Fp32);
    assert_eq!(t.name, "input");
    assert_eq!(t.shape, vec![1, 3, 224, 224]);
    assert_eq!(t.dtype, DType::Fp32);
    assert!(t.data.is_none());
    assert!(!t.variable);
    assert!(!t.is_unranked);
    assert!(t.variable_name.is_none());
}

#[test]
fn tensor_with_data() {
    let data = vec![0u8; 32];
    let t = TosaTensor::with_data("weights", vec![4, 2], DType::Fp32, data.clone());
    assert_eq!(t.name, "weights");
    assert_eq!(t.shape, vec![4, 2]);
    assert_eq!(t.dtype, DType::Fp32);
    assert_eq!(t.data, Some(data));
    assert!(!t.variable);
}

#[test]
fn tensor_variable() {
    let t = TosaTensor::variable("var0", vec![10], DType::Int32, "my_variable");
    assert_eq!(t.name, "var0");
    assert_eq!(t.shape, vec![10]);
    assert_eq!(t.dtype, DType::Int32);
    assert!(t.variable);
    assert!(t.data.is_none());
    assert_eq!(t.variable_name, Some("my_variable".to_string()));
}

// ---------------------------------------------------------------------------
// 5. Adding tensors, inputs, outputs to a block
// ---------------------------------------------------------------------------

#[test]
fn block_tensors_inputs_outputs() {
    let graph = TosaGraphBuilder::new(1, 0, 0)
        .add_region("main")
        .add_block("entry")
        .add_tensor(TosaTensor::new("in0", vec![1, 8], DType::Fp32))
        .add_tensor(TosaTensor::new("in1", vec![1, 8], DType::Fp32))
        .add_tensor(TosaTensor::new("out0", vec![1, 8], DType::Fp32))
        .add_const_tensor("bias", &[8], DType::Fp32, vec![0u8; 32])
        .add_input("in0")
        .add_input("in1")
        .add_output("out0")
        .finish_block()
        .finish_region()
        .build();

    let block = &graph.regions[0].blocks[0];
    assert_eq!(block.tensors.len(), 4);
    assert_eq!(block.inputs, vec!["in0", "in1"]);
    assert_eq!(block.outputs, vec!["out0"]);

    // The const tensor should have data
    let bias = block.tensors.iter().find(|t| t.name == "bias").unwrap();
    assert!(bias.data.is_some());
    assert_eq!(bias.data.as_ref().unwrap().len(), 32);
}

// ---------------------------------------------------------------------------
// 6. Build -> serialize -> deserialize roundtrip
// ---------------------------------------------------------------------------

#[test]
fn serialize_deserialize_roundtrip() {
    let graph = TosaGraphBuilder::new(1, 2, 3)
        .draft(true)
        .add_region("main")
        .add_block("entry")
        .add_tensor(TosaTensor::new("x", vec![2, 4], DType::Fp32))
        .add_tensor(TosaTensor::new("y", vec![2, 4], DType::Fp32))
        .add_input("x")
        .add_output("y")
        .add_operator(
            Op::Sigmoid,
            Attribute::Sigmoid(SigmoidAttribute {}),
            &["x"],
            &["y"],
        )
        .finish_block()
        .finish_region()
        .build();

    let bytes = serialize(&graph);
    let roundtripped = deserialize(&bytes).expect("deserialize should succeed");

    assert_eq!(roundtripped.version.major, graph.version.major);
    assert_eq!(roundtripped.version.minor, graph.version.minor);
    assert_eq!(roundtripped.version.patch, graph.version.patch);
    assert_eq!(roundtripped.version.draft, graph.version.draft);

    assert_eq!(roundtripped.regions.len(), 1);
    let region = &roundtripped.regions[0];
    assert_eq!(region.name, "main");
    assert_eq!(region.blocks.len(), 1);

    let block = &region.blocks[0];
    assert_eq!(block.name, "entry");
    assert_eq!(block.inputs, vec!["x"]);
    assert_eq!(block.outputs, vec!["y"]);
    assert_eq!(block.tensors.len(), 2);
    assert_eq!(block.operators.len(), 1);
    assert_eq!(block.operators[0].op, Op::Sigmoid);
    assert_eq!(block.operators[0].inputs, vec!["x"]);
    assert_eq!(block.operators[0].outputs, vec!["y"]);
}

// ---------------------------------------------------------------------------
// 7. Realistic model: input -> Conv2d -> Add (bias) -> output
// ---------------------------------------------------------------------------

#[test]
fn realistic_conv2d_add_model() {
    let graph = TosaGraphBuilder::new(1, 0, 0)
        .add_region("main")
        .add_block("entry")
        // tensors
        .add_tensor(TosaTensor::new("input", vec![1, 3, 32, 32], DType::Fp32))
        .add_tensor(TosaTensor::new("conv_weight", vec![16, 3, 3, 3], DType::Fp32))
        .add_tensor(TosaTensor::new("conv_bias", vec![16], DType::Fp32))
        .add_tensor(TosaTensor::new("conv_out", vec![1, 16, 30, 30], DType::Fp32))
        .add_tensor(TosaTensor::new("bias", vec![1, 16, 30, 30], DType::Fp32))
        .add_tensor(TosaTensor::new("output", vec![1, 16, 30, 30], DType::Fp32))
        // block inputs/outputs
        .add_input("input")
        .add_output("output")
        // Conv2d operator
        .add_operator(
            Op::Conv2d,
            Attribute::Conv2d(Conv2dAttribute {
                pad: vec![0, 0, 0, 0],
                stride: vec![1, 1],
                dilation: vec![1, 1],
                local_bound: false,
                acc_type: DType::Fp32,
            }),
            &["input", "conv_weight", "conv_bias"],
            &["conv_out"],
        )
        // Add operator (bias add)
        .add_operator(
            Op::Add,
            Attribute::Add(AddAttribute {}),
            &["conv_out", "bias"],
            &["output"],
        )
        .finish_block()
        .finish_region()
        .build();

    assert_eq!(graph.regions.len(), 1);
    let block = &graph.regions[0].blocks[0];
    assert_eq!(block.tensors.len(), 6);
    assert_eq!(block.operators.len(), 2);
    assert_eq!(block.operators[0].op, Op::Conv2d);
    assert_eq!(block.operators[1].op, Op::Add);

    // Verify the Conv2d attribute fields
    if let Attribute::Conv2d(ref attr) = block.operators[0].attribute {
        assert_eq!(attr.pad, vec![0, 0, 0, 0]);
        assert_eq!(attr.stride, vec![1, 1]);
        assert_eq!(attr.dilation, vec![1, 1]);
        assert!(!attr.local_bound);
        assert_eq!(attr.acc_type, DType::Fp32);
    } else {
        panic!("Expected Conv2d attribute");
    }

    // Roundtrip the realistic model
    let bytes = serialize(&graph);
    let rt = deserialize(&bytes).expect("roundtrip failed");
    assert_eq!(rt.regions[0].blocks[0].operators.len(), 2);
    assert_eq!(rt.regions[0].blocks[0].operators[0].op, Op::Conv2d);
    assert_eq!(rt.regions[0].blocks[0].operators[1].op, Op::Add);
}

// ---------------------------------------------------------------------------
// 8. Version fields (major, minor, patch, draft)
// ---------------------------------------------------------------------------

#[test]
fn version_fields_default() {
    let graph = TosaGraphBuilder::new(0, 100, 1).build();
    assert_eq!(graph.version.major, 0);
    assert_eq!(graph.version.minor, 100);
    assert_eq!(graph.version.patch, 1);
    assert!(!graph.version.draft); // default is false
}

#[test]
fn version_draft_flag() {
    let graph = TosaGraphBuilder::new(2, 3, 4).draft(true).build();
    assert_eq!(graph.version.major, 2);
    assert_eq!(graph.version.minor, 3);
    assert_eq!(graph.version.patch, 4);
    assert!(graph.version.draft);
}

#[test]
fn version_roundtrips() {
    let graph = TosaGraphBuilder::new(5, 6, 7).draft(true).build();
    let bytes = serialize(&graph);
    let rt = deserialize(&bytes).expect("deserialize failed");
    assert_eq!(rt.version.major, 5);
    assert_eq!(rt.version.minor, 6);
    assert_eq!(rt.version.patch, 7);
    assert!(rt.version.draft);
}

// ---------------------------------------------------------------------------
// 9. Operator with all inputs/outputs connected by name
// ---------------------------------------------------------------------------

#[test]
fn operator_inputs_outputs_connected_by_name() {
    let graph = TosaGraphBuilder::new(1, 0, 0)
        .add_region("main")
        .add_block("entry")
        .add_tensor(TosaTensor::new("lhs", vec![4, 4], DType::Fp32))
        .add_tensor(TosaTensor::new("rhs", vec![4, 4], DType::Fp32))
        .add_tensor(TosaTensor::new("result", vec![4, 4], DType::Fp32))
        .add_input("lhs")
        .add_input("rhs")
        .add_output("result")
        .add_operator(
            Op::Sub,
            Attribute::Sub(SubAttribute {}),
            &["lhs", "rhs"],
            &["result"],
        )
        .finish_block()
        .finish_region()
        .build();

    let op = &graph.regions[0].blocks[0].operators[0];
    assert_eq!(op.op, Op::Sub);
    assert_eq!(op.inputs, vec!["lhs", "rhs"]);
    assert_eq!(op.outputs, vec!["result"]);

    // Verify tensor names match operator references
    let block = &graph.regions[0].blocks[0];
    let tensor_names: Vec<&str> = block.tensors.iter().map(|t| t.name.as_str()).collect();
    for input_name in &op.inputs {
        assert!(
            tensor_names.contains(&input_name.as_str()),
            "operator input '{}' not found in block tensors",
            input_name
        );
    }
    for output_name in &op.outputs {
        assert!(
            tensor_names.contains(&output_name.as_str()),
            "operator output '{}' not found in block tensors",
            output_name
        );
    }
}

// ---------------------------------------------------------------------------
// 10. Empty graph (no regions) roundtrips
// ---------------------------------------------------------------------------

#[test]
fn empty_graph_roundtrips() {
    let graph = TosaGraphBuilder::new(1, 0, 0).build();
    assert!(graph.regions.is_empty());

    let bytes = serialize(&graph);
    let rt = deserialize(&bytes).expect("deserialize empty graph failed");
    assert!(rt.regions.is_empty());
    assert_eq!(rt.version.major, 1);
    assert_eq!(rt.version.minor, 0);
    assert_eq!(rt.version.patch, 0);
}
