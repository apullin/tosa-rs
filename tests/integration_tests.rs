use tosa_rs::types::*;
use tosa_rs::serialize::deserialize;

#[test]
fn test_load_minimal_add_fixture() {
    let bytes = include_bytes!("fixtures/minimal_add.tosa");
    let graph = deserialize(bytes).expect("failed to parse minimal_add fixture");

    // Version
    assert_eq!(graph.version.major, 1);
    assert_eq!(graph.version.minor, 0);
    assert_eq!(graph.version.patch, 0);
    assert_eq!(graph.version.draft, false);

    // Regions
    assert_eq!(graph.regions.len(), 1);
    let region = &graph.regions[0];
    assert_eq!(region.name, "main");

    // Blocks
    assert_eq!(region.blocks.len(), 1);
    let block = &region.blocks[0];
    assert_eq!(block.name, "block0");

    // Block inputs/outputs
    assert_eq!(block.inputs, vec!["input0", "input1"]);
    assert_eq!(block.outputs, vec!["output0"]);

    // Operators
    assert_eq!(block.operators.len(), 1);
    let op = &block.operators[0];
    assert_eq!(op.op, Op::Add);
    assert_eq!(op.inputs, vec!["input0", "input1"]);
    assert_eq!(op.outputs, vec!["output0"]);
    assert!(matches!(op.attribute, Attribute::Add(_)));

    // Tensors
    assert_eq!(block.tensors.len(), 3);

    let t0 = &block.tensors[0];
    assert_eq!(t0.name, "input0");
    assert_eq!(t0.shape, vec![1, 8]);
    assert_eq!(t0.dtype, DType::Fp32);

    let t1 = &block.tensors[1];
    assert_eq!(t1.name, "input1");
    assert_eq!(t1.shape, vec![1, 8]);
    assert_eq!(t1.dtype, DType::Fp32);

    let t2 = &block.tensors[2];
    assert_eq!(t2.name, "output0");
    assert_eq!(t2.shape, vec![1, 8]);
    assert_eq!(t2.dtype, DType::Fp32);
}

#[test]
fn test_load_conv2d_relu_fixture() {
    let bytes = include_bytes!("fixtures/conv2d_relu.tosa");
    let graph = deserialize(bytes).expect("failed to parse conv2d_relu fixture");

    // Version
    assert_eq!(graph.version.major, 1);
    assert_eq!(graph.version.minor, 0);
    assert_eq!(graph.version.patch, 0);
    assert_eq!(graph.version.draft, false);

    // Region and block structure
    assert_eq!(graph.regions.len(), 1);
    let region = &graph.regions[0];
    assert_eq!(region.name, "main");
    assert_eq!(region.blocks.len(), 1);
    let block = &region.blocks[0];
    assert_eq!(block.name, "block0");
    assert_eq!(block.inputs, vec!["input"]);
    assert_eq!(block.outputs, vec!["output"]);

    // Two operators: CONV2D then CLAMP
    assert_eq!(block.operators.len(), 2);

    // Conv2d operator
    let conv_op = &block.operators[0];
    assert_eq!(conv_op.op, Op::Conv2d);
    assert_eq!(conv_op.inputs, vec!["input", "weight", "bias"]);
    assert_eq!(conv_op.outputs, vec!["conv_out"]);
    match &conv_op.attribute {
        Attribute::Conv2d(attr) => {
            assert_eq!(attr.pad, vec![1, 1, 1, 1]);
            assert_eq!(attr.stride, vec![1, 1]);
            assert_eq!(attr.dilation, vec![1, 1]);
            assert_eq!(attr.local_bound, false);
            assert_eq!(attr.acc_type, DType::Fp32);
        }
        other => panic!("Expected Conv2d attribute, got {:?}", other),
    }

    // Clamp operator (ReLU)
    let clamp_op = &block.operators[1];
    assert_eq!(clamp_op.op, Op::Clamp);
    assert_eq!(clamp_op.inputs, vec!["conv_out"]);
    assert_eq!(clamp_op.outputs, vec!["output"]);
    match &clamp_op.attribute {
        Attribute::Clamp(attr) => {
            assert_eq!(attr.min_val, vec![0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(attr.max_val, vec![255, 255, 255, 255, 255, 255, 255, 255]);
            assert_eq!(attr.nan_mode, NanPropagationMode::Propagate);
        }
        other => panic!("Expected Clamp attribute, got {:?}", other),
    }

    // Tensors
    assert_eq!(block.tensors.len(), 5);

    let input = &block.tensors[0];
    assert_eq!(input.name, "input");
    assert_eq!(input.shape, vec![1, 3, 224, 224]);
    assert_eq!(input.dtype, DType::Fp32);

    let weight = &block.tensors[1];
    assert_eq!(weight.name, "weight");
    assert_eq!(weight.shape, vec![16, 3, 3, 3]);
    assert_eq!(weight.dtype, DType::Fp32);

    let bias = &block.tensors[2];
    assert_eq!(bias.name, "bias");
    assert_eq!(bias.shape, vec![16]);
    assert_eq!(bias.dtype, DType::Fp32);

    let conv_out = &block.tensors[3];
    assert_eq!(conv_out.name, "conv_out");
    assert_eq!(conv_out.shape, vec![1, 16, 224, 224]);
    assert_eq!(conv_out.dtype, DType::Fp32);

    let output = &block.tensors[4];
    assert_eq!(output.name, "output");
    assert_eq!(output.shape, vec![1, 16, 224, 224]);
    assert_eq!(output.dtype, DType::Fp32);
}
