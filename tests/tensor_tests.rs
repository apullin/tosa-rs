use tosa_rs::{
    DType, TosaBasicBlock, TosaGraph, TosaRegion, TosaTensor, Version,
    serialize::{serialize, deserialize},
};

fn roundtrip(graph: &TosaGraph) -> TosaGraph {
    let bytes = serialize(graph);
    deserialize(&bytes).unwrap()
}

/// Wrap a single tensor in a minimal TosaGraph for roundtrip testing.
fn wrap_tensor(tensor: TosaTensor) -> TosaGraph {
    TosaGraph {
        version: Version {
            major: 1,
            minor: 0,
            patch: 0,
            draft: false,
        },
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "entry".to_string(),
                operators: vec![],
                tensors: vec![tensor],
                inputs: vec![],
                outputs: vec![],
                ..Default::default()
            }],
        }],
        ..Default::default()
    }
}

/// Wrap multiple tensors in a minimal TosaGraph for roundtrip testing.
fn wrap_tensors(tensors: Vec<TosaTensor>) -> TosaGraph {
    TosaGraph {
        version: Version {
            major: 1,
            minor: 0,
            patch: 0,
            draft: false,
        },
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "entry".to_string(),
                operators: vec![],
                tensors,
                inputs: vec![],
                outputs: vec![],
                ..Default::default()
            }],
        }],
        ..Default::default()
    }
}

/// Extract the first tensor from a roundtripped graph.
fn first_tensor(graph: &TosaGraph) -> &TosaTensor {
    &graph.regions[0].blocks[0].tensors[0]
}

// -----------------------------------------------------------------------
// 1. Tensor with each DType variant
// -----------------------------------------------------------------------

#[test]
fn tensor_dtype_fp32() {
    let t = TosaTensor::new("t_fp32", vec![2, 3], DType::Fp32);
    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);
    assert_eq!(out.name, "t_fp32");
    assert_eq!(out.dtype, DType::Fp32);
    assert_eq!(out.shape, vec![2, 3]);
}

#[test]
fn tensor_dtype_int8() {
    let t = TosaTensor::new("t_int8", vec![4], DType::Int8);
    let rt = roundtrip(&wrap_tensor(t));
    assert_eq!(first_tensor(&rt).dtype, DType::Int8);
}

#[test]
fn tensor_dtype_bool() {
    let t = TosaTensor::new("t_bool", vec![1], DType::Bool);
    let rt = roundtrip(&wrap_tensor(t));
    assert_eq!(first_tensor(&rt).dtype, DType::Bool);
}

#[test]
fn tensor_dtype_fp16() {
    let t = TosaTensor::new("t_fp16", vec![8], DType::Fp16);
    let rt = roundtrip(&wrap_tensor(t));
    assert_eq!(first_tensor(&rt).dtype, DType::Fp16);
}

#[test]
fn tensor_dtype_bf16() {
    let t = TosaTensor::new("t_bf16", vec![5], DType::Bf16);
    let rt = roundtrip(&wrap_tensor(t));
    assert_eq!(first_tensor(&rt).dtype, DType::Bf16);
}

#[test]
fn tensor_dtype_int4() {
    let t = TosaTensor::new("t_int4", vec![16], DType::Int4);
    let rt = roundtrip(&wrap_tensor(t));
    assert_eq!(first_tensor(&rt).dtype, DType::Int4);
}

#[test]
fn tensor_dtype_int32() {
    let t = TosaTensor::new("t_int32", vec![10], DType::Int32);
    let rt = roundtrip(&wrap_tensor(t));
    assert_eq!(first_tensor(&rt).dtype, DType::Int32);
}

// -----------------------------------------------------------------------
// 2. Tensor with data payload
// -----------------------------------------------------------------------

#[test]
fn tensor_with_data_payload() {
    let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = TosaTensor::with_data("const0", vec![4], DType::Fp32, data.clone());

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "const0");
    assert_eq!(out.dtype, DType::Fp32);
    assert_eq!(out.shape, vec![4]);
    assert!(out.data.is_some());
    assert_eq!(out.data.as_ref().unwrap(), &data);

    // Verify the actual float values survived the roundtrip.
    let recovered: Vec<f32> = out
        .data
        .as_ref()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(recovered, vec![1.0, 2.0, 3.0, 4.0]);
}

// -----------------------------------------------------------------------
// 3. Tensor without data (None)
// -----------------------------------------------------------------------

#[test]
fn tensor_without_data() {
    let t = TosaTensor::new("no_data", vec![2, 2], DType::Int32);
    assert!(t.data.is_none());

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "no_data");
    assert!(out.data.is_none());
}

// -----------------------------------------------------------------------
// 4. Variable tensor with variable_name
// -----------------------------------------------------------------------

#[test]
fn variable_tensor_with_variable_name() {
    let t = TosaTensor::variable("var_tensor", vec![3, 3], DType::Fp32, "my_variable");

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "var_tensor");
    assert!(out.variable);
    assert_eq!(out.variable_name.as_deref(), Some("my_variable"));
    assert_eq!(out.dtype, DType::Fp32);
    assert_eq!(out.shape, vec![3, 3]);
}

// -----------------------------------------------------------------------
// 5. Unranked tensor (is_unranked = true)
// -----------------------------------------------------------------------

#[test]
fn unranked_tensor() {
    let t = TosaTensor {
        name: "unranked".to_string(),
        shape: vec![],
        dtype: DType::Fp32,
        data: None,
        variable: false,
        is_unranked: true,
        variable_name: None,
        ..Default::default()
    };

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "unranked");
    assert!(out.is_unranked);
    assert!(out.shape.is_empty());
}

// -----------------------------------------------------------------------
// 6. Scalar tensor (empty shape)
// -----------------------------------------------------------------------

#[test]
fn scalar_tensor_empty_shape() {
    let t = TosaTensor::new("scalar", vec![], DType::Int32);

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "scalar");
    assert!(out.shape.is_empty());
    assert!(!out.is_unranked);
    assert_eq!(out.dtype, DType::Int32);
}

// -----------------------------------------------------------------------
// 7. Multi-dimensional tensor shape
// -----------------------------------------------------------------------

#[test]
fn multidimensional_tensor_shape() {
    let shape = vec![1, 3, 224, 224];
    let t = TosaTensor::new("image", shape.clone(), DType::Fp32);

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "image");
    assert_eq!(out.shape, shape);
}

// -----------------------------------------------------------------------
// 8. Tensor with empty name
// -----------------------------------------------------------------------

#[test]
fn tensor_with_empty_name() {
    let t = TosaTensor::new("", vec![2], DType::Fp32);

    let rt = roundtrip(&wrap_tensor(t));
    let out = first_tensor(&rt);

    assert_eq!(out.name, "");
    assert_eq!(out.shape, vec![2]);
    assert_eq!(out.dtype, DType::Fp32);
}

// -----------------------------------------------------------------------
// 9. Multiple tensors in one block
// -----------------------------------------------------------------------

#[test]
fn multiple_tensors_in_one_block() {
    let tensors = vec![
        TosaTensor::new("input", vec![1, 3, 224, 224], DType::Fp32),
        TosaTensor::with_data(
            "weights",
            vec![64, 3, 7, 7],
            DType::Fp32,
            vec![0u8; 64 * 3 * 7 * 7 * 4],
        ),
        TosaTensor::new("output", vec![1, 64, 112, 112], DType::Fp32),
        TosaTensor::variable("state", vec![1], DType::Int32, "counter"),
        TosaTensor {
            name: "dynamic".to_string(),
            shape: vec![],
            dtype: DType::Bf16,
            data: None,
            variable: false,
            is_unranked: true,
            variable_name: None,
            ..Default::default()
        },
    ];

    let rt = roundtrip(&wrap_tensors(tensors));
    let block = &rt.regions[0].blocks[0];
    assert_eq!(block.tensors.len(), 5);

    // Verify each tensor preserved its properties.
    assert_eq!(block.tensors[0].name, "input");
    assert_eq!(block.tensors[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(block.tensors[0].dtype, DType::Fp32);
    assert!(block.tensors[0].data.is_none());

    assert_eq!(block.tensors[1].name, "weights");
    assert_eq!(block.tensors[1].shape, vec![64, 3, 7, 7]);
    assert!(block.tensors[1].data.is_some());
    assert_eq!(block.tensors[1].data.as_ref().unwrap().len(), 64 * 3 * 7 * 7 * 4);

    assert_eq!(block.tensors[2].name, "output");
    assert_eq!(block.tensors[2].shape, vec![1, 64, 112, 112]);

    assert_eq!(block.tensors[3].name, "state");
    assert!(block.tensors[3].variable);
    assert_eq!(block.tensors[3].variable_name.as_deref(), Some("counter"));

    assert_eq!(block.tensors[4].name, "dynamic");
    assert!(block.tensors[4].is_unranked);
    assert_eq!(block.tensors[4].dtype, DType::Bf16);
}
