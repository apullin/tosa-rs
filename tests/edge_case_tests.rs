use tosa::error::TosaError;
use tosa::serialize::{deserialize, serialize};
use tosa::types::*;

// ---------------------------------------------------------------------------
// Helper: build a minimal valid TosaGraph
// ---------------------------------------------------------------------------

fn minimal_graph() -> TosaGraph {
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
                name: "block0".to_string(),
                operators: vec![TosaOperator {
                    op: Op::Add,
                    attribute: Attribute::Add(AddAttribute {}),
                    inputs: vec!["input0".to_string(), "input1".to_string()],
                    outputs: vec!["output0".to_string()],
                }],
                tensors: vec![TosaTensor {
                    name: "input0".to_string(),
                    shape: vec![1, 4],
                    dtype: DType::Fp32,
                    data: None,
                    variable: false,
                    is_unranked: false,
                    variable_name: None,
                }],
                inputs: vec!["input0".to_string(), "input1".to_string()],
                outputs: vec!["output0".to_string()],
            }],
        }],
    }
}

// ===========================================================================
// 1. Empty bytes should return Err
// ===========================================================================

#[test]
fn deserialize_empty_bytes_returns_err() {
    let result = deserialize(&[]);
    assert!(result.is_err(), "deserializing empty bytes must return Err");
    // Verify it's the right error variant
    match result.unwrap_err() {
        TosaError::InvalidFlatbuffer(_) => {} // expected
        other => panic!("expected InvalidFlatbuffer, got: {:?}", other),
    }
}

// ===========================================================================
// 2. Random garbage should return Err, not panic
// ===========================================================================

#[test]
fn deserialize_garbage_bytes_returns_err() {
    let result = deserialize(&[1, 2, 3, 4]);
    assert!(
        result.is_err(),
        "deserializing random garbage must return Err"
    );
}

// ===========================================================================
// 3. Zeroed buffer should return Err or at least not panic
// ===========================================================================

#[test]
fn deserialize_zeroed_buffer_does_not_panic() {
    let buf = [0u8; 100];
    let result = deserialize(&buf);
    // It's fine if it returns Ok with some default-ish graph or Err,
    // but it must NEVER panic.
    let _ = result;
}

// ===========================================================================
// 4. Truncated valid flatbuffer
// ===========================================================================

#[test]
fn deserialize_truncated_flatbuffer_returns_err() {
    let graph = minimal_graph();
    let bytes = serialize(&graph);

    // Chop off the last 10 bytes
    assert!(
        bytes.len() > 10,
        "serialized graph must be longer than 10 bytes"
    );
    let truncated = &bytes[..bytes.len() - 10];

    let result = deserialize(truncated);
    assert!(
        result.is_err(),
        "deserializing truncated flatbuffer must return Err, but got Ok"
    );
}

// ===========================================================================
// 5. Large tensor data payload (1MB+)
// ===========================================================================

#[test]
fn large_tensor_data_roundtrip() {
    let data_size = 1024 * 1024 + 37; // 1MB + 37 bytes
    let large_data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

    let graph = TosaGraph {
        version: Version {
            major: 1,
            minor: 0,
            patch: 0,
            draft: false,
        },
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![],
                tensors: vec![TosaTensor {
                    name: "big_tensor".to_string(),
                    shape: vec![1, data_size as i32],
                    dtype: DType::Uint8,
                    data: Some(large_data.clone()),
                    variable: false,
                    is_unranked: false,
                    variable_name: None,
                }],
                inputs: vec![],
                outputs: vec![],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize large tensor");

    let tensor = &round.regions[0].blocks[0].tensors[0];
    assert_eq!(tensor.name, "big_tensor");
    assert_eq!(tensor.data.as_ref().unwrap().len(), data_size);
    assert_eq!(tensor.data.as_ref().unwrap(), &large_data);
}

// ===========================================================================
// 6. Graph with many regions (50)
// ===========================================================================

#[test]
fn many_regions_roundtrip() {
    let regions: Vec<TosaRegion> = (0..50)
        .map(|i| TosaRegion {
            name: format!("region_{}", i),
            blocks: vec![TosaBasicBlock {
                name: format!("block_in_region_{}", i),
                operators: vec![],
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
            }],
        })
        .collect();

    let graph = TosaGraph {
        version: Version::default(),
        regions,
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize many regions");

    assert_eq!(round.regions.len(), 50);
    for i in 0..50 {
        assert_eq!(round.regions[i].name, format!("region_{}", i));
        assert_eq!(
            round.regions[i].blocks[0].name,
            format!("block_in_region_{}", i)
        );
    }
}

// ===========================================================================
// 7. Graph with many operators in a block (100)
// ===========================================================================

#[test]
fn many_operators_in_block_roundtrip() {
    let operators: Vec<TosaOperator> = (0..100)
        .map(|i| TosaOperator {
            op: Op::Add,
            attribute: Attribute::Add(AddAttribute {}),
            inputs: vec![format!("in_{}", i)],
            outputs: vec![format!("out_{}", i)],
        })
        .collect();

    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators,
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize many operators");

    let block = &round.regions[0].blocks[0];
    assert_eq!(block.operators.len(), 100);
    for i in 0..100 {
        assert_eq!(block.operators[i].inputs, vec![format!("in_{}", i)]);
        assert_eq!(block.operators[i].outputs, vec![format!("out_{}", i)]);
    }
}

// ===========================================================================
// 8. Operator with empty inputs and outputs lists
// ===========================================================================

#[test]
fn operator_empty_inputs_outputs_roundtrip() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![TosaOperator {
                    op: Op::Const,
                    attribute: Attribute::Const(ConstAttribute {}),
                    inputs: vec![],
                    outputs: vec![],
                }],
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize empty inputs/outputs");

    let op = &round.regions[0].blocks[0].operators[0];
    assert_eq!(op.op, Op::Const);
    assert!(op.inputs.is_empty());
    assert!(op.outputs.is_empty());
}

// ===========================================================================
// 9. Tensor with empty name
// ===========================================================================

#[test]
fn tensor_empty_name_roundtrip() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![],
                tensors: vec![TosaTensor {
                    name: "".to_string(),
                    shape: vec![2, 3],
                    dtype: DType::Int32,
                    data: None,
                    variable: false,
                    is_unranked: false,
                    variable_name: None,
                }],
                inputs: vec![],
                outputs: vec![],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize empty-name tensor");

    let tensor = &round.regions[0].blocks[0].tensors[0];
    assert_eq!(tensor.name, "");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.dtype, DType::Int32);
}

// ===========================================================================
// 10. Unicode in names
// ===========================================================================

#[test]
fn unicode_names_roundtrip() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "領域".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "блок".to_string(),
                operators: vec![],
                tensors: vec![TosaTensor {
                    name: "输入".to_string(),
                    shape: vec![1],
                    dtype: DType::Fp32,
                    data: None,
                    variable: false,
                    is_unranked: false,
                    variable_name: None,
                }],
                inputs: vec!["输入".to_string()],
                outputs: vec![],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize unicode names");

    assert_eq!(round.regions[0].name, "領域");
    assert_eq!(round.regions[0].blocks[0].name, "блок");
    assert_eq!(round.regions[0].blocks[0].tensors[0].name, "输入");
    assert_eq!(round.regions[0].blocks[0].inputs, vec!["输入".to_string()]);
}

// ===========================================================================
// 11. ClampAttribute with specific byte patterns
// ===========================================================================

#[test]
fn clamp_attribute_byte_patterns_roundtrip() {
    // Encode f32 min/max as little-endian bytes, which is the TOSA convention
    let min_val = (-1.0f32).to_le_bytes().to_vec();
    let max_val = (1.0f32).to_le_bytes().to_vec();

    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![TosaOperator {
                    op: Op::Clamp,
                    attribute: Attribute::Clamp(ClampAttribute {
                        min_val: min_val.clone(),
                        max_val: max_val.clone(),
                        nan_mode: NanPropagationMode::Propagate,
                    }),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                }],
                tensors: vec![],
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize clamp");

    let op = &round.regions[0].blocks[0].operators[0];
    match &op.attribute {
        Attribute::Clamp(c) => {
            assert_eq!(c.min_val, min_val, "min_val bytes mismatch");
            assert_eq!(c.max_val, max_val, "max_val bytes mismatch");
            assert_eq!(c.nan_mode, NanPropagationMode::Propagate);

            // Verify we can decode back to f32
            let decoded_min = f32::from_le_bytes(c.min_val[..4].try_into().unwrap());
            let decoded_max = f32::from_le_bytes(c.max_val[..4].try_into().unwrap());
            assert_eq!(decoded_min, -1.0f32);
            assert_eq!(decoded_max, 1.0f32);
        }
        other => panic!("expected Clamp attribute, got: {:?}", other),
    }
}

// ===========================================================================
// 12. CustomAttribute with all fields populated
// ===========================================================================

#[test]
fn custom_attribute_all_fields_roundtrip() {
    let impl_attrs = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];

    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![TosaOperator {
                    op: Op::Custom,
                    attribute: Attribute::Custom(CustomAttribute {
                        operator_name: "my_custom_op".to_string(),
                        domain_name: "com.example.ops".to_string(),
                        implementation_attrs: impl_attrs.clone(),
                    }),
                    inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                    outputs: vec!["result".to_string()],
                }],
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize custom");

    let op = &round.regions[0].blocks[0].operators[0];
    assert_eq!(op.op, Op::Custom);
    match &op.attribute {
        Attribute::Custom(c) => {
            assert_eq!(c.operator_name, "my_custom_op");
            assert_eq!(c.domain_name, "com.example.ops");
            assert_eq!(c.implementation_attrs, impl_attrs);
        }
        other => panic!("expected Custom attribute, got: {:?}", other),
    }
    assert_eq!(op.inputs, vec!["a", "b", "c"]);
    assert_eq!(op.outputs, vec!["result"]);
}

// ===========================================================================
// 13. CondIfAttribute and WhileLoopAttribute graph name references
// ===========================================================================

#[test]
fn cond_if_attribute_graph_names_roundtrip() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![TosaOperator {
                    op: Op::CondIf,
                    attribute: Attribute::CondIf(CondIfAttribute {
                        then_graph: "then_branch_region".to_string(),
                        else_graph: "else_branch_region".to_string(),
                    }),
                    inputs: vec!["cond".to_string(), "input".to_string()],
                    outputs: vec!["result".to_string()],
                }],
                tensors: vec![],
                inputs: vec!["cond".to_string(), "input".to_string()],
                outputs: vec!["result".to_string()],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize cond_if");

    let op = &round.regions[0].blocks[0].operators[0];
    assert_eq!(op.op, Op::CondIf);
    match &op.attribute {
        Attribute::CondIf(c) => {
            assert_eq!(c.then_graph, "then_branch_region");
            assert_eq!(c.else_graph, "else_branch_region");
        }
        other => panic!("expected CondIf attribute, got: {:?}", other),
    }
}

#[test]
fn while_loop_attribute_graph_names_roundtrip() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "block0".to_string(),
                operators: vec![TosaOperator {
                    op: Op::WhileLoop,
                    attribute: Attribute::WhileLoop(WhileLoopAttribute {
                        cond_graph: "loop_condition".to_string(),
                        body_graph: "loop_body".to_string(),
                    }),
                    inputs: vec!["init".to_string()],
                    outputs: vec!["final".to_string()],
                }],
                tensors: vec![],
                inputs: vec!["init".to_string()],
                outputs: vec!["final".to_string()],
            }],
        }],
    };

    let bytes = serialize(&graph);
    let round = deserialize(&bytes).expect("deserialize while_loop");

    let op = &round.regions[0].blocks[0].operators[0];
    assert_eq!(op.op, Op::WhileLoop);
    match &op.attribute {
        Attribute::WhileLoop(w) => {
            assert_eq!(w.cond_graph, "loop_condition");
            assert_eq!(w.body_graph, "loop_body");
        }
        other => panic!("expected WhileLoop attribute, got: {:?}", other),
    }
}
