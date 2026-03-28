use std::path::PathBuf;

use tosa::serialize::{deserialize, load_file, save_file, serialize};
use tosa::types::*;

/// Helper: serialize then deserialize, returning the roundtripped graph.
fn roundtrip(graph: &TosaGraph) -> TosaGraph {
    let bytes = serialize(graph);
    deserialize(&bytes).expect("deserialize failed")
}

// -------------------------------------------------------------------------
// 1. Empty graph (version only, empty regions vec)
// -------------------------------------------------------------------------

#[test]
fn test_empty_graph_roundtrip() {
    let graph = TosaGraph {
        version: Version {
            major: 1,
            minor: 0,
            patch: 0,
            draft: false,
        },
        regions: vec![],
    };
    let rt = roundtrip(&graph);
    assert_eq!(rt.version, graph.version);
    assert!(rt.regions.is_empty());
}

// -------------------------------------------------------------------------
// 2. One region, one empty block
// -------------------------------------------------------------------------

#[test]
fn test_one_region_one_empty_block() {
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
                name: "entry".to_string(),
                operators: vec![],
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
            }],
        }],
    };
    let rt = roundtrip(&graph);
    assert_eq!(rt.regions.len(), 1);
    assert_eq!(rt.regions[0].name, "main");
    assert_eq!(rt.regions[0].blocks.len(), 1);
    assert_eq!(rt.regions[0].blocks[0].name, "entry");
    assert!(rt.regions[0].blocks[0].operators.is_empty());
    assert!(rt.regions[0].blocks[0].tensors.is_empty());
}

// -------------------------------------------------------------------------
// 3. One region, one block with operators and tensors
// -------------------------------------------------------------------------

#[test]
fn test_one_region_one_block_with_ops_and_tensors() {
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
                name: "bb0".to_string(),
                operators: vec![
                    TosaOperator {
                        op: Op::Add,
                        attribute: Attribute::Add(AddAttribute {}),
                        inputs: vec!["t0".to_string(), "t1".to_string()],
                        outputs: vec!["t2".to_string()],
                    },
                    TosaOperator {
                        op: Op::Reshape,
                        attribute: Attribute::Reshape(ReshapeAttribute {}),
                        inputs: vec!["t2".to_string()],
                        outputs: vec!["t3".to_string()],
                    },
                ],
                tensors: vec![
                    TosaTensor {
                        name: "t0".to_string(),
                        shape: vec![1, 3, 224, 224],
                        dtype: DType::Fp32,
                        data: None,
                        variable: false,
                        is_unranked: false,
                        variable_name: None,
                    },
                    TosaTensor {
                        name: "t1".to_string(),
                        shape: vec![1, 3, 224, 224],
                        dtype: DType::Fp32,
                        data: Some(vec![0u8; 16]),
                        variable: false,
                        is_unranked: false,
                        variable_name: None,
                    },
                ],
                inputs: vec!["t0".to_string()],
                outputs: vec!["t3".to_string()],
            }],
        }],
    };
    let rt = roundtrip(&graph);
    let block = &rt.regions[0].blocks[0];
    assert_eq!(block.operators.len(), 2);
    assert_eq!(block.operators[0].op, Op::Add);
    assert_eq!(block.operators[1].op, Op::Reshape);
    assert_eq!(block.tensors.len(), 2);
    assert_eq!(block.tensors[0].name, "t0");
    assert_eq!(block.tensors[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(block.tensors[0].dtype, DType::Fp32);
    assert_eq!(block.tensors[1].data, Some(vec![0u8; 16]));
}

// -------------------------------------------------------------------------
// 4. Multiple regions
// -------------------------------------------------------------------------

#[test]
fn test_multiple_regions() {
    let graph = TosaGraph {
        version: Version {
            major: 0,
            minor: 80,
            patch: 0,
            draft: true,
        },
        regions: vec![
            TosaRegion {
                name: "region_a".to_string(),
                blocks: vec![TosaBasicBlock {
                    name: "block_a".to_string(),
                    ..Default::default()
                }],
            },
            TosaRegion {
                name: "region_b".to_string(),
                blocks: vec![TosaBasicBlock {
                    name: "block_b".to_string(),
                    ..Default::default()
                }],
            },
            TosaRegion {
                name: "region_c".to_string(),
                blocks: vec![],
            },
        ],
    };
    let rt = roundtrip(&graph);
    assert_eq!(rt.regions.len(), 3);
    assert_eq!(rt.regions[0].name, "region_a");
    assert_eq!(rt.regions[1].name, "region_b");
    assert_eq!(rt.regions[2].name, "region_c");
    assert_eq!(rt.regions[0].blocks[0].name, "block_a");
    assert_eq!(rt.regions[1].blocks[0].name, "block_b");
    assert!(rt.regions[2].blocks.is_empty());
}

// -------------------------------------------------------------------------
// 5. Multiple blocks per region
// -------------------------------------------------------------------------

#[test]
fn test_multiple_blocks_per_region() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![
                TosaBasicBlock {
                    name: "entry".to_string(),
                    ..Default::default()
                },
                TosaBasicBlock {
                    name: "then_block".to_string(),
                    ..Default::default()
                },
                TosaBasicBlock {
                    name: "else_block".to_string(),
                    ..Default::default()
                },
            ],
        }],
    };
    let rt = roundtrip(&graph);
    assert_eq!(rt.regions[0].blocks.len(), 3);
    assert_eq!(rt.regions[0].blocks[0].name, "entry");
    assert_eq!(rt.regions[0].blocks[1].name, "then_block");
    assert_eq!(rt.regions[0].blocks[2].name, "else_block");
}

// -------------------------------------------------------------------------
// 6. Version fields: various major/minor/patch, draft true and false
// -------------------------------------------------------------------------

#[test]
fn test_version_fields() {
    let cases = vec![
        Version {
            major: 0,
            minor: 0,
            patch: 0,
            draft: false,
        },
        Version {
            major: 1,
            minor: 2,
            patch: 3,
            draft: false,
        },
        Version {
            major: 100,
            minor: 200,
            patch: 300,
            draft: true,
        },
        Version {
            major: -1,
            minor: -1,
            patch: -1,
            draft: true,
        },
    ];
    for v in &cases {
        let graph = TosaGraph {
            version: v.clone(),
            regions: vec![],
        };
        let rt = roundtrip(&graph);
        assert_eq!(
            rt.version, *v,
            "Version roundtrip failed for {:?}",
            v
        );
    }
}

// -------------------------------------------------------------------------
// 7. Block inputs/outputs string lists roundtrip
// -------------------------------------------------------------------------

#[test]
fn test_block_inputs_outputs_roundtrip() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![TosaRegion {
            name: "main".to_string(),
            blocks: vec![TosaBasicBlock {
                name: "bb0".to_string(),
                operators: vec![],
                tensors: vec![],
                inputs: vec![
                    "input_0".to_string(),
                    "input_1".to_string(),
                    "input_2".to_string(),
                ],
                outputs: vec!["output_0".to_string(), "output_1".to_string()],
            }],
        }],
    };
    let rt = roundtrip(&graph);
    let block = &rt.regions[0].blocks[0];
    assert_eq!(
        block.inputs,
        vec!["input_0", "input_1", "input_2"]
    );
    assert_eq!(block.outputs, vec!["output_0", "output_1"]);
}

// -------------------------------------------------------------------------
// 8. "TOSA" file identifier in serialized bytes
// -------------------------------------------------------------------------

#[test]
fn test_tosa_file_identifier_in_bytes() {
    let graph = TosaGraph {
        version: Version::default(),
        regions: vec![],
    };
    let bytes = serialize(&graph);
    // FlatBuffers file identifier sits at bytes[4..8]
    assert!(
        bytes.len() >= 8,
        "serialized bytes too short: {}",
        bytes.len()
    );
    assert_eq!(
        &bytes[4..8],
        b"TOSA",
        "Expected TOSA file identifier at bytes[4..8]"
    );
}

// -------------------------------------------------------------------------
// 9. load_file / save_file with a temp file
// -------------------------------------------------------------------------

#[test]
fn test_save_and_load_file() {
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
                name: "entry".to_string(),
                operators: vec![TosaOperator {
                    op: Op::Identity,
                    attribute: Attribute::Identity(IdentityAttribute {}),
                    inputs: vec!["in".to_string()],
                    outputs: vec!["out".to_string()],
                }],
                tensors: vec![TosaTensor {
                    name: "in".to_string(),
                    shape: vec![2, 3],
                    dtype: DType::Int32,
                    data: None,
                    variable: false,
                    is_unranked: false,
                    variable_name: None,
                }],
                inputs: vec!["in".to_string()],
                outputs: vec!["out".to_string()],
            }],
        }],
    };

    let tmp_dir = std::env::temp_dir();
    let file_path: PathBuf = tmp_dir.join("rust_tosa_test_graph.tosa");

    save_file(&graph, &file_path).expect("save_file failed");
    let loaded = load_file(&file_path).expect("load_file failed");

    // Clean up
    let _ = std::fs::remove_file(&file_path);

    assert_eq!(loaded.version, graph.version);
    assert_eq!(loaded.regions.len(), 1);
    assert_eq!(loaded.regions[0].blocks[0].name, "entry");
    assert_eq!(loaded.regions[0].blocks[0].operators[0].op, Op::Identity);
    assert_eq!(
        loaded.regions[0].blocks[0].tensors[0].name,
        "in"
    );
}

// -------------------------------------------------------------------------
// 10. Operators reference tensors by name; name connectivity survives roundtrip
// -------------------------------------------------------------------------

#[test]
fn test_operator_tensor_name_connectivity() {
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
                name: "bb0".to_string(),
                operators: vec![
                    TosaOperator {
                        op: Op::Const,
                        attribute: Attribute::Const(ConstAttribute {}),
                        inputs: vec![],
                        outputs: vec!["weight".to_string()],
                    },
                    TosaOperator {
                        op: Op::Add,
                        attribute: Attribute::Add(AddAttribute {}),
                        inputs: vec!["input".to_string(), "weight".to_string()],
                        outputs: vec!["sum_out".to_string()],
                    },
                    TosaOperator {
                        op: Op::Sigmoid,
                        attribute: Attribute::Sigmoid(SigmoidAttribute {}),
                        inputs: vec!["sum_out".to_string()],
                        outputs: vec!["result".to_string()],
                    },
                ],
                tensors: vec![
                    TosaTensor {
                        name: "input".to_string(),
                        shape: vec![1, 64],
                        dtype: DType::Fp32,
                        data: None,
                        variable: false,
                        is_unranked: false,
                        variable_name: None,
                    },
                    TosaTensor {
                        name: "weight".to_string(),
                        shape: vec![1, 64],
                        dtype: DType::Fp32,
                        data: Some(vec![1, 2, 3, 4]),
                        variable: false,
                        is_unranked: false,
                        variable_name: None,
                    },
                    TosaTensor {
                        name: "sum_out".to_string(),
                        shape: vec![1, 64],
                        dtype: DType::Fp32,
                        data: None,
                        variable: false,
                        is_unranked: false,
                        variable_name: None,
                    },
                    TosaTensor {
                        name: "result".to_string(),
                        shape: vec![1, 64],
                        dtype: DType::Fp32,
                        data: None,
                        variable: false,
                        is_unranked: false,
                        variable_name: None,
                    },
                ],
                inputs: vec!["input".to_string()],
                outputs: vec!["result".to_string()],
            }],
        }],
    };

    let rt = roundtrip(&graph);
    let block = &rt.regions[0].blocks[0];

    // Build a set of tensor names for lookup
    let tensor_names: Vec<&str> = block.tensors.iter().map(|t| t.name.as_str()).collect();

    // Verify every operator input/output name references a known tensor
    for op in &block.operators {
        for inp in &op.inputs {
            assert!(
                tensor_names.contains(&inp.as_str()),
                "Operator {:?} input '{}' not found in tensor names {:?}",
                op.op,
                inp,
                tensor_names
            );
        }
        for out in &op.outputs {
            assert!(
                tensor_names.contains(&out.as_str()),
                "Operator {:?} output '{}' not found in tensor names {:?}",
                op.op,
                out,
                tensor_names
            );
        }
    }

    // Verify the dataflow chain: const->weight, add(input,weight)->sum_out, sigmoid(sum_out)->result
    assert_eq!(block.operators[0].outputs, vec!["weight"]);
    assert_eq!(
        block.operators[1].inputs,
        vec!["input", "weight"]
    );
    assert_eq!(block.operators[1].outputs, vec!["sum_out"]);
    assert_eq!(block.operators[2].inputs, vec!["sum_out"]);
    assert_eq!(block.operators[2].outputs, vec!["result"]);
}
