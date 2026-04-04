use tosa_rs::types::*;

fn roundtrip_op(op: Op, attr: Attribute, inputs: &[&str], outputs: &[&str]) -> TosaOperator {
    let graph = TosaGraph {
        version: Version { major: 0, minor: 80, patch: 1, draft: false },
        regions: vec![TosaRegion {
            name: "main".into(),
            blocks: vec![TosaBasicBlock {
                name: "block0".into(),
                operators: vec![TosaOperator {
                    op,
                    attribute: attr,
                    inputs: inputs.iter().map(|s| s.to_string()).collect(),
                    outputs: outputs.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                }],
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
                ..Default::default()
            }],
        }],
        ..Default::default()
    };
    let bytes = tosa_rs::serialize::serialize(&graph);
    let result = tosa_rs::serialize::deserialize(&bytes).unwrap();
    result.regions[0].blocks[0].operators[0].clone()
}

// ===== Attributes with fields =====

#[test]
fn test_argmax() {
    let attr = Attribute::ArgMax(ArgMaxAttribute { axis: 2, nan_mode: NanPropagationMode::Propagate });
    let op_out = roundtrip_op(Op::ArgMax, attr.clone(), &["input"], &["output"]);
    assert_eq!(op_out.op, Op::ArgMax);
    assert_eq!(op_out.inputs, vec!["input"]);
    assert_eq!(op_out.outputs, vec!["output"]);
    match op_out.attribute {
        Attribute::ArgMax(a) => {
            assert_eq!(a.axis, 2);
            assert_eq!(a.nan_mode, NanPropagationMode::Propagate);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_avg_pool2d() {
    let attr = Attribute::AvgPool2d(AvgPool2dAttribute {
        kernel: vec![3, 3],
        stride: vec![1, 1],
        pad: vec![1, 1, 1, 1],
        acc_type: DType::Int32,
    });
    let op_out = roundtrip_op(Op::AvgPool2d, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::AvgPool2d);
    match op_out.attribute {
        Attribute::AvgPool2d(a) => {
            assert_eq!(a.kernel, vec![3, 3]);
            assert_eq!(a.stride, vec![1, 1]);
            assert_eq!(a.pad, vec![1, 1, 1, 1]);
            assert_eq!(a.acc_type, DType::Int32);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_conv2d() {
    let attr = Attribute::Conv2d(Conv2dAttribute {
        pad: vec![1, 1, 1, 1],
        stride: vec![2, 2],
        dilation: vec![1, 1],
        local_bound: true,
        acc_type: DType::Int32,
    });
    let op_out = roundtrip_op(Op::Conv2d, attr, &["input", "weight", "bias"], &["output"]);
    assert_eq!(op_out.op, Op::Conv2d);
    assert_eq!(op_out.inputs, vec!["input", "weight", "bias"]);
    match op_out.attribute {
        Attribute::Conv2d(a) => {
            assert_eq!(a.pad, vec![1, 1, 1, 1]);
            assert_eq!(a.stride, vec![2, 2]);
            assert_eq!(a.dilation, vec![1, 1]);
            assert!(a.local_bound);
            assert_eq!(a.acc_type, DType::Int32);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_conv3d() {
    let attr = Attribute::Conv3d(Conv3dAttribute {
        pad: vec![1, 1, 1, 1, 1, 1],
        stride: vec![2, 2, 2],
        dilation: vec![1, 1, 1],
        local_bound: true,
        acc_type: DType::Int32,
    });
    let op_out = roundtrip_op(Op::Conv3d, attr, &["input", "weight", "bias"], &["output"]);
    assert_eq!(op_out.op, Op::Conv3d);
    match op_out.attribute {
        Attribute::Conv3d(a) => {
            assert_eq!(a.pad, vec![1, 1, 1, 1, 1, 1]);
            assert_eq!(a.stride, vec![2, 2, 2]);
            assert_eq!(a.dilation, vec![1, 1, 1]);
            assert!(a.local_bound);
            assert_eq!(a.acc_type, DType::Int32);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_depthwise_conv2d() {
    let attr = Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
        pad: vec![1, 1, 1, 1],
        stride: vec![2, 2],
        dilation: vec![1, 1],
        local_bound: true,
        acc_type: DType::Int32,
    });
    let op_out = roundtrip_op(Op::DepthwiseConv2d, attr, &["input", "weight", "bias"], &["output"]);
    assert_eq!(op_out.op, Op::DepthwiseConv2d);
    match op_out.attribute {
        Attribute::DepthwiseConv2d(a) => {
            assert_eq!(a.pad, vec![1, 1, 1, 1]);
            assert_eq!(a.stride, vec![2, 2]);
            assert_eq!(a.dilation, vec![1, 1]);
            assert!(a.local_bound);
            assert_eq!(a.acc_type, DType::Int32);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_fft2d() {
    let attr = Attribute::FFT2d(FFT2dAttribute { inverse: true, local_bound: true });
    let op_out = roundtrip_op(Op::Fft2d, attr, &["in_real", "in_imag"], &["out_real", "out_imag"]);
    assert_eq!(op_out.op, Op::Fft2d);
    match op_out.attribute {
        Attribute::FFT2d(a) => {
            assert!(a.inverse);
            assert!(a.local_bound);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_max_pool2d() {
    let attr = Attribute::MaxPool2d(MaxPool2dAttribute {
        kernel: vec![2, 2],
        stride: vec![2, 2],
        pad: vec![0, 0, 0, 0],
        nan_mode: NanPropagationMode::Ignore,
    });
    let op_out = roundtrip_op(Op::MaxPool2d, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::MaxPool2d);
    match op_out.attribute {
        Attribute::MaxPool2d(a) => {
            assert_eq!(a.kernel, vec![2, 2]);
            assert_eq!(a.stride, vec![2, 2]);
            assert_eq!(a.pad, vec![0, 0, 0, 0]);
            assert_eq!(a.nan_mode, NanPropagationMode::Ignore);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_rfft2d() {
    let attr = Attribute::RFFT2d(RFFT2dAttribute { local_bound: true });
    let op_out = roundtrip_op(Op::Rfft2d, attr, &["in"], &["out_real", "out_imag"]);
    assert_eq!(op_out.op, Op::Rfft2d);
    match op_out.attribute {
        Attribute::RFFT2d(a) => {
            assert!(a.local_bound);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_transpose_conv2d() {
    let attr = Attribute::TransposeConv2d(TransposeConv2dAttribute {
        out_pad: vec![0, 0],
        stride: vec![2, 2],
        local_bound: false,
        acc_type: DType::Fp32,
    });
    let op_out = roundtrip_op(Op::TransposeConv2d, attr, &["input", "weight", "bias"], &["output"]);
    assert_eq!(op_out.op, Op::TransposeConv2d);
    match op_out.attribute {
        Attribute::TransposeConv2d(a) => {
            assert_eq!(a.out_pad, vec![0, 0]);
            assert_eq!(a.stride, vec![2, 2]);
            assert!(!a.local_bound);
            assert_eq!(a.acc_type, DType::Fp32);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_clamp() {
    let attr = Attribute::Clamp(ClampAttribute {
        min_val: vec![0, 0, 0, 0, 0, 0, 0, 0],
        max_val: vec![0, 0, 128, 63, 0, 0, 0, 0],
        nan_mode: NanPropagationMode::Propagate,
    });
    let op_out = roundtrip_op(Op::Clamp, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::Clamp);
    match op_out.attribute {
        Attribute::Clamp(a) => {
            assert_eq!(a.min_val, vec![0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(a.max_val, vec![0, 0, 128, 63, 0, 0, 0, 0]);
            assert_eq!(a.nan_mode, NanPropagationMode::Propagate);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_arithmetic_right_shift() {
    let attr = Attribute::ArithmeticRightShift(ArithmeticRightShiftAttribute {});
    let op_out = roundtrip_op(Op::ArithmeticRightShift, attr, &["a", "b"], &["out"]);
    assert_eq!(op_out.op, Op::ArithmeticRightShift);
    assert!(matches!(op_out.attribute, Attribute::ArithmeticRightShift(_)));
}

#[test]
fn test_maximum() {
    let attr = Attribute::Maximum(MaximumAttribute { nan_mode: NanPropagationMode::Ignore });
    let op_out = roundtrip_op(Op::Maximum, attr, &["a", "b"], &["out"]);
    assert_eq!(op_out.op, Op::Maximum);
    match op_out.attribute {
        Attribute::Maximum(a) => assert_eq!(a.nan_mode, NanPropagationMode::Ignore),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_minimum() {
    let attr = Attribute::Minimum(MinimumAttribute { nan_mode: NanPropagationMode::Propagate });
    let op_out = roundtrip_op(Op::Minimum, attr, &["a", "b"], &["out"]);
    assert_eq!(op_out.op, Op::Minimum);
    match op_out.attribute {
        Attribute::Minimum(a) => assert_eq!(a.nan_mode, NanPropagationMode::Propagate),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_negate() {
    let attr = Attribute::Negate(NegateAttribute {});
    let op_out = roundtrip_op(Op::Negate, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::Negate);
    assert!(matches!(op_out.attribute, Attribute::Negate(_)));
}

#[test]
fn test_reduce_all() {
    let attr = Attribute::ReduceAll(ReduceAllAttribute { axis: 1 });
    let op_out = roundtrip_op(Op::ReduceAll, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::ReduceAll);
    match op_out.attribute {
        Attribute::ReduceAll(a) => assert_eq!(a.axis, 1),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_reduce_any() {
    let attr = Attribute::ReduceAny(ReduceAnyAttribute { axis: 1 });
    let op_out = roundtrip_op(Op::ReduceAny, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::ReduceAny);
    match op_out.attribute {
        Attribute::ReduceAny(a) => assert_eq!(a.axis, 1),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_reduce_max() {
    let attr = Attribute::ReduceMax(ReduceMaxAttribute { axis: 1, nan_mode: NanPropagationMode::Ignore });
    let op_out = roundtrip_op(Op::ReduceMax, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::ReduceMax);
    match op_out.attribute {
        Attribute::ReduceMax(a) => {
            assert_eq!(a.axis, 1);
            assert_eq!(a.nan_mode, NanPropagationMode::Ignore);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_reduce_min() {
    let attr = Attribute::ReduceMin(ReduceMinAttribute { axis: 1, nan_mode: NanPropagationMode::Propagate });
    let op_out = roundtrip_op(Op::ReduceMin, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::ReduceMin);
    match op_out.attribute {
        Attribute::ReduceMin(a) => {
            assert_eq!(a.axis, 1);
            assert_eq!(a.nan_mode, NanPropagationMode::Propagate);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_reduce_product() {
    let attr = Attribute::ReduceProduct(ReduceProductAttribute { axis: 1 });
    let op_out = roundtrip_op(Op::ReduceProduct, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::ReduceProduct);
    match op_out.attribute {
        Attribute::ReduceProduct(a) => assert_eq!(a.axis, 1),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_reduce_sum() {
    let attr = Attribute::ReduceSum(ReduceSumAttribute { axis: 1 });
    let op_out = roundtrip_op(Op::ReduceSum, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::ReduceSum);
    match op_out.attribute {
        Attribute::ReduceSum(a) => assert_eq!(a.axis, 1),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_concat() {
    let attr = Attribute::Concat(ConcatAttribute { axis: 0 });
    let op_out = roundtrip_op(Op::Concat, attr, &["a", "b"], &["out"]);
    assert_eq!(op_out.op, Op::Concat);
    match op_out.attribute {
        Attribute::Concat(a) => assert_eq!(a.axis, 0),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_pad() {
    let attr = Attribute::Pad(PadAttribute {});
    let op_out = roundtrip_op(Op::Pad, attr, &["in", "padding"], &["out"]);
    assert_eq!(op_out.op, Op::Pad);
    assert!(matches!(op_out.attribute, Attribute::Pad(_)));
}

#[test]
fn test_reverse() {
    let attr = Attribute::Reverse(ReverseAttribute { axis: 2 });
    let op_out = roundtrip_op(Op::Reverse, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::Reverse);
    match op_out.attribute {
        Attribute::Reverse(a) => assert_eq!(a.axis, 2),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_transpose() {
    let attr = Attribute::Transpose(TransposeAttribute { perms: vec![0, 3, 1, 2] });
    let op_out = roundtrip_op(Op::Transpose, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::Transpose);
    match op_out.attribute {
        Attribute::Transpose(a) => assert_eq!(a.perms, vec![0, 3, 1, 2]),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_resize() {
    let attr = Attribute::Resize(ResizeAttribute { mode: ResizeMode::Bilinear });
    let op_out = roundtrip_op(Op::Resize, attr, &["in", "scale", "offset", "border"], &["out"]);
    assert_eq!(op_out.op, Op::Resize);
    match op_out.attribute {
        Attribute::Resize(a) => assert_eq!(a.mode, ResizeMode::Bilinear),
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_rescale() {
    let attr = Attribute::Rescale(RescaleAttribute {
        scale32: true,
        rounding_mode: RoundingMode::DoubleRound,
        per_channel: true,
        input_unsigned: true,
        output_unsigned: false,
    });
    let op_out = roundtrip_op(Op::Rescale, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::Rescale);
    match op_out.attribute {
        Attribute::Rescale(a) => {
            assert!(a.scale32);
            assert_eq!(a.rounding_mode, RoundingMode::DoubleRound);
            assert!(a.per_channel);
            assert!(a.input_unsigned);
            assert!(!a.output_unsigned);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_custom() {
    let attr = Attribute::Custom(CustomAttribute {
        operator_name: "my_op".into(),
        domain_name: "custom".into(),
        implementation_attrs: vec![1, 2, 3],
    });
    let op_out = roundtrip_op(Op::Custom, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::Custom);
    match op_out.attribute {
        Attribute::Custom(a) => {
            assert_eq!(a.operator_name, "my_op");
            assert_eq!(a.domain_name, "custom");
            assert_eq!(a.implementation_attrs, vec![1, 2, 3]);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_cond_if() {
    let attr = Attribute::CondIf(CondIfAttribute {
        then_graph: "then_block".into(),
        else_graph: "else_block".into(),
    });
    let op_out = roundtrip_op(Op::CondIf, attr, &["cond"], &["out"]);
    assert_eq!(op_out.op, Op::CondIf);
    match op_out.attribute {
        Attribute::CondIf(a) => {
            assert_eq!(a.then_graph, "then_block");
            assert_eq!(a.else_graph, "else_block");
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_while_loop() {
    let attr = Attribute::WhileLoop(WhileLoopAttribute {
        cond_graph: "cond_block".into(),
        body_graph: "body_block".into(),
    });
    let op_out = roundtrip_op(Op::WhileLoop, attr, &["init"], &["result"]);
    assert_eq!(op_out.op, Op::WhileLoop);
    match op_out.attribute {
        Attribute::WhileLoop(a) => {
            assert_eq!(a.cond_graph, "cond_block");
            assert_eq!(a.body_graph, "body_block");
        }
        _ => panic!("wrong attribute type"),
    }
}

// ===== Empty attribute roundtrips =====

macro_rules! test_empty_attr {
    ($test_name:ident, $op:ident, $attr_variant:ident, $attr_type:ident) => {
        #[test]
        fn $test_name() {
            let attr = Attribute::$attr_variant($attr_type::default());
            let op_out = roundtrip_op(Op::$op, attr, &["in"], &["out"]);
            assert_eq!(op_out.op, Op::$op);
            assert!(matches!(op_out.attribute, Attribute::$attr_variant(_)));
            assert_eq!(op_out.inputs, vec!["in"]);
            assert_eq!(op_out.outputs, vec!["out"]);
        }
    };
}

test_empty_attr!(test_matmul, MatMul, MatMul, MatMulAttribute);
test_empty_attr!(test_erf, Erf, Erf, ErfAttribute);
test_empty_attr!(test_sigmoid, Sigmoid, Sigmoid, SigmoidAttribute);
test_empty_attr!(test_tanh, Tanh, Tanh, TanhAttribute);
test_empty_attr!(test_add, Add, Add, AddAttribute);
test_empty_attr!(test_bitwise_and, BitwiseAnd, BitwiseAnd, BitwiseAndAttribute);
test_empty_attr!(test_bitwise_or, BitwiseOr, BitwiseOr, BitwiseOrAttribute);
test_empty_attr!(test_bitwise_xor, BitwiseXor, BitwiseXor, BitwiseXorAttribute);
test_empty_attr!(test_intdiv, IntDiv, IntDiv, IntDivAttribute);
test_empty_attr!(test_logical_and, LogicalAnd, LogicalAnd, LogicalAndAttribute);
test_empty_attr!(test_logical_left_shift, LogicalLeftShift, LogicalLeftShift, LogicalLeftShiftAttribute);
test_empty_attr!(test_logical_right_shift, LogicalRightShift, LogicalRightShift, LogicalRightShiftAttribute);
test_empty_attr!(test_logical_or, LogicalOr, LogicalOr, LogicalOrAttribute);
test_empty_attr!(test_logical_xor, LogicalXor, LogicalXor, LogicalXorAttribute);
test_empty_attr!(test_mul, Mul, Mul, MulAttribute);
test_empty_attr!(test_pow, Pow, Pow, PowAttribute);
test_empty_attr!(test_sub, Sub, Sub, SubAttribute);
test_empty_attr!(test_table, Table, Table, TableAttribute);
test_empty_attr!(test_abs, Abs, Abs, AbsAttribute);
test_empty_attr!(test_bitwise_not, BitwiseNot, BitwiseNot, BitwiseNotAttribute);
test_empty_attr!(test_ceil, Ceil, Ceil, CeilAttribute);
test_empty_attr!(test_clz, Clz, Clz, ClzAttribute);
test_empty_attr!(test_cos, Cos, Cos, CosAttribute);
test_empty_attr!(test_exp, Exp, Exp, ExpAttribute);
test_empty_attr!(test_floor, Floor, Floor, FloorAttribute);
test_empty_attr!(test_log, Log, Log, LogAttribute);
test_empty_attr!(test_logical_not, LogicalNot, LogicalNot, LogicalNotAttribute);
test_empty_attr!(test_reciprocal, Reciprocal, Reciprocal, ReciprocalAttribute);
test_empty_attr!(test_rsqrt, Rsqrt, Rsqrt, RsqrtAttribute);
test_empty_attr!(test_sin, Sin, Sin, SinAttribute);
test_empty_attr!(test_select, Select, Select, SelectAttribute);
test_empty_attr!(test_equal, Equal, Equal, EqualAttribute);
test_empty_attr!(test_greater, Greater, Greater, GreaterAttribute);
test_empty_attr!(test_greater_equal, GreaterEqual, GreaterEqual, GreaterEqualAttribute);
test_empty_attr!(test_reshape, Reshape, Reshape, ReshapeAttribute);
test_empty_attr!(test_slice, Slice, Slice, SliceAttribute);
test_empty_attr!(test_tile, Tile, Tile, TileAttribute);
test_empty_attr!(test_gather, Gather, Gather, GatherAttribute);
test_empty_attr!(test_scatter, Scatter, Scatter, ScatterAttribute);
test_empty_attr!(test_cast, Cast, Cast, CastAttribute);
test_empty_attr!(test_const_op, Const, Const, ConstAttribute);
test_empty_attr!(test_identity, Identity, Identity, IdentityAttribute);
test_empty_attr!(test_variable, Variable, Variable, VariableAttribute);
test_empty_attr!(test_variable_write, VariableWrite, VariableWrite, VariableWriteAttribute);
test_empty_attr!(test_variable_read, VariableRead, VariableRead, VariableReadAttribute);
test_empty_attr!(test_const_shape, ConstShape, ConstShape, ConstShapeAttribute);

// ===== Attribute::None and default-field roundtrips =====

#[test]
fn test_operator_with_attribute_none() {
    let graph = TosaGraph {
        version: Version { major: 0, minor: 80, patch: 1, draft: false },
        regions: vec![TosaRegion {
            name: "main".into(),
            blocks: vec![TosaBasicBlock {
                name: "block0".into(),
                operators: vec![TosaOperator {
                    op: Op::Unknown,
                    attribute: Attribute::None,
                    inputs: vec!["in".into()],
                    outputs: vec!["out".into()],
                    ..Default::default()
                }],
                tensors: vec![],
                inputs: vec![],
                outputs: vec![],
                ..Default::default()
            }],
        }],
        ..Default::default()
    };
    let bytes = tosa_rs::serialize::serialize(&graph);
    let result = tosa_rs::serialize::deserialize(&bytes).unwrap();
    let op = &result.regions[0].blocks[0].operators[0];
    assert_eq!(op.op, Op::Unknown);
    assert_eq!(op.attribute, Attribute::None);
    assert_eq!(op.inputs, vec!["in"]);
    assert_eq!(op.outputs, vec!["out"]);
}

#[test]
fn test_conv2d_default_fields() {
    // All fields at their default values (empty vecs, false, Unknown dtype)
    let attr = Attribute::Conv2d(Conv2dAttribute::default());
    let op_out = roundtrip_op(Op::Conv2d, attr, &["input", "weight", "bias"], &["output"]);
    match op_out.attribute {
        Attribute::Conv2d(a) => {
            assert!(a.pad.is_empty());
            assert!(a.stride.is_empty());
            assert!(a.dilation.is_empty());
            assert!(!a.local_bound);
            assert_eq!(a.acc_type, DType::Unknown);
        }
        _ => panic!("wrong attribute type"),
    }
}

#[test]
fn test_rescale_default_fields() {
    let attr = Attribute::Rescale(RescaleAttribute::default());
    let op_out = roundtrip_op(Op::Rescale, attr, &["in"], &["out"]);
    match op_out.attribute {
        Attribute::Rescale(a) => {
            assert!(!a.scale32);
            assert_eq!(a.rounding_mode, RoundingMode::Unknown);
            assert!(!a.per_channel);
            assert!(!a.input_unsigned);
            assert!(!a.output_unsigned);
        }
        _ => panic!("wrong attribute type"),
    }
}

// ===== New TOSA 1.0 ops =====

#[test]
fn test_matmul_t_block_scaled() {
    let attr = Attribute::MatMulTBlockScaled(MatMulTBlockScaledAttribute { block_size: BlockSize::BlockSize32 });
    let op_out = roundtrip_op(Op::MatmulTBlockScaled, attr, &["a", "b"], &["c"]);
    assert_eq!(op_out.op, Op::MatmulTBlockScaled);
    match op_out.attribute {
        Attribute::MatMulTBlockScaled(a) => assert_eq!(a.block_size, BlockSize::BlockSize32),
        _ => panic!("wrong attribute"),
    }
}

#[test]
fn test_cast_from_block_scaled() {
    let attr = Attribute::CastFromBlockScaled(CastFromBlockScaledAttribute { block_size: BlockSize::BlockSize32 });
    let op_out = roundtrip_op(Op::CastFromBlockScaled, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::CastFromBlockScaled);
    match op_out.attribute {
        Attribute::CastFromBlockScaled(a) => assert_eq!(a.block_size, BlockSize::BlockSize32),
        _ => panic!("wrong attribute"),
    }
}

#[test]
fn test_cast_to_block_scaled() {
    let attr = Attribute::CastToBlockScaled(CastToBlockScaledAttribute { block_size: BlockSize::BlockSize32 });
    let op_out = roundtrip_op(Op::CastToBlockScaled, attr, &["in"], &["out"]);
    assert_eq!(op_out.op, Op::CastToBlockScaled);
    match op_out.attribute {
        Attribute::CastToBlockScaled(a) => assert_eq!(a.block_size, BlockSize::BlockSize32),
        _ => panic!("wrong attribute"),
    }
}
