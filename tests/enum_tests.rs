use tosa_rs::generated;
use tosa_rs::types::*;

// ---------------------------------------------------------------------------
// DType: discriminant values
// ---------------------------------------------------------------------------

#[test]
fn dtype_discriminant_values() {
    assert_eq!(DType::Unknown as u32, 0);
    assert_eq!(DType::Bool as u32, 1);
    assert_eq!(DType::Int4 as u32, 2);
    assert_eq!(DType::Int8 as u32, 3);
    assert_eq!(DType::Int16 as u32, 4);
    assert_eq!(DType::Int32 as u32, 5);
    assert_eq!(DType::Int48 as u32, 6);
    assert_eq!(DType::Fp32 as u32, 7);
    assert_eq!(DType::Fp16 as u32, 8);
    assert_eq!(DType::Bf16 as u32, 9);
    assert_eq!(DType::Shape as u32, 10);
    assert_eq!(DType::Fp8E4M3 as u32, 11);
    assert_eq!(DType::Fp8E5M2 as u32, 12);
    assert_eq!(DType::Fp6E2M3 as u32, 13);
    assert_eq!(DType::Fp6E3M2 as u32, 14);
    assert_eq!(DType::Fp4E2M1 as u32, 15);
    assert_eq!(DType::Fp8Ue8M0 as u32, 16);
    assert_eq!(DType::Int64 as u32, 17);
    assert_eq!(DType::Mxint8 as u32, 18);
}

#[test]
fn dtype_variant_count() {
    // Ensure all 19 variants are accounted for by matching exhaustively.
    let all: [DType; 19] = [
        DType::Unknown,
        DType::Bool,
        DType::Int4,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int48,
        DType::Fp32,
        DType::Fp16,
        DType::Bf16,
        DType::Shape,
        DType::Fp8E4M3,
        DType::Fp8E5M2,
        DType::Fp6E2M3,
        DType::Fp6E3M2,
        DType::Fp4E2M1,
        DType::Fp8Ue8M0,
        DType::Int64,
        DType::Mxint8,
    ];
    for (i, &dt) in all.iter().enumerate() {
        assert_eq!(dt as u32, i as u32);
    }
}

#[test]
fn dtype_from_generated_roundtrip() {
    let all: [DType; 19] = [
        DType::Unknown,
        DType::Bool,
        DType::Int4,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int48,
        DType::Fp32,
        DType::Fp16,
        DType::Bf16,
        DType::Shape,
        DType::Fp8E4M3,
        DType::Fp8E5M2,
        DType::Fp6E2M3,
        DType::Fp6E3M2,
        DType::Fp4E2M1,
        DType::Fp8Ue8M0,
        DType::Int64,
        DType::Mxint8,
    ];
    for &dt in &all {
        let r#gen: generated::DType = dt.into();
        let back: DType = r#gen.into();
        assert_eq!(back, dt, "DType roundtrip failed for {:?}", dt);
    }
}

#[test]
fn dtype_from_generated_unknown_fallback() {
    // Out-of-range generated values should map to Unknown.
    let r#gen = generated::DType(999);
    let dt: DType = r#gen.into();
    assert_eq!(dt, DType::Unknown);
}

// ---------------------------------------------------------------------------
// Op: discriminant values
// ---------------------------------------------------------------------------

#[test]
fn op_discriminant_values() {
    let expected: [(Op, u32); 79] = [
        (Op::Unknown, 0),
        (Op::ArgMax, 1),
        (Op::AvgPool2d, 2),
        (Op::Conv2d, 3),
        (Op::Conv3d, 4),
        (Op::DepthwiseConv2d, 5),
        (Op::Fft2d, 6),
        (Op::MatMul, 7),
        (Op::MaxPool2d, 8),
        (Op::Rfft2d, 9),
        (Op::TransposeConv2d, 10),
        (Op::Clamp, 11),
        (Op::Erf, 12),
        (Op::Sigmoid, 13),
        (Op::Tanh, 14),
        (Op::Add, 15),
        (Op::ArithmeticRightShift, 16),
        (Op::BitwiseAnd, 17),
        (Op::BitwiseOr, 18),
        (Op::BitwiseXor, 19),
        (Op::IntDiv, 20),
        (Op::LogicalAnd, 21),
        (Op::LogicalLeftShift, 22),
        (Op::LogicalRightShift, 23),
        (Op::LogicalOr, 24),
        (Op::LogicalXor, 25),
        (Op::Maximum, 26),
        (Op::Minimum, 27),
        (Op::Mul, 28),
        (Op::Pow, 29),
        (Op::Sub, 30),
        (Op::Table, 31),
        (Op::Abs, 32),
        (Op::BitwiseNot, 33),
        (Op::Ceil, 34),
        (Op::Clz, 35),
        (Op::Cos, 36),
        (Op::Exp, 37),
        (Op::Floor, 38),
        (Op::Log, 39),
        (Op::LogicalNot, 40),
        (Op::Negate, 41),
        (Op::Reciprocal, 42),
        (Op::Rsqrt, 43),
        (Op::Sin, 44),
        (Op::Select, 45),
        (Op::Equal, 46),
        (Op::Greater, 47),
        (Op::GreaterEqual, 48),
        (Op::ReduceAll, 49),
        (Op::ReduceAny, 50),
        (Op::ReduceMax, 51),
        (Op::ReduceMin, 52),
        (Op::ReduceProduct, 53),
        (Op::ReduceSum, 54),
        (Op::Concat, 55),
        (Op::Pad, 56),
        (Op::Reshape, 57),
        (Op::Reverse, 58),
        (Op::Slice, 59),
        (Op::Tile, 60),
        (Op::Transpose, 61),
        (Op::Gather, 62),
        (Op::Scatter, 63),
        (Op::Resize, 64),
        (Op::Cast, 65),
        (Op::Rescale, 66),
        (Op::Const, 67),
        (Op::Identity, 68),
        (Op::Custom, 69),
        (Op::CondIf, 70),
        (Op::WhileLoop, 71),
        (Op::Variable, 72),
        (Op::VariableWrite, 73),
        (Op::VariableRead, 74),
        (Op::ConstShape, 75),
        (Op::MatmulTBlockScaled, 76),
        (Op::CastFromBlockScaled, 77),
        (Op::CastToBlockScaled, 78),
    ];
    for &(op, disc) in &expected {
        assert_eq!(op as u32, disc, "Op::{:?} should have discriminant {}", op, disc);
    }
}

#[test]
fn op_variant_count() {
    // All 79 variants exist (0..=78).
    let all: [Op; 79] = [
        Op::Unknown,
        Op::ArgMax,
        Op::AvgPool2d,
        Op::Conv2d,
        Op::Conv3d,
        Op::DepthwiseConv2d,
        Op::Fft2d,
        Op::MatMul,
        Op::MaxPool2d,
        Op::Rfft2d,
        Op::TransposeConv2d,
        Op::Clamp,
        Op::Erf,
        Op::Sigmoid,
        Op::Tanh,
        Op::Add,
        Op::ArithmeticRightShift,
        Op::BitwiseAnd,
        Op::BitwiseOr,
        Op::BitwiseXor,
        Op::IntDiv,
        Op::LogicalAnd,
        Op::LogicalLeftShift,
        Op::LogicalRightShift,
        Op::LogicalOr,
        Op::LogicalXor,
        Op::Maximum,
        Op::Minimum,
        Op::Mul,
        Op::Pow,
        Op::Sub,
        Op::Table,
        Op::Abs,
        Op::BitwiseNot,
        Op::Ceil,
        Op::Clz,
        Op::Cos,
        Op::Exp,
        Op::Floor,
        Op::Log,
        Op::LogicalNot,
        Op::Negate,
        Op::Reciprocal,
        Op::Rsqrt,
        Op::Sin,
        Op::Select,
        Op::Equal,
        Op::Greater,
        Op::GreaterEqual,
        Op::ReduceAll,
        Op::ReduceAny,
        Op::ReduceMax,
        Op::ReduceMin,
        Op::ReduceProduct,
        Op::ReduceSum,
        Op::Concat,
        Op::Pad,
        Op::Reshape,
        Op::Reverse,
        Op::Slice,
        Op::Tile,
        Op::Transpose,
        Op::Gather,
        Op::Scatter,
        Op::Resize,
        Op::Cast,
        Op::Rescale,
        Op::Const,
        Op::Identity,
        Op::Custom,
        Op::CondIf,
        Op::WhileLoop,
        Op::Variable,
        Op::VariableWrite,
        Op::VariableRead,
        Op::ConstShape,
        Op::MatmulTBlockScaled,
        Op::CastFromBlockScaled,
        Op::CastToBlockScaled,
    ];
    assert_eq!(all.len(), 79);
}

#[test]
fn op_from_generated_roundtrip() {
    for disc in 0u32..=78 {
        let r#gen = generated::Op(disc);
        let op: Op = r#gen.into();
        let gen2: generated::Op = op.into();
        assert_eq!(
            gen2.0, disc,
            "Op roundtrip failed for discriminant {}",
            disc
        );
    }
}

#[test]
fn op_from_generated_unknown_fallback() {
    let r#gen = generated::Op(9999);
    let op: Op = r#gen.into();
    assert_eq!(op, Op::Unknown);
}

// ---------------------------------------------------------------------------
// ResizeMode: discriminant values and roundtrip
// ---------------------------------------------------------------------------

#[test]
fn resize_mode_discriminant_values() {
    assert_eq!(ResizeMode::Unknown as u32, 0);
    assert_eq!(ResizeMode::Nearest as u32, 1);
    assert_eq!(ResizeMode::Bilinear as u32, 2);
}

#[test]
fn resize_mode_variant_count() {
    let all: [ResizeMode; 3] = [
        ResizeMode::Unknown,
        ResizeMode::Nearest,
        ResizeMode::Bilinear,
    ];
    for (i, &m) in all.iter().enumerate() {
        assert_eq!(m as u32, i as u32);
    }
}

#[test]
fn resize_mode_from_generated_roundtrip() {
    for disc in 0u32..=2 {
        let r#gen = generated::ResizeMode(disc);
        let mode: ResizeMode = r#gen.into();
        let gen2: generated::ResizeMode = mode.into();
        assert_eq!(gen2.0, disc, "ResizeMode roundtrip failed for {}", disc);
    }
}

#[test]
fn resize_mode_from_generated_unknown_fallback() {
    let r#gen = generated::ResizeMode(42);
    let mode: ResizeMode = r#gen.into();
    assert_eq!(mode, ResizeMode::Unknown);
}

// ---------------------------------------------------------------------------
// NanPropagationMode: discriminant values and roundtrip
// ---------------------------------------------------------------------------

#[test]
fn nan_propagation_mode_discriminant_values() {
    assert_eq!(NanPropagationMode::Unknown as u32, 0);
    assert_eq!(NanPropagationMode::Propagate as u32, 1);
    assert_eq!(NanPropagationMode::Ignore as u32, 2);
}

#[test]
fn nan_propagation_mode_variant_count() {
    let all: [NanPropagationMode; 3] = [
        NanPropagationMode::Unknown,
        NanPropagationMode::Propagate,
        NanPropagationMode::Ignore,
    ];
    for (i, &m) in all.iter().enumerate() {
        assert_eq!(m as u32, i as u32);
    }
}

#[test]
fn nan_propagation_mode_from_generated_roundtrip() {
    for disc in 0u32..=2 {
        let r#gen = generated::NanPropagationMode(disc);
        let mode: NanPropagationMode = r#gen.into();
        let gen2: generated::NanPropagationMode = mode.into();
        assert_eq!(
            gen2.0, disc,
            "NanPropagationMode roundtrip failed for {}",
            disc
        );
    }
}

#[test]
fn nan_propagation_mode_from_generated_unknown_fallback() {
    let r#gen = generated::NanPropagationMode(100);
    let mode: NanPropagationMode = r#gen.into();
    assert_eq!(mode, NanPropagationMode::Unknown);
}

// ---------------------------------------------------------------------------
// Default impls: all enums default to Unknown
// ---------------------------------------------------------------------------

#[test]
fn dtype_default_is_unknown() {
    assert_eq!(DType::default(), DType::Unknown);
}

#[test]
fn op_default_is_unknown() {
    assert_eq!(Op::default(), Op::Unknown);
}

#[test]
fn resize_mode_default_is_unknown() {
    assert_eq!(ResizeMode::default(), ResizeMode::Unknown);
}

#[test]
fn nan_propagation_mode_default_is_unknown() {
    assert_eq!(NanPropagationMode::default(), NanPropagationMode::Unknown);
}

// ---------------------------------------------------------------------------
// RoundingMode: discriminant values and roundtrip
// ---------------------------------------------------------------------------

#[test]
fn rounding_mode_discriminant_values() {
    assert_eq!(RoundingMode::Unknown as u32, 0);
    assert_eq!(RoundingMode::SingleRound as u32, 1);
    assert_eq!(RoundingMode::InexactRound as u32, 2);
    assert_eq!(RoundingMode::DoubleRound as u32, 3);
}

#[test]
fn rounding_mode_variant_count() {
    let all: [RoundingMode; 4] = [
        RoundingMode::Unknown,
        RoundingMode::SingleRound,
        RoundingMode::InexactRound,
        RoundingMode::DoubleRound,
    ];
    for (i, &m) in all.iter().enumerate() {
        assert_eq!(m as u32, i as u32);
    }
}

#[test]
fn rounding_mode_from_generated_roundtrip() {
    for disc in 0u32..=3 {
        let r#gen = generated::RoundingMode(disc);
        let mode: RoundingMode = r#gen.into();
        let gen2: generated::RoundingMode = mode.into();
        assert_eq!(gen2.0, disc, "RoundingMode roundtrip failed for {}", disc);
    }
}

#[test]
fn rounding_mode_from_generated_unknown_fallback() {
    let r#gen = generated::RoundingMode(99);
    let mode: RoundingMode = r#gen.into();
    assert_eq!(mode, RoundingMode::Unknown);
}

#[test]
fn rounding_mode_default_is_unknown() {
    assert_eq!(RoundingMode::default(), RoundingMode::Unknown);
}

// ---------------------------------------------------------------------------
// BlockSize: discriminant values and roundtrip
// ---------------------------------------------------------------------------

#[test]
fn block_size_discriminant_values() {
    assert_eq!(BlockSize::Unknown as u32, 0);
    assert_eq!(BlockSize::BlockSize32 as u32, 32);
}

#[test]
fn block_size_variant_count() {
    let all: [BlockSize; 2] = [
        BlockSize::Unknown,
        BlockSize::BlockSize32,
    ];
    assert_eq!(all.len(), 2);
}

#[test]
fn block_size_from_generated_roundtrip() {
    let cases: [(u32, BlockSize); 2] = [
        (0, BlockSize::Unknown),
        (32, BlockSize::BlockSize32),
    ];
    for &(disc, expected) in &cases {
        let r#gen = generated::BlockSize(disc);
        let bs: BlockSize = r#gen.into();
        assert_eq!(bs, expected);
        let gen2: generated::BlockSize = bs.into();
        assert_eq!(gen2.0, disc, "BlockSize roundtrip failed for {}", disc);
    }
}

#[test]
fn block_size_from_generated_unknown_fallback() {
    let r#gen = generated::BlockSize(99);
    let bs: BlockSize = r#gen.into();
    assert_eq!(bs, BlockSize::Unknown);
}

#[test]
fn block_size_default_is_unknown() {
    assert_eq!(BlockSize::default(), BlockSize::Unknown);
}
