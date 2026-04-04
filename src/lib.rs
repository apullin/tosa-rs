// Include the flatc-generated code
#[allow(
    unused_imports,
    dead_code,
    non_snake_case,
    clippy::all,
    mismatched_lifetime_syntaxes
)]
mod tosa_generated {
    include!(concat!(env!("OUT_DIR"), "/tosa_generated.rs"));
}

pub mod builder;
pub mod error;
pub mod serialize;
pub mod types;

// Re-export generated types
pub use tosa_generated::tosa as generated;

// Re-export public types
pub use builder::{BlockBuilder, RegionBuilder, TosaGraphBuilder};
pub use serialize::{deserialize, load_file, save_file, serialize};
pub use error::TosaError;
pub use types::{
    // Enums
    BlockSize, DType, NanPropagationMode, Op, ResizeMode, RoundingMode,
    // Attribute union
    Attribute,
    // Attribute structs
    AbsAttribute, AddAttribute, ArgMaxAttribute, ArithmeticRightShiftAttribute,
    AvgPool2dAttribute, BitwiseAndAttribute, BitwiseNotAttribute, BitwiseOrAttribute,
    BitwiseXorAttribute, CastAttribute, CastFromBlockScaledAttribute, CastToBlockScaledAttribute,
    CeilAttribute, ClampAttribute, ClzAttribute,
    ConcatAttribute, CondIfAttribute, ConstAttribute, ConstShapeAttribute, Conv2dAttribute,
    Conv3dAttribute, CosAttribute, CustomAttribute, DepthwiseConv2dAttribute, EqualAttribute,
    ErfAttribute, ExpAttribute, FFT2dAttribute, FloorAttribute, GatherAttribute,
    GreaterAttribute, GreaterEqualAttribute, IdentityAttribute, IntDivAttribute, LogAttribute,
    LogicalAndAttribute, LogicalLeftShiftAttribute, LogicalNotAttribute, LogicalOrAttribute,
    LogicalRightShiftAttribute, LogicalXorAttribute, MatMulAttribute, MatMulTBlockScaledAttribute,
    MaxPool2dAttribute, MaximumAttribute, MinimumAttribute, MulAttribute, NegateAttribute,
    PadAttribute, PowAttribute, RFFT2dAttribute, ReciprocalAttribute, ReduceAllAttribute,
    ReduceAnyAttribute, ReduceMaxAttribute, ReduceMinAttribute, ReduceProductAttribute,
    ReduceSumAttribute, RescaleAttribute, ReshapeAttribute, ResizeAttribute, ReverseAttribute,
    RsqrtAttribute, ScatterAttribute, SelectAttribute, SigmoidAttribute, SinAttribute,
    SliceAttribute, SubAttribute, TableAttribute, TanhAttribute, TileAttribute,
    TransposeAttribute, TransposeConv2dAttribute, VariableAttribute, VariableReadAttribute,
    VariableWriteAttribute, WhileLoopAttribute,
    // Core types
    OpLocation, SoftwareVersion, TosaBasicBlock, TosaGraph, TosaOperator, TosaRegion,
    TosaShape, TosaTensor, Version,
};
