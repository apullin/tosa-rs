use serde::{Deserialize, Serialize};

use crate::generated;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum DType {
    Unknown = 0,
    Bool = 1,
    Uint8 = 2,
    Int4 = 3,
    Int8 = 4,
    Int16 = 5,
    Int32 = 6,
    Int48 = 7,
    Fp32 = 8,
    Uint16 = 9,
    Fp16 = 10,
    Bf16 = 11,
    Shape = 12,
    Fp8E4M3 = 13,
    Fp8E5M2 = 14,
}

impl Default for DType {
    fn default() -> Self {
        DType::Unknown
    }
}

impl From<generated::DType> for DType {
    fn from(v: generated::DType) -> Self {
        match v.0 {
            0 => DType::Unknown,
            1 => DType::Bool,
            2 => DType::Uint8,
            3 => DType::Int4,
            4 => DType::Int8,
            5 => DType::Int16,
            6 => DType::Int32,
            7 => DType::Int48,
            8 => DType::Fp32,
            9 => DType::Uint16,
            10 => DType::Fp16,
            11 => DType::Bf16,
            12 => DType::Shape,
            13 => DType::Fp8E4M3,
            14 => DType::Fp8E5M2,
            _ => DType::Unknown,
        }
    }
}

impl From<DType> for generated::DType {
    fn from(v: DType) -> Self {
        generated::DType(v as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ResizeMode {
    Unknown = 0,
    Nearest = 1,
    Bilinear = 2,
}

impl Default for ResizeMode {
    fn default() -> Self {
        ResizeMode::Unknown
    }
}

impl From<generated::ResizeMode> for ResizeMode {
    fn from(v: generated::ResizeMode) -> Self {
        match v.0 {
            0 => ResizeMode::Unknown,
            1 => ResizeMode::Nearest,
            2 => ResizeMode::Bilinear,
            _ => ResizeMode::Unknown,
        }
    }
}

impl From<ResizeMode> for generated::ResizeMode {
    fn from(v: ResizeMode) -> Self {
        generated::ResizeMode(v as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum NanPropagationMode {
    Unknown = 0,
    Propagate = 1,
    Ignore = 2,
}

impl Default for NanPropagationMode {
    fn default() -> Self {
        NanPropagationMode::Unknown
    }
}

impl From<generated::NanPropagationMode> for NanPropagationMode {
    fn from(v: generated::NanPropagationMode) -> Self {
        match v.0 {
            0 => NanPropagationMode::Unknown,
            1 => NanPropagationMode::Propagate,
            2 => NanPropagationMode::Ignore,
            _ => NanPropagationMode::Unknown,
        }
    }
}

impl From<NanPropagationMode> for generated::NanPropagationMode {
    fn from(v: NanPropagationMode) -> Self {
        generated::NanPropagationMode(v as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Op {
    Unknown = 0,
    ArgMax = 1,
    AvgPool2d = 2,
    Conv2d = 3,
    Conv3d = 4,
    DepthwiseConv2d = 5,
    Fft2d = 6,
    MatMul = 7,
    MaxPool2d = 8,
    Rfft2d = 9,
    TransposeConv2d = 10,
    Clamp = 11,
    Erf = 12,
    Sigmoid = 13,
    Tanh = 14,
    Add = 15,
    ArithmeticRightShift = 16,
    BitwiseAnd = 17,
    BitwiseOr = 18,
    BitwiseXor = 19,
    IntDiv = 20,
    LogicalAnd = 21,
    LogicalLeftShift = 22,
    LogicalRightShift = 23,
    LogicalOr = 24,
    LogicalXor = 25,
    Maximum = 26,
    Minimum = 27,
    Mul = 28,
    Pow = 29,
    Sub = 30,
    Table = 31,
    Abs = 32,
    BitwiseNot = 33,
    Ceil = 34,
    Clz = 35,
    Cos = 36,
    Exp = 37,
    Floor = 38,
    Log = 39,
    LogicalNot = 40,
    Negate = 41,
    Reciprocal = 42,
    Rsqrt = 43,
    Sin = 44,
    Select = 45,
    Equal = 46,
    Greater = 47,
    GreaterEqual = 48,
    ReduceAll = 49,
    ReduceAny = 50,
    ReduceMax = 51,
    ReduceMin = 52,
    ReduceProduct = 53,
    ReduceSum = 54,
    Concat = 55,
    Pad = 56,
    Reshape = 57,
    Reverse = 58,
    Slice = 59,
    Tile = 60,
    Transpose = 61,
    Gather = 62,
    Scatter = 63,
    Resize = 64,
    Cast = 65,
    Rescale = 66,
    Const = 67,
    Identity = 68,
    Custom = 69,
    CondIf = 70,
    WhileLoop = 71,
    Yield = 72,
    Variable = 73,
    VariableWrite = 74,
    VariableRead = 75,
    ConstShape = 76,
}

impl Default for Op {
    fn default() -> Self {
        Op::Unknown
    }
}

impl From<generated::Op> for Op {
    fn from(v: generated::Op) -> Self {
        match v.0 {
            0 => Op::Unknown,
            1 => Op::ArgMax,
            2 => Op::AvgPool2d,
            3 => Op::Conv2d,
            4 => Op::Conv3d,
            5 => Op::DepthwiseConv2d,
            6 => Op::Fft2d,
            7 => Op::MatMul,
            8 => Op::MaxPool2d,
            9 => Op::Rfft2d,
            10 => Op::TransposeConv2d,
            11 => Op::Clamp,
            12 => Op::Erf,
            13 => Op::Sigmoid,
            14 => Op::Tanh,
            15 => Op::Add,
            16 => Op::ArithmeticRightShift,
            17 => Op::BitwiseAnd,
            18 => Op::BitwiseOr,
            19 => Op::BitwiseXor,
            20 => Op::IntDiv,
            21 => Op::LogicalAnd,
            22 => Op::LogicalLeftShift,
            23 => Op::LogicalRightShift,
            24 => Op::LogicalOr,
            25 => Op::LogicalXor,
            26 => Op::Maximum,
            27 => Op::Minimum,
            28 => Op::Mul,
            29 => Op::Pow,
            30 => Op::Sub,
            31 => Op::Table,
            32 => Op::Abs,
            33 => Op::BitwiseNot,
            34 => Op::Ceil,
            35 => Op::Clz,
            36 => Op::Cos,
            37 => Op::Exp,
            38 => Op::Floor,
            39 => Op::Log,
            40 => Op::LogicalNot,
            41 => Op::Negate,
            42 => Op::Reciprocal,
            43 => Op::Rsqrt,
            44 => Op::Sin,
            45 => Op::Select,
            46 => Op::Equal,
            47 => Op::Greater,
            48 => Op::GreaterEqual,
            49 => Op::ReduceAll,
            50 => Op::ReduceAny,
            51 => Op::ReduceMax,
            52 => Op::ReduceMin,
            53 => Op::ReduceProduct,
            54 => Op::ReduceSum,
            55 => Op::Concat,
            56 => Op::Pad,
            57 => Op::Reshape,
            58 => Op::Reverse,
            59 => Op::Slice,
            60 => Op::Tile,
            61 => Op::Transpose,
            62 => Op::Gather,
            63 => Op::Scatter,
            64 => Op::Resize,
            65 => Op::Cast,
            66 => Op::Rescale,
            67 => Op::Const,
            68 => Op::Identity,
            69 => Op::Custom,
            70 => Op::CondIf,
            71 => Op::WhileLoop,
            72 => Op::Yield,
            73 => Op::Variable,
            74 => Op::VariableWrite,
            75 => Op::VariableRead,
            76 => Op::ConstShape,
            _ => Op::Unknown,
        }
    }
}

impl From<Op> for generated::Op {
    fn from(v: Op) -> Self {
        generated::Op(v as u32)
    }
}

// ---------------------------------------------------------------------------
// Attribute structs (all 78)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ArgMaxAttribute {
    pub axis: i32,
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AvgPool2dAttribute {
    pub kernel: Vec<i32>,
    pub stride: Vec<i32>,
    pub pad: Vec<i32>,
    pub input_zp: i32,
    pub output_zp: i32,
    pub acc_type: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Conv2dAttribute {
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub dilation: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Conv3dAttribute {
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub dilation: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DepthwiseConv2dAttribute {
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub dilation: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FFT2dAttribute {
    pub inverse: bool,
    pub local_bound: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MatMulAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MaxPool2dAttribute {
    pub kernel: Vec<i32>,
    pub stride: Vec<i32>,
    pub pad: Vec<i32>,
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RFFT2dAttribute {
    pub local_bound: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TransposeConv2dAttribute {
    pub out_pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ClampAttribute {
    pub min_val: Vec<u8>,
    pub max_val: Vec<u8>,
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ErfAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SigmoidAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TanhAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AddAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ArithmeticRightShiftAttribute {
    pub round: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BitwiseAndAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BitwiseOrAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BitwiseXorAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct IntDivAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogicalAndAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogicalLeftShiftAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogicalRightShiftAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogicalOrAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogicalXorAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MaximumAttribute {
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MinimumAttribute {
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MulAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PowAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SubAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TableAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AbsAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BitwiseNotAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CeilAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ClzAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CosAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExpAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FloorAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LogicalNotAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NegateAttribute {
    pub input1_zp: i32,
    pub output_zp: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReciprocalAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RsqrtAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SinAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SelectAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EqualAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GreaterAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GreaterEqualAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReduceAllAttribute {
    pub axis: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReduceAnyAttribute {
    pub axis: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReduceMaxAttribute {
    pub axis: i32,
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReduceMinAttribute {
    pub axis: i32,
    pub nan_mode: NanPropagationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReduceProductAttribute {
    pub axis: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReduceSumAttribute {
    pub axis: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ConcatAttribute {
    pub axis: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PadAttribute {
    pub pad_const: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReshapeAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ReverseAttribute {
    pub axis: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SliceAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TileAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TransposeAttribute {
    pub perms: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GatherAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ScatterAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResizeAttribute {
    pub mode: ResizeMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CastAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RescaleAttribute {
    pub input_zp: i32,
    pub output_zp: i32,
    pub scale32: bool,
    pub double_round: bool,
    pub per_channel: bool,
    pub input_unsigned: bool,
    pub output_unsigned: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ConstAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct IdentityAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CustomAttribute {
    pub operator_name: String,
    pub domain_name: String,
    pub implementation_attrs: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CondIfAttribute {
    pub then_graph: String,
    pub else_graph: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct WhileLoopAttribute {
    pub cond_graph: String,
    pub body_graph: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct YieldAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VariableAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VariableWriteAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VariableReadAttribute {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ConstShapeAttribute {}

// ---------------------------------------------------------------------------
// Attribute union enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    ArgMax(ArgMaxAttribute),
    AvgPool2d(AvgPool2dAttribute),
    Conv2d(Conv2dAttribute),
    Conv3d(Conv3dAttribute),
    DepthwiseConv2d(DepthwiseConv2dAttribute),
    FFT2d(FFT2dAttribute),
    MatMul(MatMulAttribute),
    MaxPool2d(MaxPool2dAttribute),
    RFFT2d(RFFT2dAttribute),
    TransposeConv2d(TransposeConv2dAttribute),
    Clamp(ClampAttribute),
    Erf(ErfAttribute),
    Sigmoid(SigmoidAttribute),
    Tanh(TanhAttribute),
    Add(AddAttribute),
    ArithmeticRightShift(ArithmeticRightShiftAttribute),
    BitwiseAnd(BitwiseAndAttribute),
    BitwiseOr(BitwiseOrAttribute),
    BitwiseXor(BitwiseXorAttribute),
    IntDiv(IntDivAttribute),
    LogicalAnd(LogicalAndAttribute),
    LogicalLeftShift(LogicalLeftShiftAttribute),
    LogicalRightShift(LogicalRightShiftAttribute),
    LogicalOr(LogicalOrAttribute),
    LogicalXor(LogicalXorAttribute),
    Maximum(MaximumAttribute),
    Minimum(MinimumAttribute),
    Mul(MulAttribute),
    Pow(PowAttribute),
    Sub(SubAttribute),
    Table(TableAttribute),
    Abs(AbsAttribute),
    BitwiseNot(BitwiseNotAttribute),
    Ceil(CeilAttribute),
    Clz(ClzAttribute),
    Cos(CosAttribute),
    Exp(ExpAttribute),
    Floor(FloorAttribute),
    Log(LogAttribute),
    LogicalNot(LogicalNotAttribute),
    Negate(NegateAttribute),
    Reciprocal(ReciprocalAttribute),
    Rsqrt(RsqrtAttribute),
    Sin(SinAttribute),
    Select(SelectAttribute),
    Equal(EqualAttribute),
    Greater(GreaterAttribute),
    GreaterEqual(GreaterEqualAttribute),
    ReduceAll(ReduceAllAttribute),
    ReduceAny(ReduceAnyAttribute),
    ReduceMax(ReduceMaxAttribute),
    ReduceMin(ReduceMinAttribute),
    ReduceProduct(ReduceProductAttribute),
    ReduceSum(ReduceSumAttribute),
    Concat(ConcatAttribute),
    Pad(PadAttribute),
    Reshape(ReshapeAttribute),
    Reverse(ReverseAttribute),
    Slice(SliceAttribute),
    Tile(TileAttribute),
    Transpose(TransposeAttribute),
    Gather(GatherAttribute),
    Scatter(ScatterAttribute),
    Resize(ResizeAttribute),
    Cast(CastAttribute),
    Rescale(RescaleAttribute),
    Const(ConstAttribute),
    Identity(IdentityAttribute),
    Custom(CustomAttribute),
    CondIf(CondIfAttribute),
    WhileLoop(WhileLoopAttribute),
    Yield(YieldAttribute),
    Variable(VariableAttribute),
    VariableWrite(VariableWriteAttribute),
    VariableRead(VariableReadAttribute),
    ConstShape(ConstShapeAttribute),
    None,
}

impl Default for Attribute {
    fn default() -> Self {
        Attribute::None
    }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Version {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
    pub draft: bool,
}

impl Default for Version {
    fn default() -> Self {
        Version {
            major: -1,
            minor: -1,
            patch: -1,
            draft: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TosaTensor {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: DType,
    pub data: Option<Vec<u8>>,
    pub variable: bool,
    pub is_unranked: bool,
    pub variable_name: Option<String>,
}

impl Default for TosaTensor {
    fn default() -> Self {
        TosaTensor {
            name: String::new(),
            shape: Vec::new(),
            dtype: DType::Unknown,
            data: None,
            variable: false,
            is_unranked: false,
            variable_name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TosaOperator {
    pub op: Op,
    pub attribute: Attribute,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl Default for TosaOperator {
    fn default() -> Self {
        TosaOperator {
            op: Op::Unknown,
            attribute: Attribute::None,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TosaBasicBlock {
    pub name: String,
    pub operators: Vec<TosaOperator>,
    pub tensors: Vec<TosaTensor>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl Default for TosaBasicBlock {
    fn default() -> Self {
        TosaBasicBlock {
            name: String::new(),
            operators: Vec::new(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TosaRegion {
    pub name: String,
    pub blocks: Vec<TosaBasicBlock>,
}

impl Default for TosaRegion {
    fn default() -> Self {
        TosaRegion {
            name: String::new(),
            blocks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TosaGraph {
    pub version: Version,
    pub regions: Vec<TosaRegion>,
}

impl Default for TosaGraph {
    fn default() -> Self {
        TosaGraph {
            version: Version::default(),
            regions: Vec::new(),
        }
    }
}
