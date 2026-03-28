use std::path::Path;

use flatbuffers::FlatBufferBuilder;

use crate::generated;
use crate::error::TosaError;
use crate::types;

// ---------------------------------------------------------------------------
// Helper: convert a FlatBuffers vector of strings to Vec<String>
// ---------------------------------------------------------------------------

fn fb_strings_to_vec(
    v: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<&str>>>,
) -> Vec<String> {
    match v {
        Some(vec) => (0..vec.len()).map(|i| vec.get(i).to_string()).collect(),
        None => Vec::new(),
    }
}

fn fb_vec_i32(v: Option<flatbuffers::Vector<'_, i32>>) -> Vec<i32> {
    match v {
        Some(vec) => vec.iter().collect(),
        None => Vec::new(),
    }
}

fn fb_vec_u8(v: Option<flatbuffers::Vector<'_, u8>>) -> Vec<u8> {
    match v {
        Some(vec) => vec.iter().collect(),
        None => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Deserialization: FlatBuffers bytes -> owned types
// ---------------------------------------------------------------------------

/// Deserialize a TOSA FlatBuffer from raw bytes into an owned `TosaGraph`.
pub fn deserialize(bytes: &[u8]) -> Result<types::TosaGraph, TosaError> {
    let opts = flatbuffers::VerifierOptions::default();
    let fb_graph = generated::root_as_tosa_graph_with_opts(&opts, bytes)
        .map_err(|e| TosaError::InvalidFlatbuffer(e.to_string()))?;

    let version = {
        let v = fb_graph.version();
        types::Version {
            major: v._major(),
            minor: v._minor(),
            patch: v._patch(),
            draft: v._draft(),
        }
    };

    let regions = match fb_graph.regions() {
        Some(regions_vec) => {
            let mut out = Vec::with_capacity(regions_vec.len());
            for i in 0..regions_vec.len() {
                out.push(deserialize_region(regions_vec.get(i))?);
            }
            out
        }
        None => Vec::new(),
    };

    Ok(types::TosaGraph { version, regions })
}

fn deserialize_region(
    r: generated::TosaRegion<'_>,
) -> Result<types::TosaRegion, TosaError> {
    let name = r.name().unwrap_or("").to_string();
    let blocks = match r.blocks() {
        Some(blocks_vec) => {
            let mut out = Vec::with_capacity(blocks_vec.len());
            for i in 0..blocks_vec.len() {
                out.push(deserialize_block(blocks_vec.get(i))?);
            }
            out
        }
        None => Vec::new(),
    };
    Ok(types::TosaRegion { name, blocks })
}

fn deserialize_block(
    b: generated::TosaBasicBlock<'_>,
) -> Result<types::TosaBasicBlock, TosaError> {
    let name = b.name().unwrap_or("").to_string();
    let inputs = fb_strings_to_vec(b.inputs());
    let outputs = fb_strings_to_vec(b.outputs());

    let operators = match b.operators() {
        Some(ops_vec) => {
            let mut out = Vec::with_capacity(ops_vec.len());
            for i in 0..ops_vec.len() {
                out.push(deserialize_operator(ops_vec.get(i))?);
            }
            out
        }
        None => Vec::new(),
    };

    let tensors = match b.tensors() {
        Some(t_vec) => {
            let mut out = Vec::with_capacity(t_vec.len());
            for i in 0..t_vec.len() {
                out.push(deserialize_tensor(t_vec.get(i)));
            }
            out
        }
        None => Vec::new(),
    };

    Ok(types::TosaBasicBlock {
        name,
        operators,
        tensors,
        inputs,
        outputs,
    })
}

fn deserialize_tensor(t: generated::TosaTensor<'_>) -> types::TosaTensor {
    types::TosaTensor {
        name: t.name().unwrap_or("").to_string(),
        shape: fb_vec_i32(t.shape()),
        dtype: types::DType::from(t.type_()),
        data: t.data().map(|v| v.iter().collect()),
        variable: t.variable(),
        is_unranked: t.is_unranked(),
        variable_name: t.variable_name().map(|s| s.to_string()),
    }
}

fn deserialize_operator(
    op: generated::TosaOperator<'_>,
) -> Result<types::TosaOperator, TosaError> {
    let op_code = types::Op::from(op.op());
    let inputs = fb_strings_to_vec(op.inputs());
    let outputs = fb_strings_to_vec(op.outputs());
    let attribute = deserialize_attribute(&op)?;

    Ok(types::TosaOperator {
        op: op_code,
        attribute,
        inputs,
        outputs,
    })
}

fn deserialize_attribute(
    op: &generated::TosaOperator<'_>,
) -> Result<types::Attribute, TosaError> {
    let attr_type = op.attribute_type();

    if attr_type == generated::Attribute::NONE {
        return Ok(types::Attribute::None);
    }

    match attr_type {
        generated::Attribute::ArgMaxAttribute => {
            let a = op.attribute_as_arg_max_attribute()
                .ok_or(TosaError::MissingField("ArgMaxAttribute"))?;
            Ok(types::Attribute::ArgMax(types::ArgMaxAttribute {
                axis: a.axis(),
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::AvgPool2dAttribute => {
            let a = op.attribute_as_avg_pool_2d_attribute()
                .ok_or(TosaError::MissingField("AvgPool2dAttribute"))?;
            Ok(types::Attribute::AvgPool2d(types::AvgPool2dAttribute {
                kernel: fb_vec_i32(a.kernel()),
                stride: fb_vec_i32(a.stride()),
                pad: fb_vec_i32(a.pad()),
                input_zp: a.input_zp(),
                output_zp: a.output_zp(),
                acc_type: types::DType::from(a.acc_type()),
            }))
        }
        generated::Attribute::Conv2dAttribute => {
            let a = op.attribute_as_conv_2d_attribute()
                .ok_or(TosaError::MissingField("Conv2dAttribute"))?;
            Ok(types::Attribute::Conv2d(types::Conv2dAttribute {
                pad: fb_vec_i32(a.pad()),
                stride: fb_vec_i32(a.stride()),
                dilation: fb_vec_i32(a.dilation()),
                local_bound: a.local_bound(),
                acc_type: types::DType::from(a.acc_type()),
            }))
        }
        generated::Attribute::Conv3dAttribute => {
            let a = op.attribute_as_conv_3d_attribute()
                .ok_or(TosaError::MissingField("Conv3dAttribute"))?;
            Ok(types::Attribute::Conv3d(types::Conv3dAttribute {
                pad: fb_vec_i32(a.pad()),
                stride: fb_vec_i32(a.stride()),
                dilation: fb_vec_i32(a.dilation()),
                local_bound: a.local_bound(),
                acc_type: types::DType::from(a.acc_type()),
            }))
        }
        generated::Attribute::DepthwiseConv2dAttribute => {
            let a = op.attribute_as_depthwise_conv_2d_attribute()
                .ok_or(TosaError::MissingField("DepthwiseConv2dAttribute"))?;
            Ok(types::Attribute::DepthwiseConv2d(types::DepthwiseConv2dAttribute {
                pad: fb_vec_i32(a.pad()),
                stride: fb_vec_i32(a.stride()),
                dilation: fb_vec_i32(a.dilation()),
                local_bound: a.local_bound(),
                acc_type: types::DType::from(a.acc_type()),
            }))
        }
        generated::Attribute::FFT2dAttribute => {
            let a = op.attribute_as_fft2d_attribute()
                .ok_or(TosaError::MissingField("FFT2dAttribute"))?;
            Ok(types::Attribute::FFT2d(types::FFT2dAttribute {
                inverse: a.inverse(),
                local_bound: a.local_bound(),
            }))
        }
        generated::Attribute::MatMulAttribute => {
            let _a = op.attribute_as_mat_mul_attribute()
                .ok_or(TosaError::MissingField("MatMulAttribute"))?;
            Ok(types::Attribute::MatMul(types::MatMulAttribute {}))
        }
        generated::Attribute::MaxPool2dAttribute => {
            let a = op.attribute_as_max_pool_2d_attribute()
                .ok_or(TosaError::MissingField("MaxPool2dAttribute"))?;
            Ok(types::Attribute::MaxPool2d(types::MaxPool2dAttribute {
                kernel: fb_vec_i32(a.kernel()),
                stride: fb_vec_i32(a.stride()),
                pad: fb_vec_i32(a.pad()),
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::RFFT2dAttribute => {
            let a = op.attribute_as_rfft2d_attribute()
                .ok_or(TosaError::MissingField("RFFT2dAttribute"))?;
            Ok(types::Attribute::RFFT2d(types::RFFT2dAttribute {
                local_bound: a.local_bound(),
            }))
        }
        generated::Attribute::TransposeConv2dAttribute => {
            let a = op.attribute_as_transpose_conv_2d_attribute()
                .ok_or(TosaError::MissingField("TransposeConv2dAttribute"))?;
            Ok(types::Attribute::TransposeConv2d(types::TransposeConv2dAttribute {
                out_pad: fb_vec_i32(a.out_pad()),
                stride: fb_vec_i32(a.stride()),
                local_bound: a.local_bound(),
                acc_type: types::DType::from(a.acc_type()),
            }))
        }
        generated::Attribute::ClampAttribute => {
            let a = op.attribute_as_clamp_attribute()
                .ok_or(TosaError::MissingField("ClampAttribute"))?;
            Ok(types::Attribute::Clamp(types::ClampAttribute {
                min_val: fb_vec_u8(a.min_val()),
                max_val: fb_vec_u8(a.max_val()),
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::ErfAttribute => {
            let _a = op.attribute_as_erf_attribute()
                .ok_or(TosaError::MissingField("ErfAttribute"))?;
            Ok(types::Attribute::Erf(types::ErfAttribute {}))
        }
        generated::Attribute::SigmoidAttribute => {
            let _a = op.attribute_as_sigmoid_attribute()
                .ok_or(TosaError::MissingField("SigmoidAttribute"))?;
            Ok(types::Attribute::Sigmoid(types::SigmoidAttribute {}))
        }
        generated::Attribute::TanhAttribute => {
            let _a = op.attribute_as_tanh_attribute()
                .ok_or(TosaError::MissingField("TanhAttribute"))?;
            Ok(types::Attribute::Tanh(types::TanhAttribute {}))
        }
        generated::Attribute::AddAttribute => {
            let _a = op.attribute_as_add_attribute()
                .ok_or(TosaError::MissingField("AddAttribute"))?;
            Ok(types::Attribute::Add(types::AddAttribute {}))
        }
        generated::Attribute::ArithmeticRightShiftAttribute => {
            let a = op.attribute_as_arithmetic_right_shift_attribute()
                .ok_or(TosaError::MissingField("ArithmeticRightShiftAttribute"))?;
            Ok(types::Attribute::ArithmeticRightShift(types::ArithmeticRightShiftAttribute {
                round: a.round(),
            }))
        }
        generated::Attribute::BitwiseAndAttribute => {
            let _a = op.attribute_as_bitwise_and_attribute()
                .ok_or(TosaError::MissingField("BitwiseAndAttribute"))?;
            Ok(types::Attribute::BitwiseAnd(types::BitwiseAndAttribute {}))
        }
        generated::Attribute::BitwiseOrAttribute => {
            let _a = op.attribute_as_bitwise_or_attribute()
                .ok_or(TosaError::MissingField("BitwiseOrAttribute"))?;
            Ok(types::Attribute::BitwiseOr(types::BitwiseOrAttribute {}))
        }
        generated::Attribute::BitwiseXorAttribute => {
            let _a = op.attribute_as_bitwise_xor_attribute()
                .ok_or(TosaError::MissingField("BitwiseXorAttribute"))?;
            Ok(types::Attribute::BitwiseXor(types::BitwiseXorAttribute {}))
        }
        generated::Attribute::IntDivAttribute => {
            let _a = op.attribute_as_int_div_attribute()
                .ok_or(TosaError::MissingField("IntDivAttribute"))?;
            Ok(types::Attribute::IntDiv(types::IntDivAttribute {}))
        }
        generated::Attribute::LogicalAndAttribute => {
            let _a = op.attribute_as_logical_and_attribute()
                .ok_or(TosaError::MissingField("LogicalAndAttribute"))?;
            Ok(types::Attribute::LogicalAnd(types::LogicalAndAttribute {}))
        }
        generated::Attribute::LogicalLeftShiftAttribute => {
            let _a = op.attribute_as_logical_left_shift_attribute()
                .ok_or(TosaError::MissingField("LogicalLeftShiftAttribute"))?;
            Ok(types::Attribute::LogicalLeftShift(types::LogicalLeftShiftAttribute {}))
        }
        generated::Attribute::LogicalRightShiftAttribute => {
            let _a = op.attribute_as_logical_right_shift_attribute()
                .ok_or(TosaError::MissingField("LogicalRightShiftAttribute"))?;
            Ok(types::Attribute::LogicalRightShift(types::LogicalRightShiftAttribute {}))
        }
        generated::Attribute::LogicalOrAttribute => {
            let _a = op.attribute_as_logical_or_attribute()
                .ok_or(TosaError::MissingField("LogicalOrAttribute"))?;
            Ok(types::Attribute::LogicalOr(types::LogicalOrAttribute {}))
        }
        generated::Attribute::LogicalXorAttribute => {
            let _a = op.attribute_as_logical_xor_attribute()
                .ok_or(TosaError::MissingField("LogicalXorAttribute"))?;
            Ok(types::Attribute::LogicalXor(types::LogicalXorAttribute {}))
        }
        generated::Attribute::MaximumAttribute => {
            let a = op.attribute_as_maximum_attribute()
                .ok_or(TosaError::MissingField("MaximumAttribute"))?;
            Ok(types::Attribute::Maximum(types::MaximumAttribute {
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::MinimumAttribute => {
            let a = op.attribute_as_minimum_attribute()
                .ok_or(TosaError::MissingField("MinimumAttribute"))?;
            Ok(types::Attribute::Minimum(types::MinimumAttribute {
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::MulAttribute => {
            let _a = op.attribute_as_mul_attribute()
                .ok_or(TosaError::MissingField("MulAttribute"))?;
            Ok(types::Attribute::Mul(types::MulAttribute {}))
        }
        generated::Attribute::PowAttribute => {
            let _a = op.attribute_as_pow_attribute()
                .ok_or(TosaError::MissingField("PowAttribute"))?;
            Ok(types::Attribute::Pow(types::PowAttribute {}))
        }
        generated::Attribute::SubAttribute => {
            let _a = op.attribute_as_sub_attribute()
                .ok_or(TosaError::MissingField("SubAttribute"))?;
            Ok(types::Attribute::Sub(types::SubAttribute {}))
        }
        generated::Attribute::TableAttribute => {
            let _a = op.attribute_as_table_attribute()
                .ok_or(TosaError::MissingField("TableAttribute"))?;
            Ok(types::Attribute::Table(types::TableAttribute {}))
        }
        generated::Attribute::AbsAttribute => {
            let _a = op.attribute_as_abs_attribute()
                .ok_or(TosaError::MissingField("AbsAttribute"))?;
            Ok(types::Attribute::Abs(types::AbsAttribute {}))
        }
        generated::Attribute::BitwiseNotAttribute => {
            let _a = op.attribute_as_bitwise_not_attribute()
                .ok_or(TosaError::MissingField("BitwiseNotAttribute"))?;
            Ok(types::Attribute::BitwiseNot(types::BitwiseNotAttribute {}))
        }
        generated::Attribute::CeilAttribute => {
            let _a = op.attribute_as_ceil_attribute()
                .ok_or(TosaError::MissingField("CeilAttribute"))?;
            Ok(types::Attribute::Ceil(types::CeilAttribute {}))
        }
        generated::Attribute::ClzAttribute => {
            let _a = op.attribute_as_clz_attribute()
                .ok_or(TosaError::MissingField("ClzAttribute"))?;
            Ok(types::Attribute::Clz(types::ClzAttribute {}))
        }
        generated::Attribute::CosAttribute => {
            let _a = op.attribute_as_cos_attribute()
                .ok_or(TosaError::MissingField("CosAttribute"))?;
            Ok(types::Attribute::Cos(types::CosAttribute {}))
        }
        generated::Attribute::ExpAttribute => {
            let _a = op.attribute_as_exp_attribute()
                .ok_or(TosaError::MissingField("ExpAttribute"))?;
            Ok(types::Attribute::Exp(types::ExpAttribute {}))
        }
        generated::Attribute::FloorAttribute => {
            let _a = op.attribute_as_floor_attribute()
                .ok_or(TosaError::MissingField("FloorAttribute"))?;
            Ok(types::Attribute::Floor(types::FloorAttribute {}))
        }
        generated::Attribute::LogAttribute => {
            let _a = op.attribute_as_log_attribute()
                .ok_or(TosaError::MissingField("LogAttribute"))?;
            Ok(types::Attribute::Log(types::LogAttribute {}))
        }
        generated::Attribute::LogicalNotAttribute => {
            let _a = op.attribute_as_logical_not_attribute()
                .ok_or(TosaError::MissingField("LogicalNotAttribute"))?;
            Ok(types::Attribute::LogicalNot(types::LogicalNotAttribute {}))
        }
        generated::Attribute::NegateAttribute => {
            let a = op.attribute_as_negate_attribute()
                .ok_or(TosaError::MissingField("NegateAttribute"))?;
            Ok(types::Attribute::Negate(types::NegateAttribute {
                input1_zp: a.input1_zp(),
                output_zp: a.output_zp(),
            }))
        }
        generated::Attribute::ReciprocalAttribute => {
            let _a = op.attribute_as_reciprocal_attribute()
                .ok_or(TosaError::MissingField("ReciprocalAttribute"))?;
            Ok(types::Attribute::Reciprocal(types::ReciprocalAttribute {}))
        }
        generated::Attribute::RsqrtAttribute => {
            let _a = op.attribute_as_rsqrt_attribute()
                .ok_or(TosaError::MissingField("RsqrtAttribute"))?;
            Ok(types::Attribute::Rsqrt(types::RsqrtAttribute {}))
        }
        generated::Attribute::SinAttribute => {
            let _a = op.attribute_as_sin_attribute()
                .ok_or(TosaError::MissingField("SinAttribute"))?;
            Ok(types::Attribute::Sin(types::SinAttribute {}))
        }
        generated::Attribute::SelectAttribute => {
            let _a = op.attribute_as_select_attribute()
                .ok_or(TosaError::MissingField("SelectAttribute"))?;
            Ok(types::Attribute::Select(types::SelectAttribute {}))
        }
        generated::Attribute::EqualAttribute => {
            let _a = op.attribute_as_equal_attribute()
                .ok_or(TosaError::MissingField("EqualAttribute"))?;
            Ok(types::Attribute::Equal(types::EqualAttribute {}))
        }
        generated::Attribute::GreaterAttribute => {
            let _a = op.attribute_as_greater_attribute()
                .ok_or(TosaError::MissingField("GreaterAttribute"))?;
            Ok(types::Attribute::Greater(types::GreaterAttribute {}))
        }
        generated::Attribute::GreaterEqualAttribute => {
            let _a = op.attribute_as_greater_equal_attribute()
                .ok_or(TosaError::MissingField("GreaterEqualAttribute"))?;
            Ok(types::Attribute::GreaterEqual(types::GreaterEqualAttribute {}))
        }
        generated::Attribute::ReduceAllAttribute => {
            let a = op.attribute_as_reduce_all_attribute()
                .ok_or(TosaError::MissingField("ReduceAllAttribute"))?;
            Ok(types::Attribute::ReduceAll(types::ReduceAllAttribute {
                axis: a.axis(),
            }))
        }
        generated::Attribute::ReduceAnyAttribute => {
            let a = op.attribute_as_reduce_any_attribute()
                .ok_or(TosaError::MissingField("ReduceAnyAttribute"))?;
            Ok(types::Attribute::ReduceAny(types::ReduceAnyAttribute {
                axis: a.axis(),
            }))
        }
        generated::Attribute::ReduceMaxAttribute => {
            let a = op.attribute_as_reduce_max_attribute()
                .ok_or(TosaError::MissingField("ReduceMaxAttribute"))?;
            Ok(types::Attribute::ReduceMax(types::ReduceMaxAttribute {
                axis: a.axis(),
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::ReduceMinAttribute => {
            let a = op.attribute_as_reduce_min_attribute()
                .ok_or(TosaError::MissingField("ReduceMinAttribute"))?;
            Ok(types::Attribute::ReduceMin(types::ReduceMinAttribute {
                axis: a.axis(),
                nan_mode: types::NanPropagationMode::from(a.nan_mode()),
            }))
        }
        generated::Attribute::ReduceProductAttribute => {
            let a = op.attribute_as_reduce_product_attribute()
                .ok_or(TosaError::MissingField("ReduceProductAttribute"))?;
            Ok(types::Attribute::ReduceProduct(types::ReduceProductAttribute {
                axis: a.axis(),
            }))
        }
        generated::Attribute::ReduceSumAttribute => {
            let a = op.attribute_as_reduce_sum_attribute()
                .ok_or(TosaError::MissingField("ReduceSumAttribute"))?;
            Ok(types::Attribute::ReduceSum(types::ReduceSumAttribute {
                axis: a.axis(),
            }))
        }
        generated::Attribute::ConcatAttribute => {
            let a = op.attribute_as_concat_attribute()
                .ok_or(TosaError::MissingField("ConcatAttribute"))?;
            Ok(types::Attribute::Concat(types::ConcatAttribute {
                axis: a.axis(),
            }))
        }
        generated::Attribute::PadAttribute => {
            let a = op.attribute_as_pad_attribute()
                .ok_or(TosaError::MissingField("PadAttribute"))?;
            Ok(types::Attribute::Pad(types::PadAttribute {
                pad_const: fb_vec_u8(a.pad_const()),
            }))
        }
        generated::Attribute::ReshapeAttribute => {
            let _a = op.attribute_as_reshape_attribute()
                .ok_or(TosaError::MissingField("ReshapeAttribute"))?;
            Ok(types::Attribute::Reshape(types::ReshapeAttribute {}))
        }
        generated::Attribute::ReverseAttribute => {
            let a = op.attribute_as_reverse_attribute()
                .ok_or(TosaError::MissingField("ReverseAttribute"))?;
            Ok(types::Attribute::Reverse(types::ReverseAttribute {
                axis: a.axis(),
            }))
        }
        generated::Attribute::SliceAttribute => {
            let _a = op.attribute_as_slice_attribute()
                .ok_or(TosaError::MissingField("SliceAttribute"))?;
            Ok(types::Attribute::Slice(types::SliceAttribute {}))
        }
        generated::Attribute::TileAttribute => {
            let _a = op.attribute_as_tile_attribute()
                .ok_or(TosaError::MissingField("TileAttribute"))?;
            Ok(types::Attribute::Tile(types::TileAttribute {}))
        }
        generated::Attribute::TransposeAttribute => {
            let a = op.attribute_as_transpose_attribute()
                .ok_or(TosaError::MissingField("TransposeAttribute"))?;
            Ok(types::Attribute::Transpose(types::TransposeAttribute {
                perms: fb_vec_i32(a.perms()),
            }))
        }
        generated::Attribute::GatherAttribute => {
            let _a = op.attribute_as_gather_attribute()
                .ok_or(TosaError::MissingField("GatherAttribute"))?;
            Ok(types::Attribute::Gather(types::GatherAttribute {}))
        }
        generated::Attribute::ScatterAttribute => {
            let _a = op.attribute_as_scatter_attribute()
                .ok_or(TosaError::MissingField("ScatterAttribute"))?;
            Ok(types::Attribute::Scatter(types::ScatterAttribute {}))
        }
        generated::Attribute::ResizeAttribute => {
            let a = op.attribute_as_resize_attribute()
                .ok_or(TosaError::MissingField("ResizeAttribute"))?;
            Ok(types::Attribute::Resize(types::ResizeAttribute {
                mode: types::ResizeMode::from(a.mode()),
            }))
        }
        generated::Attribute::CastAttribute => {
            let _a = op.attribute_as_cast_attribute()
                .ok_or(TosaError::MissingField("CastAttribute"))?;
            Ok(types::Attribute::Cast(types::CastAttribute {}))
        }
        generated::Attribute::RescaleAttribute => {
            let a = op.attribute_as_rescale_attribute()
                .ok_or(TosaError::MissingField("RescaleAttribute"))?;
            Ok(types::Attribute::Rescale(types::RescaleAttribute {
                input_zp: a.input_zp(),
                output_zp: a.output_zp(),
                scale32: a.scale32(),
                double_round: a.double_round(),
                per_channel: a.per_channel(),
                input_unsigned: a.input_unsigned(),
                output_unsigned: a.output_unsigned(),
            }))
        }
        generated::Attribute::ConstAttribute => {
            let _a = op.attribute_as_const_attribute()
                .ok_or(TosaError::MissingField("ConstAttribute"))?;
            Ok(types::Attribute::Const(types::ConstAttribute {}))
        }
        generated::Attribute::IdentityAttribute => {
            let _a = op.attribute_as_identity_attribute()
                .ok_or(TosaError::MissingField("IdentityAttribute"))?;
            Ok(types::Attribute::Identity(types::IdentityAttribute {}))
        }
        generated::Attribute::CustomAttribute => {
            let a = op.attribute_as_custom_attribute()
                .ok_or(TosaError::MissingField("CustomAttribute"))?;
            Ok(types::Attribute::Custom(types::CustomAttribute {
                operator_name: a.operator_name().unwrap_or("").to_string(),
                domain_name: a.domain_name().unwrap_or("").to_string(),
                implementation_attrs: fb_vec_u8(a.implementation_attrs()),
            }))
        }
        generated::Attribute::CondIfAttribute => {
            let a = op.attribute_as_cond_if_attribute()
                .ok_or(TosaError::MissingField("CondIfAttribute"))?;
            Ok(types::Attribute::CondIf(types::CondIfAttribute {
                then_graph: a.then_graph().unwrap_or("").to_string(),
                else_graph: a.else_graph().unwrap_or("").to_string(),
            }))
        }
        generated::Attribute::WhileLoopAttribute => {
            let a = op.attribute_as_while_loop_attribute()
                .ok_or(TosaError::MissingField("WhileLoopAttribute"))?;
            Ok(types::Attribute::WhileLoop(types::WhileLoopAttribute {
                cond_graph: a.cond_graph().unwrap_or("").to_string(),
                body_graph: a.body_graph().unwrap_or("").to_string(),
            }))
        }
        generated::Attribute::YieldAttribute => {
            let _a = op.attribute_as_yield_attribute()
                .ok_or(TosaError::MissingField("YieldAttribute"))?;
            Ok(types::Attribute::Yield(types::YieldAttribute {}))
        }
        generated::Attribute::VariableAttribute => {
            let _a = op.attribute_as_variable_attribute()
                .ok_or(TosaError::MissingField("VariableAttribute"))?;
            Ok(types::Attribute::Variable(types::VariableAttribute {}))
        }
        generated::Attribute::VariableWriteAttribute => {
            let _a = op.attribute_as_variable_write_attribute()
                .ok_or(TosaError::MissingField("VariableWriteAttribute"))?;
            Ok(types::Attribute::VariableWrite(types::VariableWriteAttribute {}))
        }
        generated::Attribute::VariableReadAttribute => {
            let _a = op.attribute_as_variable_read_attribute()
                .ok_or(TosaError::MissingField("VariableReadAttribute"))?;
            Ok(types::Attribute::VariableRead(types::VariableReadAttribute {}))
        }
        generated::Attribute::ConstShapeAttribute => {
            let _a = op.attribute_as_const_shape_attribute()
                .ok_or(TosaError::MissingField("ConstShapeAttribute"))?;
            Ok(types::Attribute::ConstShape(types::ConstShapeAttribute {}))
        }
        _ => Err(TosaError::UnknownAttribute(attr_type.0)),
    }
}

// ---------------------------------------------------------------------------
// Serialization: owned types -> FlatBuffers bytes
// ---------------------------------------------------------------------------

/// Serialize an owned `TosaGraph` into a FlatBuffer byte vector.
pub fn serialize(graph: &types::TosaGraph) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();

    // Build regions bottom-up
    let regions_offsets: Vec<_> = graph
        .regions
        .iter()
        .map(|r| serialize_region(&mut fbb, r))
        .collect();

    let regions_vec = if regions_offsets.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&regions_offsets))
    };

    // Build version
    let version = generated::Version::create(
        &mut fbb,
        &generated::VersionArgs {
            _major: graph.version.major,
            _minor: graph.version.minor,
            _patch: graph.version.patch,
            _draft: graph.version.draft,
        },
    );

    let tosa_graph = generated::TosaGraph::create(
        &mut fbb,
        &generated::TosaGraphArgs {
            version: Some(version),
            regions: regions_vec,
        },
    );

    generated::finish_tosa_graph_buffer(&mut fbb, tosa_graph);
    fbb.finished_data().to_vec()
}

fn serialize_region<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    region: &types::TosaRegion,
) -> flatbuffers::WIPOffset<generated::TosaRegion<'a>> {
    let name = fbb.create_string(&region.name);

    let block_offsets: Vec<_> = region
        .blocks
        .iter()
        .map(|b| serialize_block(fbb, b))
        .collect();

    let blocks_vec = if block_offsets.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&block_offsets))
    };

    generated::TosaRegion::create(
        fbb,
        &generated::TosaRegionArgs {
            name: Some(name),
            blocks: blocks_vec,
        },
    )
}

fn serialize_block<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    block: &types::TosaBasicBlock,
) -> flatbuffers::WIPOffset<generated::TosaBasicBlock<'a>> {
    let name = fbb.create_string(&block.name);

    // Build tensors
    let tensor_offsets: Vec<_> = block
        .tensors
        .iter()
        .map(|t| serialize_tensor(fbb, t))
        .collect();
    let tensors_vec = if tensor_offsets.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&tensor_offsets))
    };

    // Build operators
    let op_offsets: Vec<_> = block
        .operators
        .iter()
        .map(|op| serialize_operator(fbb, op))
        .collect();
    let operators_vec = if op_offsets.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&op_offsets))
    };

    // Build input/output string vectors
    let inputs = create_string_vector(fbb, &block.inputs);
    let outputs = create_string_vector(fbb, &block.outputs);

    generated::TosaBasicBlock::create(
        fbb,
        &generated::TosaBasicBlockArgs {
            name: Some(name),
            operators: operators_vec,
            tensors: tensors_vec,
            inputs,
            outputs,
        },
    )
}

fn create_string_vector<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    strings: &[String],
) -> Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<&'a str>>>> {
    if strings.is_empty() {
        return None;
    }
    let offsets: Vec<_> = strings.iter().map(|s| fbb.create_string(s)).collect();
    Some(fbb.create_vector(&offsets))
}

fn serialize_tensor<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    tensor: &types::TosaTensor,
) -> flatbuffers::WIPOffset<generated::TosaTensor<'a>> {
    let name = fbb.create_string(&tensor.name);
    let shape = if tensor.shape.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&tensor.shape))
    };
    let data = tensor.data.as_ref().map(|d| fbb.create_vector(d));
    let variable_name = tensor
        .variable_name
        .as_ref()
        .map(|s| fbb.create_string(s));

    generated::TosaTensor::create(
        fbb,
        &generated::TosaTensorArgs {
            name: Some(name),
            shape,
            type_: generated::DType::from(tensor.dtype),
            data,
            variable: tensor.variable,
            is_unranked: tensor.is_unranked,
            variable_name,
        },
    )
}

fn serialize_operator<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    op: &types::TosaOperator,
) -> flatbuffers::WIPOffset<generated::TosaOperator<'a>> {
    let inputs = create_string_vector(fbb, &op.inputs);
    let outputs = create_string_vector(fbb, &op.outputs);

    let (attr_type, attr_offset) = serialize_attribute(fbb, &op.attribute);

    generated::TosaOperator::create(
        fbb,
        &generated::TosaOperatorArgs {
            op: generated::Op::from(op.op),
            attribute_type: attr_type,
            attribute: attr_offset,
            inputs,
            outputs,
        },
    )
}

fn serialize_attribute<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    attr: &types::Attribute,
) -> (
    generated::Attribute,
    Option<flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>>,
) {
    match attr {
        types::Attribute::None => (generated::Attribute::NONE, None),

        types::Attribute::ArgMax(a) => {
            let o = generated::ArgMaxAttribute::create(
                fbb,
                &generated::ArgMaxAttributeArgs {
                    axis: a.axis,
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::ArgMaxAttribute, Some(o.as_union_value()))
        }

        types::Attribute::AvgPool2d(a) => {
            let kernel = if a.kernel.is_empty() { None } else { Some(fbb.create_vector(&a.kernel)) };
            let stride = if a.stride.is_empty() { None } else { Some(fbb.create_vector(&a.stride)) };
            let pad = if a.pad.is_empty() { None } else { Some(fbb.create_vector(&a.pad)) };
            let o = generated::AvgPool2dAttribute::create(
                fbb,
                &generated::AvgPool2dAttributeArgs {
                    kernel,
                    stride,
                    pad,
                    input_zp: a.input_zp,
                    output_zp: a.output_zp,
                    acc_type: generated::DType::from(a.acc_type),
                },
            );
            (generated::Attribute::AvgPool2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Conv2d(a) => {
            let pad = if a.pad.is_empty() { None } else { Some(fbb.create_vector(&a.pad)) };
            let stride = if a.stride.is_empty() { None } else { Some(fbb.create_vector(&a.stride)) };
            let dilation = if a.dilation.is_empty() { None } else { Some(fbb.create_vector(&a.dilation)) };
            let o = generated::Conv2dAttribute::create(
                fbb,
                &generated::Conv2dAttributeArgs {
                    pad,
                    stride,
                    dilation,
                    local_bound: a.local_bound,
                    acc_type: generated::DType::from(a.acc_type),
                },
            );
            (generated::Attribute::Conv2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Conv3d(a) => {
            let pad = if a.pad.is_empty() { None } else { Some(fbb.create_vector(&a.pad)) };
            let stride = if a.stride.is_empty() { None } else { Some(fbb.create_vector(&a.stride)) };
            let dilation = if a.dilation.is_empty() { None } else { Some(fbb.create_vector(&a.dilation)) };
            let o = generated::Conv3dAttribute::create(
                fbb,
                &generated::Conv3dAttributeArgs {
                    pad,
                    stride,
                    dilation,
                    local_bound: a.local_bound,
                    acc_type: generated::DType::from(a.acc_type),
                },
            );
            (generated::Attribute::Conv3dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::DepthwiseConv2d(a) => {
            let pad = if a.pad.is_empty() { None } else { Some(fbb.create_vector(&a.pad)) };
            let stride = if a.stride.is_empty() { None } else { Some(fbb.create_vector(&a.stride)) };
            let dilation = if a.dilation.is_empty() { None } else { Some(fbb.create_vector(&a.dilation)) };
            let o = generated::DepthwiseConv2dAttribute::create(
                fbb,
                &generated::DepthwiseConv2dAttributeArgs {
                    pad,
                    stride,
                    dilation,
                    local_bound: a.local_bound,
                    acc_type: generated::DType::from(a.acc_type),
                },
            );
            (generated::Attribute::DepthwiseConv2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::FFT2d(a) => {
            let o = generated::FFT2dAttribute::create(
                fbb,
                &generated::FFT2dAttributeArgs {
                    inverse: a.inverse,
                    local_bound: a.local_bound,
                },
            );
            (generated::Attribute::FFT2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::MatMul(_) => {
            let o = generated::MatMulAttribute::create(fbb, &generated::MatMulAttributeArgs {});
            (generated::Attribute::MatMulAttribute, Some(o.as_union_value()))
        }

        types::Attribute::MaxPool2d(a) => {
            let kernel = if a.kernel.is_empty() { None } else { Some(fbb.create_vector(&a.kernel)) };
            let stride = if a.stride.is_empty() { None } else { Some(fbb.create_vector(&a.stride)) };
            let pad = if a.pad.is_empty() { None } else { Some(fbb.create_vector(&a.pad)) };
            let o = generated::MaxPool2dAttribute::create(
                fbb,
                &generated::MaxPool2dAttributeArgs {
                    kernel,
                    stride,
                    pad,
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::MaxPool2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::RFFT2d(a) => {
            let o = generated::RFFT2dAttribute::create(
                fbb,
                &generated::RFFT2dAttributeArgs {
                    local_bound: a.local_bound,
                },
            );
            (generated::Attribute::RFFT2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::TransposeConv2d(a) => {
            let out_pad = if a.out_pad.is_empty() { None } else { Some(fbb.create_vector(&a.out_pad)) };
            let stride = if a.stride.is_empty() { None } else { Some(fbb.create_vector(&a.stride)) };
            let o = generated::TransposeConv2dAttribute::create(
                fbb,
                &generated::TransposeConv2dAttributeArgs {
                    out_pad,
                    stride,
                    local_bound: a.local_bound,
                    acc_type: generated::DType::from(a.acc_type),
                },
            );
            (generated::Attribute::TransposeConv2dAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Clamp(a) => {
            let min_val = if a.min_val.is_empty() { None } else { Some(fbb.create_vector(&a.min_val)) };
            let max_val = if a.max_val.is_empty() { None } else { Some(fbb.create_vector(&a.max_val)) };
            let o = generated::ClampAttribute::create(
                fbb,
                &generated::ClampAttributeArgs {
                    min_val,
                    max_val,
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::ClampAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Erf(_) => {
            let o = generated::ErfAttribute::create(fbb, &generated::ErfAttributeArgs {});
            (generated::Attribute::ErfAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Sigmoid(_) => {
            let o = generated::SigmoidAttribute::create(fbb, &generated::SigmoidAttributeArgs {});
            (generated::Attribute::SigmoidAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Tanh(_) => {
            let o = generated::TanhAttribute::create(fbb, &generated::TanhAttributeArgs {});
            (generated::Attribute::TanhAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Add(_) => {
            let o = generated::AddAttribute::create(fbb, &generated::AddAttributeArgs {});
            (generated::Attribute::AddAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ArithmeticRightShift(a) => {
            let o = generated::ArithmeticRightShiftAttribute::create(
                fbb,
                &generated::ArithmeticRightShiftAttributeArgs {
                    round: a.round,
                },
            );
            (generated::Attribute::ArithmeticRightShiftAttribute, Some(o.as_union_value()))
        }

        types::Attribute::BitwiseAnd(_) => {
            let o = generated::BitwiseAndAttribute::create(fbb, &generated::BitwiseAndAttributeArgs {});
            (generated::Attribute::BitwiseAndAttribute, Some(o.as_union_value()))
        }

        types::Attribute::BitwiseOr(_) => {
            let o = generated::BitwiseOrAttribute::create(fbb, &generated::BitwiseOrAttributeArgs {});
            (generated::Attribute::BitwiseOrAttribute, Some(o.as_union_value()))
        }

        types::Attribute::BitwiseXor(_) => {
            let o = generated::BitwiseXorAttribute::create(fbb, &generated::BitwiseXorAttributeArgs {});
            (generated::Attribute::BitwiseXorAttribute, Some(o.as_union_value()))
        }

        types::Attribute::IntDiv(_) => {
            let o = generated::IntDivAttribute::create(fbb, &generated::IntDivAttributeArgs {});
            (generated::Attribute::IntDivAttribute, Some(o.as_union_value()))
        }

        types::Attribute::LogicalAnd(_) => {
            let o = generated::LogicalAndAttribute::create(fbb, &generated::LogicalAndAttributeArgs {});
            (generated::Attribute::LogicalAndAttribute, Some(o.as_union_value()))
        }

        types::Attribute::LogicalLeftShift(_) => {
            let o = generated::LogicalLeftShiftAttribute::create(fbb, &generated::LogicalLeftShiftAttributeArgs {});
            (generated::Attribute::LogicalLeftShiftAttribute, Some(o.as_union_value()))
        }

        types::Attribute::LogicalRightShift(_) => {
            let o = generated::LogicalRightShiftAttribute::create(fbb, &generated::LogicalRightShiftAttributeArgs {});
            (generated::Attribute::LogicalRightShiftAttribute, Some(o.as_union_value()))
        }

        types::Attribute::LogicalOr(_) => {
            let o = generated::LogicalOrAttribute::create(fbb, &generated::LogicalOrAttributeArgs {});
            (generated::Attribute::LogicalOrAttribute, Some(o.as_union_value()))
        }

        types::Attribute::LogicalXor(_) => {
            let o = generated::LogicalXorAttribute::create(fbb, &generated::LogicalXorAttributeArgs {});
            (generated::Attribute::LogicalXorAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Maximum(a) => {
            let o = generated::MaximumAttribute::create(
                fbb,
                &generated::MaximumAttributeArgs {
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::MaximumAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Minimum(a) => {
            let o = generated::MinimumAttribute::create(
                fbb,
                &generated::MinimumAttributeArgs {
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::MinimumAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Mul(_) => {
            let o = generated::MulAttribute::create(fbb, &generated::MulAttributeArgs {});
            (generated::Attribute::MulAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Pow(_) => {
            let o = generated::PowAttribute::create(fbb, &generated::PowAttributeArgs {});
            (generated::Attribute::PowAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Sub(_) => {
            let o = generated::SubAttribute::create(fbb, &generated::SubAttributeArgs {});
            (generated::Attribute::SubAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Table(_) => {
            let o = generated::TableAttribute::create(fbb, &generated::TableAttributeArgs {});
            (generated::Attribute::TableAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Abs(_) => {
            let o = generated::AbsAttribute::create(fbb, &generated::AbsAttributeArgs {});
            (generated::Attribute::AbsAttribute, Some(o.as_union_value()))
        }

        types::Attribute::BitwiseNot(_) => {
            let o = generated::BitwiseNotAttribute::create(fbb, &generated::BitwiseNotAttributeArgs {});
            (generated::Attribute::BitwiseNotAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Ceil(_) => {
            let o = generated::CeilAttribute::create(fbb, &generated::CeilAttributeArgs {});
            (generated::Attribute::CeilAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Clz(_) => {
            let o = generated::ClzAttribute::create(fbb, &generated::ClzAttributeArgs {});
            (generated::Attribute::ClzAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Cos(_) => {
            let o = generated::CosAttribute::create(fbb, &generated::CosAttributeArgs {});
            (generated::Attribute::CosAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Exp(_) => {
            let o = generated::ExpAttribute::create(fbb, &generated::ExpAttributeArgs {});
            (generated::Attribute::ExpAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Floor(_) => {
            let o = generated::FloorAttribute::create(fbb, &generated::FloorAttributeArgs {});
            (generated::Attribute::FloorAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Log(_) => {
            let o = generated::LogAttribute::create(fbb, &generated::LogAttributeArgs {});
            (generated::Attribute::LogAttribute, Some(o.as_union_value()))
        }

        types::Attribute::LogicalNot(_) => {
            let o = generated::LogicalNotAttribute::create(fbb, &generated::LogicalNotAttributeArgs {});
            (generated::Attribute::LogicalNotAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Negate(a) => {
            let o = generated::NegateAttribute::create(
                fbb,
                &generated::NegateAttributeArgs {
                    input1_zp: a.input1_zp,
                    output_zp: a.output_zp,
                },
            );
            (generated::Attribute::NegateAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Reciprocal(_) => {
            let o = generated::ReciprocalAttribute::create(fbb, &generated::ReciprocalAttributeArgs {});
            (generated::Attribute::ReciprocalAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Rsqrt(_) => {
            let o = generated::RsqrtAttribute::create(fbb, &generated::RsqrtAttributeArgs {});
            (generated::Attribute::RsqrtAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Sin(_) => {
            let o = generated::SinAttribute::create(fbb, &generated::SinAttributeArgs {});
            (generated::Attribute::SinAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Select(_) => {
            let o = generated::SelectAttribute::create(fbb, &generated::SelectAttributeArgs {});
            (generated::Attribute::SelectAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Equal(_) => {
            let o = generated::EqualAttribute::create(fbb, &generated::EqualAttributeArgs {});
            (generated::Attribute::EqualAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Greater(_) => {
            let o = generated::GreaterAttribute::create(fbb, &generated::GreaterAttributeArgs {});
            (generated::Attribute::GreaterAttribute, Some(o.as_union_value()))
        }

        types::Attribute::GreaterEqual(_) => {
            let o = generated::GreaterEqualAttribute::create(fbb, &generated::GreaterEqualAttributeArgs {});
            (generated::Attribute::GreaterEqualAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ReduceAll(a) => {
            let o = generated::ReduceAllAttribute::create(
                fbb,
                &generated::ReduceAllAttributeArgs { axis: a.axis },
            );
            (generated::Attribute::ReduceAllAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ReduceAny(a) => {
            let o = generated::ReduceAnyAttribute::create(
                fbb,
                &generated::ReduceAnyAttributeArgs { axis: a.axis },
            );
            (generated::Attribute::ReduceAnyAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ReduceMax(a) => {
            let o = generated::ReduceMaxAttribute::create(
                fbb,
                &generated::ReduceMaxAttributeArgs {
                    axis: a.axis,
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::ReduceMaxAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ReduceMin(a) => {
            let o = generated::ReduceMinAttribute::create(
                fbb,
                &generated::ReduceMinAttributeArgs {
                    axis: a.axis,
                    nan_mode: generated::NanPropagationMode::from(a.nan_mode),
                },
            );
            (generated::Attribute::ReduceMinAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ReduceProduct(a) => {
            let o = generated::ReduceProductAttribute::create(
                fbb,
                &generated::ReduceProductAttributeArgs { axis: a.axis },
            );
            (generated::Attribute::ReduceProductAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ReduceSum(a) => {
            let o = generated::ReduceSumAttribute::create(
                fbb,
                &generated::ReduceSumAttributeArgs { axis: a.axis },
            );
            (generated::Attribute::ReduceSumAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Concat(a) => {
            let o = generated::ConcatAttribute::create(
                fbb,
                &generated::ConcatAttributeArgs { axis: a.axis },
            );
            (generated::Attribute::ConcatAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Pad(a) => {
            let pad_const = if a.pad_const.is_empty() { None } else { Some(fbb.create_vector(&a.pad_const)) };
            let o = generated::PadAttribute::create(
                fbb,
                &generated::PadAttributeArgs { pad_const },
            );
            (generated::Attribute::PadAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Reshape(_) => {
            let o = generated::ReshapeAttribute::create(fbb, &generated::ReshapeAttributeArgs {});
            (generated::Attribute::ReshapeAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Reverse(a) => {
            let o = generated::ReverseAttribute::create(
                fbb,
                &generated::ReverseAttributeArgs { axis: a.axis },
            );
            (generated::Attribute::ReverseAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Slice(_) => {
            let o = generated::SliceAttribute::create(fbb, &generated::SliceAttributeArgs {});
            (generated::Attribute::SliceAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Tile(_) => {
            let o = generated::TileAttribute::create(fbb, &generated::TileAttributeArgs {});
            (generated::Attribute::TileAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Transpose(a) => {
            let perms = if a.perms.is_empty() { None } else { Some(fbb.create_vector(&a.perms)) };
            let o = generated::TransposeAttribute::create(
                fbb,
                &generated::TransposeAttributeArgs { perms },
            );
            (generated::Attribute::TransposeAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Gather(_) => {
            let o = generated::GatherAttribute::create(fbb, &generated::GatherAttributeArgs {});
            (generated::Attribute::GatherAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Scatter(_) => {
            let o = generated::ScatterAttribute::create(fbb, &generated::ScatterAttributeArgs {});
            (generated::Attribute::ScatterAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Resize(a) => {
            let o = generated::ResizeAttribute::create(
                fbb,
                &generated::ResizeAttributeArgs {
                    mode: generated::ResizeMode::from(a.mode),
                },
            );
            (generated::Attribute::ResizeAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Cast(_) => {
            let o = generated::CastAttribute::create(fbb, &generated::CastAttributeArgs {});
            (generated::Attribute::CastAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Rescale(a) => {
            let o = generated::RescaleAttribute::create(
                fbb,
                &generated::RescaleAttributeArgs {
                    input_zp: a.input_zp,
                    output_zp: a.output_zp,
                    scale32: a.scale32,
                    double_round: a.double_round,
                    per_channel: a.per_channel,
                    input_unsigned: a.input_unsigned,
                    output_unsigned: a.output_unsigned,
                },
            );
            (generated::Attribute::RescaleAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Const(_) => {
            let o = generated::ConstAttribute::create(fbb, &generated::ConstAttributeArgs {});
            (generated::Attribute::ConstAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Identity(_) => {
            let o = generated::IdentityAttribute::create(fbb, &generated::IdentityAttributeArgs {});
            (generated::Attribute::IdentityAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Custom(a) => {
            let operator_name = fbb.create_string(&a.operator_name);
            let domain_name = fbb.create_string(&a.domain_name);
            let implementation_attrs = if a.implementation_attrs.is_empty() {
                None
            } else {
                Some(fbb.create_vector(&a.implementation_attrs))
            };
            let o = generated::CustomAttribute::create(
                fbb,
                &generated::CustomAttributeArgs {
                    operator_name: Some(operator_name),
                    domain_name: Some(domain_name),
                    implementation_attrs,
                },
            );
            (generated::Attribute::CustomAttribute, Some(o.as_union_value()))
        }

        types::Attribute::CondIf(a) => {
            let then_graph = fbb.create_string(&a.then_graph);
            let else_graph = fbb.create_string(&a.else_graph);
            let o = generated::CondIfAttribute::create(
                fbb,
                &generated::CondIfAttributeArgs {
                    then_graph: Some(then_graph),
                    else_graph: Some(else_graph),
                },
            );
            (generated::Attribute::CondIfAttribute, Some(o.as_union_value()))
        }

        types::Attribute::WhileLoop(a) => {
            let cond_graph = fbb.create_string(&a.cond_graph);
            let body_graph = fbb.create_string(&a.body_graph);
            let o = generated::WhileLoopAttribute::create(
                fbb,
                &generated::WhileLoopAttributeArgs {
                    cond_graph: Some(cond_graph),
                    body_graph: Some(body_graph),
                },
            );
            (generated::Attribute::WhileLoopAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Yield(_) => {
            let o = generated::YieldAttribute::create(fbb, &generated::YieldAttributeArgs {});
            (generated::Attribute::YieldAttribute, Some(o.as_union_value()))
        }

        types::Attribute::Variable(_) => {
            let o = generated::VariableAttribute::create(fbb, &generated::VariableAttributeArgs {});
            (generated::Attribute::VariableAttribute, Some(o.as_union_value()))
        }

        types::Attribute::VariableWrite(_) => {
            let o = generated::VariableWriteAttribute::create(fbb, &generated::VariableWriteAttributeArgs {});
            (generated::Attribute::VariableWriteAttribute, Some(o.as_union_value()))
        }

        types::Attribute::VariableRead(_) => {
            let o = generated::VariableReadAttribute::create(fbb, &generated::VariableReadAttributeArgs {});
            (generated::Attribute::VariableReadAttribute, Some(o.as_union_value()))
        }

        types::Attribute::ConstShape(_) => {
            let o = generated::ConstShapeAttribute::create(fbb, &generated::ConstShapeAttributeArgs {});
            (generated::Attribute::ConstShapeAttribute, Some(o.as_union_value()))
        }
    }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Load a TOSA FlatBuffer file and deserialize it into an owned `TosaGraph`.
pub fn load_file(path: &Path) -> Result<types::TosaGraph, TosaError> {
    let bytes = std::fs::read(path)?;
    deserialize(&bytes)
}

/// Serialize an owned `TosaGraph` and write it to a file.
pub fn save_file(graph: &types::TosaGraph, path: &Path) -> Result<(), TosaError> {
    let bytes = serialize(graph);
    std::fs::write(path, bytes)?;
    Ok(())
}
