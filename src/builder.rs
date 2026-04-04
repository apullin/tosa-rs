use crate::types::*;

// ---------------------------------------------------------------------------
// Convenience constructors for TosaTensor
// ---------------------------------------------------------------------------

impl TosaTensor {
    /// Create a new tensor with a name, shape, and data type.
    pub fn new(name: &str, shape: Vec<i32>, dtype: DType) -> Self {
        TosaTensor {
            name: name.to_string(),
            shape,
            dtype,
            data: None,
            variable: false,
            is_unranked: false,
            variable_name: None,
            offset: 0,
            size: 0,
        }
    }

    /// Create a new tensor with constant data.
    pub fn with_data(name: &str, shape: Vec<i32>, dtype: DType, data: Vec<u8>) -> Self {
        TosaTensor {
            name: name.to_string(),
            shape,
            dtype,
            data: Some(data),
            variable: false,
            is_unranked: false,
            variable_name: None,
            offset: 0,
            size: 0,
        }
    }

    /// Create a variable tensor.
    pub fn variable(name: &str, shape: Vec<i32>, dtype: DType, var_name: &str) -> Self {
        TosaTensor {
            name: name.to_string(),
            shape,
            dtype,
            data: None,
            variable: true,
            is_unranked: false,
            variable_name: Some(var_name.to_string()),
            offset: 0,
            size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TosaGraphBuilder
// ---------------------------------------------------------------------------

pub struct TosaGraphBuilder {
    version: Version,
    regions: Vec<TosaRegion>,
}

impl TosaGraphBuilder {
    /// Create a new graph builder with the given version.
    pub fn new(major: i32, minor: i32, patch: i32) -> Self {
        TosaGraphBuilder {
            version: Version {
                major,
                minor,
                patch,
                draft: false,
            },
            regions: Vec::new(),
        }
    }

    /// Set whether this is a draft version.
    pub fn draft(mut self, draft: bool) -> Self {
        self.version.draft = draft;
        self
    }

    /// Begin building a new region with the given name.
    /// Returns a `RegionBuilder`; call `finish_region()` to return to this builder.
    pub fn add_region(self, name: &str) -> RegionBuilder {
        RegionBuilder {
            parent: self,
            name: name.to_string(),
            blocks: Vec::new(),
        }
    }

    /// Consume the builder and produce a `TosaGraph`.
    pub fn build(self) -> TosaGraph {
        TosaGraph {
            version: self.version,
            regions: self.regions,
            software_version: None,
        }
    }
}

// ---------------------------------------------------------------------------
// RegionBuilder
// ---------------------------------------------------------------------------

pub struct RegionBuilder {
    parent: TosaGraphBuilder,
    name: String,
    blocks: Vec<TosaBasicBlock>,
}

impl RegionBuilder {
    /// Begin building a new block within this region.
    /// Returns a `BlockBuilder`; call `finish_block()` to return to this builder.
    pub fn add_block(self, name: &str) -> BlockBuilder {
        BlockBuilder {
            parent: self,
            name: name.to_string(),
            tensors: Vec::new(),
            operators: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            shapes: Vec::new(),
        }
    }

    /// Finish this region and return to the parent graph builder.
    pub fn finish_region(mut self) -> TosaGraphBuilder {
        let region = TosaRegion {
            name: self.name,
            blocks: self.blocks,
        };
        self.parent.regions.push(region);
        self.parent
    }
}

// ---------------------------------------------------------------------------
// BlockBuilder
// ---------------------------------------------------------------------------

pub struct BlockBuilder {
    parent: RegionBuilder,
    name: String,
    tensors: Vec<TosaTensor>,
    operators: Vec<TosaOperator>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    shapes: Vec<TosaShape>,
}

impl BlockBuilder {
    /// Add a pre-built tensor to this block.
    pub fn add_tensor(mut self, tensor: TosaTensor) -> Self {
        self.tensors.push(tensor);
        self
    }

    /// Convenience: add a constant tensor with data.
    pub fn add_const_tensor(
        mut self,
        name: &str,
        shape: &[i32],
        dtype: DType,
        data: Vec<u8>,
    ) -> Self {
        self.tensors
            .push(TosaTensor::with_data(name, shape.to_vec(), dtype, data));
        self
    }

    /// Declare a block input by tensor name.
    pub fn add_input(mut self, name: &str) -> Self {
        self.inputs.push(name.to_string());
        self
    }

    /// Declare a block output by tensor name.
    pub fn add_output(mut self, name: &str) -> Self {
        self.outputs.push(name.to_string());
        self
    }

    /// Add an operator to this block.
    pub fn add_operator(
        mut self,
        op: Op,
        attribute: Attribute,
        inputs: &[&str],
        outputs: &[&str],
    ) -> Self {
        self.operators.push(TosaOperator {
            op,
            attribute,
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            location: None,
        });
        self
    }

    /// Finish this block and return to the parent region builder.
    pub fn finish_block(mut self) -> RegionBuilder {
        let block = TosaBasicBlock {
            name: self.name,
            operators: self.operators,
            tensors: self.tensors,
            inputs: self.inputs,
            outputs: self.outputs,
            shapes: self.shapes,
        };
        self.parent.blocks.push(block);
        self.parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_graph() {
        let graph = TosaGraphBuilder::new(1, 0, 0)
            .draft(false)
            .add_region("main")
            .add_block("entry")
            .add_input("input0")
            .add_output("output0")
            .add_tensor(TosaTensor::new("input0", vec![1, 3, 224, 224], DType::Fp32))
            .add_tensor(TosaTensor::new("output0", vec![1, 3, 224, 224], DType::Fp32))
            .add_operator(
                Op::Sigmoid,
                Attribute::Sigmoid(SigmoidAttribute {}),
                &["input0"],
                &["output0"],
            )
            .finish_block()
            .finish_region()
            .build();

        assert_eq!(graph.version.major, 1);
        assert_eq!(graph.version.minor, 0);
        assert_eq!(graph.version.patch, 0);
        assert!(!graph.version.draft);
        assert_eq!(graph.regions.len(), 1);
        assert_eq!(graph.regions[0].name, "main");
        assert_eq!(graph.regions[0].blocks.len(), 1);

        let block = &graph.regions[0].blocks[0];
        assert_eq!(block.name, "entry");
        assert_eq!(block.inputs, vec!["input0"]);
        assert_eq!(block.outputs, vec!["output0"]);
        assert_eq!(block.tensors.len(), 2);
        assert_eq!(block.operators.len(), 1);
        assert_eq!(block.operators[0].op, Op::Sigmoid);
    }

    #[test]
    fn tensor_constructors() {
        let t = TosaTensor::new("x", vec![2, 3], DType::Int32);
        assert_eq!(t.name, "x");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.dtype, DType::Int32);
        assert!(t.data.is_none());
        assert!(!t.variable);

        let t = TosaTensor::with_data("w", vec![4], DType::Fp32, vec![0u8; 16]);
        assert!(t.data.is_some());
        assert_eq!(t.data.as_ref().unwrap().len(), 16);

        let t = TosaTensor::variable("v", vec![1], DType::Fp32, "my_var");
        assert!(t.variable);
        assert_eq!(t.variable_name.as_deref(), Some("my_var"));
    }
}
