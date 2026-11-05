# tosa-rs

Rust serialization for [TOSA](https://www.mlplatform.org/tosa/) (Tensor Operator Set Architecture) FlatBuffers. Targets TOSA 1.0 and 1.1.

## Usage

Building needs no `flatc`: the generated FlatBuffers code is committed in `src/generated/`.

```toml
[dependencies]
tosa-rs = { git = "https://github.com/apullin/tosa-rs" }
```

```rust
use tosa_rs::*;

// Read a .tosa file
let graph = load_file(Path::new("model.tosa")).unwrap();

// Build a graph
let graph = TosaGraphBuilder::new(1, 0, 0)
    .add_region("main")
    .add_block("block0")
    .add_operator(Op::Add, Attribute::Add(AddAttribute::default()), &["a", "b"], &["c"])
    .finish_block()
    .finish_region()
    .build();

// Serialize / deserialize
let bytes = serialize(&graph);
let roundtripped = deserialize(&bytes).unwrap();
```

## Coverage

97 ops, 97 attribute types, 19 data types, 6 enums. 170 tests.

## Schemas

`docs/tosa.fbs` is the TOSA 1.0 schema and `docs/tosa_1.1.fbs` the 1.1 schema. TOSA 1.1 only appends
ops and attribute tables to 1.0, so the code generated from the 1.1 schema reads and writes 1.0
graphs too. The version a graph declares comes from `TosaGraph::version`.

After changing a schema, regenerate `src/generated/tosa_generated.rs` with
`scripts/regen_generated.sh`. It insists on flatc 25.2.10, so the generated code only changes when a
schema does; set `FLATC=/path/to/flatc` if that isn't the one on PATH.

## License

Apache-2.0
