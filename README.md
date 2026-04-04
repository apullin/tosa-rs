# tosa-rs

Rust serialization for [TOSA](https://www.mlplatform.org/tosa/) (Tensor Operator Set Architecture) FlatBuffers. Targets TOSA 1.0.

## Usage

Requires `flatc` on PATH (`brew install flatbuffers` / `apt install flatbuffers-compiler`).

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

79 ops, 79 attribute types, 19 data types, 6 enums. 169 tests.

## License

Apache-2.0
