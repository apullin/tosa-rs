use tosa_rs::load_file;
use tosa_rs::types::Attribute;
use std::collections::HashMap;
fn dec(d: &[u8]) -> Vec<i64> { d.chunks(8).filter(|c| c.len()==8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect() }
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let g = load_file(std::path::Path::new(&path)).unwrap();
    for region in &g.regions { for block in &region.blocks {
        let dmap: HashMap<String, Vec<i64>> = block.tensors.iter()
            .filter_map(|t| t.data.as_ref().filter(|d| !d.is_empty()).map(|d| (t.name.clone(), dec(d)))).collect();
        let tmap: HashMap<String, Vec<i32>> = block.tensors.iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
        for op in &block.operators {
            let extra = match &op.attribute {
                Attribute::Conv2d(a) => format!(" pad={:?} stride={:?} dil={:?}", a.pad, a.stride, a.dilation),
                Attribute::Transpose(a) => format!(" perms={:?}", a.perms),
                _ => String::new(),
            };
            // for ops with shape-const inputs, show their decoded values
            let vals: Vec<String> = op.inputs.iter().filter_map(|i| dmap.get(i).map(|v| format!("{}={:?}", i, v))).collect();
            let outshape = op.outputs.first().and_then(|o| tmap.get(o)).cloned().unwrap_or_default();
            println!("  {:?}{} -> {:?}  {}", op.op, extra, outshape, vals.join(" "));
        }
    }}
}
