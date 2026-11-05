use tosa_rs::load_file;
fn dec(d: &[u8]) -> Vec<i64> { d.chunks(8).filter(|c| c.len()==8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect() }
fn main() {
    let g = load_file(std::path::Path::new(&std::env::args().nth(1).unwrap())).unwrap();
    for region in &g.regions { for block in &region.blocks {
        for t in &block.tensors {
            let dl = t.data.as_ref().map(|d| d.len()).unwrap_or(0);
            let vals = if format!("{:?}",t.dtype).to_lowercase().contains("shape") {
                t.data.as_ref().map(|d| format!("{:?}", dec(d))).unwrap_or_default()
            } else { String::new() };
            println!("  {} dtype={:?} shape={:?} datalen={} {}", t.name, t.dtype, t.shape, dl, vals);
        }
    }}
}
