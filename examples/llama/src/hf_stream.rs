use half::{bf16, f16};
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor data with shape and converted FP32 bytes
struct StoredTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Try to download single shard model first
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }
    // Otherwise download sharded model
    let index_path = repo.get("model.safetensors.index.json")?;
    // Parse index to find shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    // Get unique shard files
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    // Download each shard
    for shard_file in &shard_files {
        repo.get(shard_file)?;
    }
    Ok(model_dir)
}

/// Convert tensor data to f32 vec
fn tensor_to_f32(tensor: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor.dtype();
    let data = tensor.data();

    match dtype {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data).to_vec(),
        Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let bf16_slice: &[bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        other => {
            panic!("Unsupported dtype for conversion: {other:?}");
        }
    }
}

fn resolve_shard_files(model_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let single = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    if single.exists() && !index_path.exists() {
        println!("Single shard model detected, converting to FP32...");
        return Ok(vec![single]);
    }

    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    let mut files: Vec<String> = index.weight_map.values().cloned().collect();
    files.sort();
    files.dedup();

    println!("Multi-shard model detected ({} shards)...", files.len());
    Ok(files.into_iter().map(|f| model_dir.join(f)).collect())
}

/// Combines sharded safetensors files into a single FP32 file.
///
/// This function:
/// 1. Loads tensors from shard(s)
/// 2. Converts all to FP32
/// 3. Writes combined file
pub fn combine_safetensors_to_fp32(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");

    // Skip if already combined
    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = resolve_shard_files(model_dir)?;

    // -- Pass 1: Collect metadata only --
    println!("Pass 1: Collecting tensor metadata");

    struct TensorMeta {
        shape: Vec<usize>,
        data_offset: u64,
        num_elements: usize,
    }

    let mut tensor_metas: HashMap<String, TensorMeta> = HashMap::new();
    let mut current_offset: u64 = 0;

    for shard_path in &shard_files {
        let file = File::open(shard_path)?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)?
        };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let num_elements: usize = shape.iter().product();
            let byte_size = (num_elements * 4) as u64;

            tensor_metas.insert(name.to_string(), TensorMeta { shape, data_offset: current_offset, num_elements });
            current_offset += byte_size;
        }
    }

    let mut header_map = serde_json::Map::new();
    header_map.insert("__metadata__".to_string(), serde_json::json!({}));

    for (name, meta) in &tensor_metas {
        let start = meta.data_offset;
        let end = start + (meta.num_elements as u64 * 4);
        header_map.insert(name.clone(), serde_json::json!({
            "dtype": "F32",
            "shape": meta.shape,
            "data_offset": [start, end]
        }));
    }

    let header_json = serde_json::to_string(&serde_json::Value::Object(header_map))?;
    let header_bytes = header_json.as_bytes();
    let padded_len = (header_bytes.len() + 7) & !7;
    let mut padded_header = header_bytes.to_vec();
    padded_header.resize(padded_len, b' ');

    // -- Pass 2: Write file: 8 byte header length + header + data placeholder --
    println!("Pass 2: Writing tensors shard-by-shard ...");

    let mut out = File::create(&output_path)?;

    let header_len = padded_len as u64;
    out.write_all(&header_len.to_le_bytes())?;
    out.write_all(&padded_header)?;

    for shard_path in &shard_files {
        println!(" Writing {}...", shard_path.file_name().unwrap().to_string_lossy());

        let file = File::open(shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)?};
        let st = SafeTensors::deserialize(&mmap)?;

        let mut shard_tensors: Vec<&str> = st.names().iter().collect();
        shard_tensors.sort_by_key(|name| tensor_metas[*name].data_offset);

        for name in shard_tensors {
            let tensor = st.tensor(name)?;
            let fp32_data = tensor_to_f32(&tensor);
            let bytes: &[u8] = bytemuck::cast_slice(&fp32_data);
            out.write_all(bytes)?;
        }

        out.flush()?;
    }

    println!("Combined FP32 model saved to {}", output_path.display());
    Ok(output_path)

}

/// Downloads a model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - model_combined.safetensors (FP32)
pub fn prepare_hf_model_stream(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_to_fp32(&model_dir)?;
    Ok(model_dir)
}
