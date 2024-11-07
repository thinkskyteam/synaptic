use std::collections::HashSet;

use anyhow::Error as E;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama as Llama3, LlamaConfig};
use hf_hub::{Repo, RepoType};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use serde::{Deserialize, Deserializer};
use tokenizers::Tokenizer;

use crate::openai::http_entities::AppState;

pub fn hub_load_safe_tensors(repo: &ApiRepo,
                             json_file: &str, ) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: WeightMaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(pathbufs)
}

// Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(deserializer: D) -> anyhow::Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

fn get_tokenizer(repo: &ApiRepo) -> anyhow::Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;

    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}


fn get_config(repo: &ApiRepo) -> anyhow::Result<Config> {
    let config_filename = repo.get("config.json")?;

    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(false);

    Ok(config)

}

fn get_device() -> Device {
    let device_cuda = Device::new_cuda(0);
    let device_metal = Device::new_metal(0);

    let device = device_metal.or(device_cuda).unwrap_or(Device::Cpu);

    device
}

#[derive(Debug, Deserialize)]
struct WeightMaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

fn get_repo(token: String) -> anyhow::Result<ApiRepo> {
    let api = ApiBuilder::new().with_token(Some(token)).build()?;
    // "meta-llama/Meta-Llama-3.1-8B"
    // "591fbcb2d5b475bbfc7976a9214934652c64149b"
    let model_id = "meta-llama/Llama-3.2-3B-Instruct".to_string();
    Ok(api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "45026b798cd537efe6a1abcb93040ad21d416c43".to_string(),
    )))
}

pub fn initialise_model(token: String) -> anyhow::Result<AppState> {
    let repo = get_repo(token)?;
    let tokenizer = get_tokenizer(&repo)?;

    let device = get_device();

    let filenames = hub_load_safe_tensors(&repo, "model.safetensors.index.json")?;

    let config = get_config(&repo)?;

    let model = {
        let dtype = DType::F16;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        Llama3::load(vb, &config)?
    };

    Ok((model, device, tokenizer, config).into())
}