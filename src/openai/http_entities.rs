use candle_core::Device;

use candle_transformers::models::llama::{Config, Llama as Llama3};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Serialize, Deserialize)]
pub struct CompletionsRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: i32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
}

impl CompletionsRequest {
    pub fn new(
        model: String,
        prompt: String,
        max_tokens: i32,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> Self {
        Self {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct CompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

impl CompletionResponse {
    pub fn new(
        id: String,
        object: String,
        created: i64,
        model: String,
        choices: Vec<Choice>,
        usage: Usage,
    ) -> Self {
        Self {
            id,
            object,
            created,
            model,
            choices,
            usage,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Choice {
    text: String,
    index: i64,
    logprobs: Option<f64>,
    finish_reason: String,
}

impl Choice {
    pub fn new(text: String, index: i64, logprobs: Option<f64>, finish_reason: String) -> Self {
        Self {
            text,
            index,
            logprobs,
            finish_reason,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Usage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

impl Usage {
    pub fn new(prompt_tokens: i64, completion_tokens: i64, total_tokens: i64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }
    }

    pub fn prompt_tokens(&self) -> i64 {
        self.prompt_tokens
    }
    pub fn completion_tokens(&self) -> i64 {
        self.completion_tokens
    }
    pub fn total_tokens(&self) -> i64 {
        self.total_tokens
    }
}

// #[derive(Deserialize)]
// pub struct Prompt {
//     pub prompt: String,
// }

#[derive(Clone)]
pub struct AppState {
    pub(crate) model: Llama3,
    pub(crate) device: Device,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) config: Config,
}

impl From<(Llama3, Device, Tokenizer, Config)> for AppState {
    fn from(e: (Llama3, Device, Tokenizer, Config)) -> Self {
        Self {
            model: e.0,
            device: e.1,
            tokenizer: e.2,
            config: e.3,
        }
    }
}
