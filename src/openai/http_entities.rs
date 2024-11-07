use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{Cache, Config, Llama as Llama3, LlamaEosToks};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::info;

use crate::core::output_stream::TokenOutputStream;
use crate::openai::models::{CreateCompletionRequest, Prompt};

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
    pub fn new(model: String, prompt: String, max_tokens: i32, temperature: Option<f64>, top_p: Option<f64>, top_k: Option<usize>) -> Self {
        Self { model, prompt, max_tokens, temperature, top_p, top_k }
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
    pub fn new(id: String, object: String, created: i64, model: String, choices: Vec<Choice>, usage: Usage) -> Self {
        Self { id, object, created, model, choices, usage }
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
        Self { text, index, logprobs, finish_reason }
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
        Self { prompt_tokens, completion_tokens, total_tokens }
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

pub struct TextGeneration {
    model: Llama3,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    pub(crate) config: Config,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Llama3,
        tokenizer: Tokenizer,
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        config: Config,
    ) -> Self {
        let logits_processor = {
            let temperature = temperature.unwrap();
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            config,
        }
    }

    pub(crate) fn create_completion_service(mut self, request: CreateCompletionRequest) -> String {
        let prompt = String::from(request.prompt.unwrap());
        let max_tokens = request.max_tokens.unwrap();

        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        info!("Got tokens!");

        let origin_config = self.config.clone();

        let eos_token = self.config
            .eos_token_id
            .or_else(|| {
                let option = self.tokenizer.tokenizer().token_to_id("</s>").unwrap();
                let toks = LlamaEosToks::Single(option);
                Some(toks)
            });


        let eos_token_value = match eos_token.clone().unwrap() {
            LlamaEosToks::Single(value) => value,
            _ => 0, // Handle other cases if necessary
        };

        info!("End of S {:?} token", eos_token_value);

        let mut string = String::new();

        let mut cache = Cache::new(false, DType::F16, &origin_config, &self.device).unwrap();

        let mut start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut token_generated = 0;


        for index in 0..max_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            if index == 1 {
                start_gen = std::time::Instant::now()
            }
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();

            let logits = self.model
                .forward(&input, context_index, &mut cache)
                .unwrap()
                .squeeze(0)
                .unwrap();

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                ).unwrap()
            };
            index_pos += ctxt.len();

            let next_token = self.logits_processor.sample(&logits).unwrap();
            token_generated += 1;
            tokens.push(next_token);

            //Diff
            match eos_token {
                Some(LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                    break;
                }
                Some(LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                    break;
                }
                _ => (),
            }
            if Some(next_token) == Some(eos_token_value) {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                info!("Found a token! {}", t);
                string.push_str(&t);
            }

            if let Some(rest) = self.tokenizer.decode_rest().map_err(Error::msg).unwrap() {
                print!("{rest}");
            }
            let dt = start_gen.elapsed();
            info!(
                    "{} tokens generated ({} token/s)",
                    token_generated,
                    (token_generated - 1) as f64 / dt.as_secs_f64())
        }

        string
    }
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

impl From<(AppState, Option<f64>, Option<f64>, Option<usize>)> for TextGeneration {
    fn from(tuple: (AppState, Option<f64>, Option<f64>, Option<usize>)) -> Self {
        let (app_state, temperature, top_p, top_k) = tuple;

        Self::new(
            app_state.model,
            app_state.tokenizer,
            299792458, // seed RNG
            temperature,  // temperature
            top_p,      // top_p - Nucleus sampling probability stuff
            top_k, // top_k - Nucleus sampling probability stuff
            1.1,       // repeat penalty
            64,        // context size to consider for the repeat penalty
            &app_state.device,
            app_state.config,
        )
    }
}
