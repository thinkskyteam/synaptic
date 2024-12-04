use crate::core::output_stream::TokenOutputStream;
use crate::openai::http_entities::AppState;
use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{Cache, Config, Llama as Llama3, LlamaEosToks};
use tokenizers::Tokenizer;
use tracing::info;

/// A struct representing text generation using the Llama3 model.
///
/// The `TextGeneration` struct contains fields for the Llama3 model, device,
/// tokenizer, logits processor, repeat penalty, repeat last n, and configuration.
/// It provides methods to create a new `TextGeneration` instance and generate
/// text based on a given prompt.
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
    /// Creates a new `TextGeneration` instance with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `model` - The Llama3 model to use for text generation.
    /// * `tokenizer` - The tokenizer to use for encoding and decoding text.
    /// * `seed` - The seed value for the random number generator.
    /// * `temperature` - Optional temperature value for sampling.
    /// * `top_p` - Optional top-p value for nucleus sampling.
    /// * `top_k` - Optional top-k value for nucleus sampling.
    /// * `repeat_penalty` - The repeat penalty value.
    /// * `repeat_last_n` - The number of last tokens to consider for repeat penalty.
    /// * `device` - The device to use for computations.
    /// * `config` - The configuration settings.
    ///
    /// # Returns
    ///
    /// A new `TextGeneration` instance with the specified parameters.
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
            let temperature = temperature.unwrap_or_else(|| 0f64);

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

    /// Generates text based on the given prompt and maximum number of tokens.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt string to use for text generation.
    /// * `max_tokens` - Optional maximum number of tokens to generate.
    ///
    /// # Returns
    ///
    /// The generated text as a string.
    pub(crate) fn generate(mut self, prompt: String, max_tokens: Option<i32>) -> String {
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

        let eos_token = self.config.eos_token_id.or_else(|| {
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

        for index in 0..max_tokens.unwrap_or_else(|| 064) {
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

            let logits = self
                .model
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
                )
                .unwrap()
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
                (token_generated - 1) as f64 / dt.as_secs_f64()
            )
        }

        string
    }
}

impl From<(AppState, Option<f64>, Option<f64>, Option<usize>)> for TextGeneration {
    /// Creates a new `TextGeneration` instance from an `AppState` tuple.
    ///
    /// # Arguments
    ///
    /// * `tuple` - A tuple containing the `AppState`, optional temperature,
    ///             optional top-p, and optional top-k values.
    ///
    /// # Returns
    ///
    /// A new `TextGeneration` instance with the specified parameters.
    fn from(tuple: (AppState, Option<f64>, Option<f64>, Option<usize>)) -> Self {
        let (app_state, temperature, top_p, top_k) = tuple;

        Self::new(
            app_state.model,
            app_state.tokenizer,
            299792458,   // seed RNG
            temperature, // temperature
            top_p,       // top_p - Nucleus sampling probability stuff
            top_k,       // top_k - Nucleus sampling probability stuff
            1.1,         // repeat penalty
            64,          // context size to consider for the repeat penalty
            &app_state.device,
            app_state.config,
        )
    }
}
