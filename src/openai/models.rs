use candle_core::{DType, Device};
use candle_transformers::models::llama::{Config, Llama};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokenizers::Tokenizer;
// Models

#[derive(Serialize, Deserialize)]
pub struct CreateChatCompletionRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub c: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<ParallelToolCalls>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionFunctions>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseFormat {
    Text(ResponseFormatText),
    JsonObject(ResponseFormatJsonObject),
    JsonSchema(ResponseFormatJsonSchema),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseFormatText {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseFormatJsonObject {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseFormatJsonSchema {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionStreamOptions {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionTool {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionToolChoiceOption {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ParallelToolCalls {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum Stop {
    String(String),
    Array(Vec<String>),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum FunctionCall {
    String(String),
    ChatCompletionFunctionCallOption(ChatCompletionFunctionCallOption),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionFunctionCallOption {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionFunctions {
    // Implement the fields based on the OpenAPI spec
    // ...
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ChatCompletionRequestMessage {
    pub(crate) role: String,
    pub(crate) content: String,
    // ... other fields
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CreateChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChatCompletionChoice>,
    pub(crate) usage: Option<CompletionUsage>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ChatCompletionChoice {
    pub(crate) index: i64,
    pub(crate) message: ChatCompletionResponseMessage,
    pub(crate) finish_reason: String,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ChatCompletionResponseMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCompletionRequest {
    pub model: String,
    pub prompt: Option<String>,
    pub best_of: Option<i32>,
    pub echo: Option<bool>,
    pub frequency_penalty: Option<f64>,
    pub logit_bias: Option<HashMap<String, i32>>,
    pub logprobs: Option<i32>,
    pub max_tokens: Option<i32>,
    pub n: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub stop: Option<StopSequence>,
    pub stream: Option<bool>,
    pub suffix: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub user: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum Prompt {
    Single(String),
    ArrayOfStrings(Vec<String>),
    ArrayOfTokens(Vec<i32>),
    ArrayOfTokenArrays(Vec<Vec<i32>>),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Array(Vec<String>),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CreateCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) choices: Vec<CompletionChoice>,
    pub(crate) usage: Option<CompletionUsage>,
}

#[derive(Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: i64,
    pub logprobs: Option<i64>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionUsage {
    pub completion_tokens: i32,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Serialize, Deserialize)]
pub struct CreateEmbeddingRequest {
    pub model: String,
    pub input: String,
    // ... other fields
}

#[derive(Serialize, Deserialize)]
pub struct CreateEmbeddingResponse {
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Serialize, Deserialize)]
pub struct Embedding {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: i64,
}

#[derive(Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize)]
pub struct DeleteModelResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Clone)]
pub struct AppState {
    pub(crate) model: Llama,
    pub(crate) device: Device,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) config: Config,
    pub(crate) d_type: DType,
}

impl From<(Llama, Device, Tokenizer, Config, DType)> for AppState {
    fn from(e: (Llama, Device, Tokenizer, Config, DType)) -> Self {
        Self {
            model: e.0,
            device: e.1,
            tokenizer: e.2,
            config: e.3,
            d_type: e.4,
        }
    }
}
