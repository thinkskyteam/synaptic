use crate::core::generator::TextGeneration;
use crate::openai::http_entities::AppState;
use crate::openai::models::{
    ChatCompletionChoice, ChatCompletionRequestMessage, ChatCompletionResponseMessage,
    CompletionChoice, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateCompletionRequest, CreateCompletionResponse, CreateEmbeddingRequest,
    CreateEmbeddingResponse, DeleteModelResponse, Embedding, ListModelsResponse, Model, Stop,
};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::Utc;
use tracing::{debug, info, trace};
use uuid::Uuid;

/// Health check endpoint.
///
/// This function is called to check the health status of the service.
/// It logs some debug information about the model state and returns a static string indicating that the service is up.
///
/// # Arguments
///
/// * `state` - The application state.
///
/// # Returns
///
/// A static string indicating that the service is up.
pub async fn health(State(state): State<AppState>) -> &'static str {
    trace!("Health endpoint called");

    debug!("Model state is {}", state.device.is_metal());

    debug!("Model state is {}", state.device.is_cpu());

    debug!("Model state is {}", state.device.is_cuda());

    "Service is up!"
}

/// Creates a chat completion.
///
/// This function takes a `CreateChatCompletionRequest` as input and generates a chat completion response.
/// It extracts the necessary information from the request, such as temperature, top_p, and messages.
/// It then generates the chat completion using the `TextGeneration` struct and returns a `CreateChatCompletionResponse`.
///
/// # Arguments
///
/// * `state` - The application state.
/// * `request` - The `CreateChatCompletionRequest` containing the input parameters.
///
/// # Returns
///
/// A tuple containing the HTTP status code and the `CreateChatCompletionResponse` wrapped in `Json`.
pub async fn create_chat_completion(
    State(state): State<AppState>,
    Json(request): Json<CreateChatCompletionRequest>,
) -> impl IntoResponse {
    let request_tuple: (AppState, Option<f64>, Option<f64>, Option<usize>) =
        (state, request.temperature, request.top_p, None);
    let text_gen = TextGeneration::from(request_tuple);
    let max_tokens = request.max_tokens;

    let content_vec: Vec<_> = request
        .messages
        .into_iter()
        .map(|message| format!("{}:{}", message.role, message.content))
        .collect();
    let messages = content_vec.join(" ");
    info!("Messages {}", messages);

    let content_result = text_gen.generate(messages, max_tokens);

    let response = CreateChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "text_completion".to_string(),
        created: Utc::now().timestamp_millis(),
        model: "Llama-3.2-3B-Instruct".parse().unwrap(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: "assistant".to_string(),
                content: content_result.to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
    };

    info!("create_chat_completion is done");

    (StatusCode::OK, Json(response))
}

/// Creates a text completion.
///
/// This function takes a `CreateCompletionRequest` as input and generates a text completion response.
/// It extracts the necessary information from the request, such as temperature, top_p, prompt, and max_tokens.
/// It then generates the text completion using the `TextGeneration` struct and returns a `CreateCompletionResponse`.
///
/// # Arguments
///
/// * `state` - The application state.
/// * `request` - The `CreateCompletionRequest` containing the input parameters.
///
/// # Returns
///
/// A tuple containing the HTTP status code and the `CreateCompletionResponse` wrapped in `Json`.
pub async fn create_completion(
    State(state): State<AppState>,
    Json(request): Json<CreateCompletionRequest>,
) -> impl IntoResponse {
    let request_tuple: (AppState, Option<f64>, Option<f64>, Option<usize>) =
        (state, request.temperature, request.top_p, None);
    let text_gen = TextGeneration::from(request_tuple);

    let prompt = String::from(request.prompt.unwrap());
    let max_tokens = request.max_tokens;

    let result = text_gen.generate(prompt, max_tokens);

    let response = CreateCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "text_completion".to_string(),
        created: Utc::now().timestamp_millis(),
        model: "Llama-3.2-3B-Instruct".parse().unwrap(),
        choices: vec![CompletionChoice {
            text: result.to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
    };

    (StatusCode::OK, Json(response))
}

/// Creates an embedding.
///
/// This function takes a `CreateEmbeddingRequest` as input and generates an embedding response.
/// It currently returns a placeholder response with hardcoded values.
///
/// # Arguments
///
/// * `state` - The application state.
/// * `req` - The `CreateEmbeddingRequest` containing the input parameters.
///
/// # Returns
///
/// A tuple containing the HTTP status code and the `CreateEmbeddingResponse` wrapped in `Json`.
pub async fn create_embedding(
    State(state): State<AppState>,
    Json(req): Json<CreateEmbeddingRequest>,
) -> impl IntoResponse {
    // TODO: Process request and return response
    let response = CreateEmbeddingResponse {
        object: "list".to_string(),
        data: vec![Embedding {
            object: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            index: 0,
        }],
        model: req.model,
    };

    (StatusCode::OK, Json(response))
}

/// Lists available models.
///
/// This function returns a list of available models.
/// It currently returns a placeholder response with hardcoded values.
///
/// # Arguments
///
/// * `state` - The application state.
///
/// # Returns
///
/// A tuple containing the HTTP status code and the `ListModelsResponse` wrapped in `Json`.
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    // TODO: Fetch list of models and return response
    let response = ListModelsResponse {
        object: "list".to_string(),
        data: vec![
            Model {
                id: "model-1".to_string(),
                object: "model".to_string(),
                created: 1677652288,
                owned_by: "organization-1".to_string(),
            },
            Model {
                id: "model-2".to_string(),
                object: "model".to_string(),
                created: 1677652288,
                owned_by: "organization-1".to_string(),
            },
        ],
    };

    (StatusCode::OK, Json(response))
}

/// Retrieves a specific model.
///
/// This function retrieves details of a specific model identified by the `model_id` parameter.
/// It currently returns a placeholder response with hardcoded values.
///
/// # Arguments
///
/// * `state` - The application state.
/// * `model_id` - The ID of the model to retrieve.
///
/// # Returns
///
/// A tuple containing the HTTP status code and the `Model` wrapped in `Json`.
pub async fn retrieve_model(
    State(state): State<AppState>,
    Path(model_id): Path<Stop>,
) -> impl IntoResponse {
    // TODO: Fetch model details and return response

    let option_model_id = match model_id {
        Stop::String(s) => Some(s), // Extract String value
        Stop::Array(_) => None,     // Return None or handle as needed
    };
    let response = Model {
        id: option_model_id.unwrap(),
        object: "model".to_string(),
        created: 1677652288,
        owned_by: "organization-1".to_string(),
    };

    (StatusCode::OK, Json(response))
}

/// Deletes a specific model.
///
/// This function deletes a specific model identified by the `model_id` parameter.
/// It currently returns a placeholder response with hardcoded values.
///
/// # Arguments
///
/// * `state` - The application state.
/// * `model_id` - The ID of the model to delete.
///
/// # Returns
///
/// A tuple containing the HTTP status code and the `DeleteModelResponse` wrapped in `Json`.
pub async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<Stop>,
) -> impl IntoResponse {
    // TODO: Delete model and return response
    let option_model_id = match model_id {
        Stop::String(s) => Some(s), // Extract String value
        Stop::Array(_) => None,     // Return None or handle as needed
    };
    let response = DeleteModelResponse {
        id: option_model_id.unwrap(),
        object: "model".to_string(),
        deleted: true,
    };

    (StatusCode::OK, Json(response))
}
