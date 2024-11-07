use crate::openai::http_entities::{AppState, TextGeneration};
use crate::openai::models::{ChatCompletionChoice, ChatCompletionResponseMessage, CompletionChoice, CreateChatCompletionRequest, CreateChatCompletionResponse, CreateCompletionRequest, CreateCompletionResponse, CreateEmbeddingRequest, CreateEmbeddingResponse, DeleteModelResponse, Embedding, ListModelsResponse, Model, Stop};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::{DateTime, Utc};
use tracing::{debug, info};
use uuid::Uuid;

pub async fn health(State(state): State<AppState>) -> &'static str {
    info!("Health endpoint called");

    debug!("Model state is {}", state.device.is_metal());

    "Service is up!"
}

// pub async fn run_completions(
//     State(state): State<AppState>,
//     Json(request): Json<CompletionsRequest>) -> impl IntoResponse {
//     
//     let request_tuple: (AppState, Option<f64>, Option<f64>, Option<usize>) = (
//         state,
//         request.temperature,
//         None,
//         None,
//     );
//     let ai_gen = TextGeneration::from(request_tuple);
//     ai_gen.run(request)
// }

pub async fn create_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> impl IntoResponse {
    // TODO: Process request and return response
    let response = CreateChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1677652288,
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: "assistant".to_string(),
                content: "Hello there! How may I assist you today?".to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
    };

    (StatusCode::OK, Json(response))
}

pub async fn create_completion(
    State(state): State<AppState>,
    Json(request): Json<CreateCompletionRequest>) -> impl IntoResponse {
    let request_tuple: (AppState, Option<f64>, Option<f64>, Option<usize>) = (
        state,
        request.temperature,
        request.top_p,
        None,
    );
    let text_gen = TextGeneration::from(request_tuple);
    let result = text_gen.create_completion_service(request);

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

pub async fn create_embedding(
    State(state): State<AppState>,
    Json(req): Json<CreateEmbeddingRequest>) -> impl IntoResponse {
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

pub async fn retrieve_model(
    State(state): State<AppState>,
    Path(model_id): Path<Stop>) -> impl IntoResponse {
    // TODO: Fetch model details and return response

    let option_model_id = match model_id {
        Stop::String(s) => Some(s), // Extract String value
        Stop::Array(_) => None, // Return None or handle as needed
    };
    let response = Model {
        id: option_model_id.unwrap(),
        object: "model".to_string(),
        created: 1677652288,
        owned_by: "organization-1".to_string(),
    };

    (StatusCode::OK, Json(response))
}

pub async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<Stop>) -> impl IntoResponse {
    // TODO: Delete model and return response
    let option_model_id = match model_id {
        Stop::String(s) => Some(s), // Extract String value
        Stop::Array(_) => None, // Return None or handle as needed
    };
    let response = DeleteModelResponse {
        id: option_model_id.unwrap(),
        object: "model".to_string(),
        deleted: true,
    };

    (StatusCode::OK, Json(response))
}