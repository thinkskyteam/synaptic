use std::time::{Duration, Instant};

use anyhow::Result;
use axum::{
    body::Bytes,
    extract::MatchedPath,
    http::{HeaderMap, Request},
    response::Response,
    routing::{get, post},
    Router,
};

use synap_forge_llm::core::load_model::initialise_model;
use synap_forge_llm::openai::http_service::{
    create_chat_completion, create_completion, create_embedding, delete_model, health, list_models,
    retrieve_model,
};
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::trace::TraceLayer;
use tracing::log::error;
use tracing::{info, info_span, Span};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                "synap_forge_llm=debug,tower_http=debug,axum::rejection=trace".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let Ok(api_token) = std::env::var("HF_TOKEN") else {
        return Err(anyhow::anyhow!("Error getting HF_TOKEN env var"));
    };

    let before = Instant::now();
    info!("Model is loading in memory");

    let state = initialise_model(api_token)?;

    info!(
        "Model loaded and is ready now with Elapsed time: {:.2?}",
        before.elapsed()
    );

    let openai_router = Router::new()
        .route("/health", get(health))
        .route("/chat/completions", post(create_chat_completion))
        .route("/completions", post(create_completion))
        .route("/embeddings", post(create_embedding))
        .route("/models", get(list_models))
        .route(
            "/models/:model_id",
            get(retrieve_model).delete(delete_model),
        )
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<_>| {
                    // Create span with request details
                    let matched_path = request
                        .extensions()
                        .get::<MatchedPath>()
                        .map(MatchedPath::as_str);

                    info_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri(),
                        matched_path = matched_path,
                        version = ?request.version(),
                        headers = ?request.headers(),
                    )
                })
                .on_request(|request: &Request<_>, _span: &Span| {
                    // Log when request starts
                    info!(
                        "Started {} request to {} body {:?}",
                        request.method(),
                        request.uri(),
                        request.body()
                    );
                })
                .on_response(|response: &Response, latency: Duration, _span: &Span| {
                    // Log response details
                    info!(
                        "Response completed with body {:?} status {} in {:?}",
                        response.body(),
                        response.status(),
                        latency
                    );
                })
                .on_body_chunk(|chunk: &Bytes, latency: Duration, _span: &Span| {
                    // Log body chunk details
                    info!(
                        "Sent body chunk of size {} bytes after {:?}",
                        chunk.len(),
                        latency
                    );
                })
                .on_eos(
                    |trailers: Option<&HeaderMap>, stream_duration: Duration, _span: &Span| {
                        // Log end of stream
                        info!(
                            "Stream completed in {:?}, trailers: {:?}",
                            stream_duration, trailers
                        );
                    },
                )
                .on_failure(
                    |error: ServerErrorsFailureClass, latency: Duration, _span: &Span| {
                        // Log errors
                        error!("Request failed after {:?}: {:?}", latency, error);
                    },
                ),
        );

    let main_router = Router::new().nest("/v1", openai_router);

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();

    axum::serve(tcp_listener, main_router).await.unwrap();

    Ok(())
}
