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

    let router = Router::new()
        .route("/health", get(health))
        // .route("/v1/completions", post(run_completions))
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
                    // Log the matched route's path (with placeholders not filled in).
                    // Use request.uri() or OriginalUri if you want the real path.
                    let matched_path = request
                        .extensions()
                        .get::<MatchedPath>()
                        .map(MatchedPath::as_str);

                    info_span!(
                        "http_request",
                        method = ?request.method(),
                        matched_path,
                        some_other_field = tracing::field::Empty,
                    )
                })
                .on_request(|_request: &Request<_>, _span: &Span| {
                    // You can use `_span.record("some_other_field", value)` in one of these
                    // closures to attach a value to the initially empty field in the info_span
                    // created above.
                })
                .on_response(|_response: &Response, _latency: Duration, _span: &Span| {
                    // ...
                })
                .on_body_chunk(|_chunk: &Bytes, _latency: Duration, _span: &Span| {
                    // ...
                })
                .on_eos(
                    |_trailers: Option<&HeaderMap>, _stream_duration: Duration, _span: &Span| {
                        // ...
                    },
                )
                .on_failure(
                    |_error: ServerErrorsFailureClass, _latency: Duration, _span: &Span| {
                        // ...
                    },
                ),
        );

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();

    axum::serve(tcp_listener, router).await.unwrap();

    Ok(())
}
