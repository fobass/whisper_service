use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy, WhisperContextParameters};
use hound;
use axum::extract::DefaultBodyLimit;
use tower_http::cors::{CorsLayer, Any};
#[derive(Clone)]
struct AppState {
    ctx: Arc<Mutex<WhisperContext>>,
}

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[tokio::main]
async fn main() {

    let ctx = WhisperContext::new_with_params("/root/whisper.cpp/models/ggml-base.bin", WhisperContextParameters::default())
        .expect("failed to load model");

    let state = AppState {
        ctx: Arc::new(Mutex::new(ctx)),
    };
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .max_age(Duration::from_secs(3600));

    let app = Router::new()
        .route("/transcribe", post(transcribe))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("server listen on port : {}", listener.local_addr().unwrap());
    axum::serve(listener, app.into_make_service()).await.unwrap();
}

async fn transcribe(
    State(state): State<AppState>,
    body: bytes::Bytes,
) -> Json<TranscriptionResponse> {
    let audio_data: Vec<f32> = decode_wav_to_f32(&body);

    let ctx = state.ctx.lock().await;
    let mut wstate = ctx.create_state().expect("failed to create state");

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_translate(false);
    params.set_language(Some("ru"));

    wstate.full(params, &audio_data).expect("failed to run model");

    let num_segments = wstate.full_n_segments().expect("failed to get segments");
    let mut text = String::new();

    for i in 0..num_segments {
        let segment = wstate
            .full_get_segment_text(i)
            .unwrap_or_default(); // Result<String, WhisperError>
        text.push_str(&segment);
    }


    Json(TranscriptionResponse { text })
}
fn decode_wav_to_f32(data: &[u8]) -> Vec<f32> {
    use std::io::Cursor;
    let reader = hound::WavReader::new(Cursor::new(data)).expect("invalid wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "audio must be mono");
    assert_eq!(spec.sample_rate, 16_000, "audio must be 16kHz");

    reader
        .into_samples::<i16>()
        .map(|s| s.expect("invalid sample") as f32 / 32768.0)
        .collect()
}
