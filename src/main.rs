use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{mpsc, Mutex, Semaphore};
use tower_http::cors::{Any, CorsLayer};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use hound;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<Job>,
    results: Arc<Mutex<HashMap<String, String>>>, // job_id -> transcript
}

#[derive(Debug)]
struct Job {
    id: String,
    audio: Vec<f32>,
}

#[derive(Serialize)]
struct EnqueueResponse {
    job_id: String,
}

#[derive(Serialize)]
struct ResultResponse {
    job_id: String,
    text: Option<String>,
}

#[tokio::main]
async fn main() {
    let ctx = WhisperContext::new_with_params("/root/whisper.cpp/models/ggml-base.bin", WhisperContextParameters::default())
        .expect("failed to load model");

    let (tx, mut rx) = mpsc::channel::<Job>(1000);
    let results: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

    let concurrency_limit = Arc::new(Semaphore::new(num_cpus::get()));

    let ctx = Arc::new(ctx);
    let results_clone = results.clone();
    let concurrency_limit_clone = concurrency_limit.clone();
    tokio::spawn(async move {
        while let Some(job) = rx.recv().await {
            let ctx = ctx.clone();
            let results = results_clone.clone();
            let permit = concurrency_limit_clone.clone().acquire_owned().await.unwrap();

            tokio::task::spawn_blocking(move || {
                println!("Worker picked job {}", job.id);

                let mut wstate = ctx.create_state().expect("failed to create state");

                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_translate(false);
                params.set_language(Some("ru"));

                if let Err(e) = wstate.full(params, &job.audio) {
                    eprintln!("Whisper error: {:?}", e);
                    return;
                }

                let num_segments = wstate.full_n_segments().unwrap_or(0);
                let mut text = String::new();
                for i in 0..num_segments {
                    if let Ok(segment) = wstate.full_get_segment_text(i) {
                        text.push_str(&segment);
                    }
                }

                tokio::runtime::Handle::current().block_on(async {
                    results.lock().await.insert(job.id.clone(), text);
                });

                println!("Job {} finished", job.id);
                drop(permit);
            });
        }
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .max_age(Duration::from_secs(3600));

    let state = AppState { tx, results };
    let app = Router::new()
        .route("/transcribe", post(enqueue_transcription))
        .route("/result/{job_id}", get(get_result))
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    println!("Server listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

async fn enqueue_transcription(
    State(state): State<AppState>,
    body: bytes::Bytes,
) -> Json<EnqueueResponse> {
    let audio_data = decode_wav_to_f32(&body);
    let job_id = Uuid::new_v4().to_string();

    let job = Job {
        id: job_id.clone(),
        audio: audio_data,
    };

    state.tx.send(job).await.expect("queue full");
    Json(EnqueueResponse { job_id })
}

async fn get_result(
    Path(job_id): Path<String>,
    State(state): State<AppState>,
) -> Json<ResultResponse> {
    let results = state.results.lock().await;
    let text = results.get(&job_id).cloned();
    Json(ResultResponse { job_id, text })
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

