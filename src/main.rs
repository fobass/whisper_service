use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{mpsc, Mutex, Semaphore, RwLock};
use tower_http::cors::{Any, CorsLayer};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use hound;
use uuid::Uuid;
use dotenv::dotenv;
use std::env;
use redis::AsyncCommands;
use blake3::Hasher;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::fs::File;
use std::os::unix::io::IntoRawFd;

fn silence_stderr() {
    if let Ok(devnull) = File::open("/dev/null") {
        let _ = unsafe { libc::dup2(devnull.into_raw_fd(), libc::STDERR_FILENO) };
    }
}

// Константы для оптимизации
const MAX_CONCURRENT_JOBS: usize = 8;
const REDIS_CACHE_TTL: usize = 86400;
const FINGERPRINT_SIMILARITY_THRESHOLD: f32 = 0.92;
const QUEUE_CAPACITY: usize = 5000;

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<Job>,
    results: Arc<Mutex<HashMap<String, String>>>,
    redis: Arc<redis::Client>,
    whisper_ctx: Arc<WhisperContext>,
    fingerprint_cache: Arc<RwLock<HashMap<String, String>>>,
}

#[derive(Debug)]
struct Job {
    id: String,
    audio: Vec<f32>,
    fingerprint: String,
}

#[derive(Serialize)]
struct EnqueueResponse {
    job_id: String,
    status: String,
    cached_job_id: Option<String>,
}

#[derive(Serialize)]
struct ResultResponse {
    job_id: String,
    text: Option<String>,
    status: String,
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    silence_stderr();
    // Инициализация Redis
    let redis_url = env::var("REDIS_URL").unwrap_or("redis://127.0.0.1:6379".to_string());
    let redis_client = redis::Client::open(redis_url).expect("Failed to connect to Redis");

    let mut conn = redis_client.get_async_connection().await.expect("Failed to get Redis connection");
    let _: () = conn.set("health_check", "ok").await.expect("Redis health check failed");
    println!("✅ Redis connected successfully");

    let model_path = env::var("WHISPER_MODEL_PATH")
        .unwrap_or_else(|_| "/root/whisper.cpp/models/ggml-base.bin".to_string());

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("failed to load model");

    println!("✅ Whisper model loaded successfully");

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);


    let (tx, rx) = mpsc::channel::<Job>(QUEUE_CAPACITY);
    let results = Arc::new(Mutex::new(HashMap::new()));
    let concurrency_limiter = Arc::new(Semaphore::new(MAX_CONCURRENT_JOBS));
    let fingerprint_cache = Arc::new(RwLock::new(HashMap::new()));

    let ctx = Arc::new(ctx);
    let results_clone = results.clone();
    let redis_client_arc = Arc::new(redis_client);
    let fingerprint_cache_clone = fingerprint_cache.clone();

    let (tx, rx) = mpsc::channel::<Job>(QUEUE_CAPACITY);
    let rx = Arc::new(Mutex::new(rx));

    // Запуск воркеров
    for worker_id in 0..MAX_CONCURRENT_JOBS {
        let ctx = ctx.clone();
        let results = results_clone.clone();
        let redis = redis_client_arc.clone();
        let limiter = concurrency_limiter.clone();
        let fingerprint_cache = fingerprint_cache_clone.clone();
        let rx = rx.clone(); // clone Arc<Mutex<Receiver>>

        tokio::spawn(async move {
            loop {
                let job = {
                    let mut rx = rx.lock().await;
                    rx.recv().await
                };

                if let Some(job) = job {
                    let permit = limiter.acquire().await.expect("Semaphore closed");

                    process_job(
                        job,
                        ctx.clone(),
                        results.clone(),
                        redis.clone(),
                        fingerprint_cache.clone(),
                        worker_id
                    ).await;

                    drop(permit);
                } else {
                    break; // channel closed
                }
            }
        });
    }


    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .max_age(Duration::from_secs(3600));

    let state = AppState {
        tx,
        results,
        redis: redis_client_arc,
        whisper_ctx: ctx,
        fingerprint_cache,
    };

    let app = Router::new()
        .route("/transcribe", post(enqueue_transcription))
        .route("/result/{job_id}", get(get_result))
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    println!("🚀 Server listening on {}", listener.local_addr().unwrap());
    // println!("💪 Ready to handle 500+ concurrent users");

    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

async fn process_job(
    job: Job,
    ctx: Arc<WhisperContext>,
    results: Arc<Mutex<HashMap<String, String>>>,
    redis: Arc<redis::Client>,
    fingerprint_cache: Arc<RwLock<HashMap<String, String>>>,
    worker_id: usize
) {
    // println!("👷 Worker {} processing job {}", worker_id, job.id);

    let start_time = std::time::Instant::now();

    let text = tokio::task::spawn_blocking(move || {
        let mut wstate = ctx.create_state().expect("failed to create state");
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_translate(false);
        params.set_language(Some("ru"));

        if let Err(e) = wstate.full(params, &job.audio) {
            eprintln!("❌ Whisper error: {:?}", e);
            return None;
        }

        let num_segments = wstate.full_n_segments().unwrap_or(0);
        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = wstate.full_get_segment_text(i) {
                text.push_str(&segment);
            }
        }

        Some(text)
    }).await.unwrap_or(None);

    if let Some(text) = text {
        results.lock().await.insert(job.id.clone(), text.clone());

        let mut conn = redis.get_async_connection().await.expect("Redis connection failed");
        let _: () = conn.set_ex(&job.id, &text, REDIS_CACHE_TTL).await.expect("Redis set failed");

        fingerprint_cache.write().await.insert(job.fingerprint, job.id.clone());

        let duration = start_time.elapsed();
        // println!("✅ Worker {} finished job {} in {:?}", worker_id, job.id, duration);
    }
}

async fn enqueue_transcription(
    State(state): State<AppState>,
    body: bytes::Bytes,
) -> Json<EnqueueResponse> {
    let start_time = std::time::Instant::now();
    let audio_data = decode_wav_to_f32(&body);
    let fingerprint = generate_audio_fingerprint(&audio_data);

    {
        let cache = state.fingerprint_cache.read().await;
        if let Some(cached_job_id) = cache.get(&fingerprint) {
            let duration = start_time.elapsed();
            println!("⚡ Memory cache hit! Response time: {:?}", duration);

            return Json(EnqueueResponse {
                job_id: cached_job_id.clone(),
                status: "cached".to_string(),
                cached_job_id: Some(cached_job_id.clone()),
            });
        }
    }

    if let Some(similar_job_id) = find_similar_fingerprint(&state, &fingerprint).await {
        let duration = start_time.elapsed();
        println!("🔍 Similar audio found! Response time: {:?}", duration);

        return Json(EnqueueResponse {
            job_id: similar_job_id.clone(),
            status: "similar".to_string(),
            cached_job_id: Some(similar_job_id),
        });
    }

    let mut conn = state.redis.get_async_connection().await.expect("Redis connection failed");
    let redis_key = format!("fingerprint:{}", fingerprint);
    if let Ok(Some(cached_job_id)) = conn.get::<_, Option<String>>(&redis_key).await {
        state.fingerprint_cache.write().await.insert(fingerprint, cached_job_id.clone());

        let duration = start_time.elapsed();
        println!("🗄️ Redis cache hit! Response time: {:?}", duration);

        return Json(EnqueueResponse {
            job_id: cached_job_id.clone(),
            status: "cached".to_string(),
            cached_job_id: Some(cached_job_id),
        });
    }

    let job_id = Uuid::new_v4().to_string();

    let job = Job {
        id: job_id.clone(),
        audio: audio_data,
        fingerprint: fingerprint.clone(),
    };

    if state.tx.send(job).await.is_err() {
        return Json(EnqueueResponse {
            job_id: "queue_full".to_string(),
            status: "error".to_string(),
            cached_job_id: None,
        });
    }

    let _: () = conn.set_ex(redis_key, &job_id, REDIS_CACHE_TTL).await.expect("Redis set failed");

    let duration = start_time.elapsed();
    // println!("🔄 New job created in {:?}", duration);

    Json(EnqueueResponse {
        job_id,
        status: "processed".to_string(),
        cached_job_id: None,
    })
}

async fn get_result(
    Path(job_id): Path<String>,
    State(state): State<AppState>,
) -> Json<ResultResponse> {
    {
        let results = state.results.lock().await;
        if let Some(text) = results.get(&job_id) {
            return Json(ResultResponse {
                job_id,
                text: Some(text.clone()),
                status: "completed".to_string(),
            });
        }
    }

    let mut conn = state.redis.get_async_connection().await.expect("Redis connection failed");
    if let Ok(Some(text)) = conn.get::<_, Option<String>>(&job_id).await {
        // Сохраняем в памяти для будущих запросов
        state.results.lock().await.insert(job_id.clone(), text.clone());

        return Json(ResultResponse {
            job_id,
            text: Some(text),
            status: "completed".to_string(),
        });
    }

    Json(ResultResponse {
        job_id,
        text: None,
        status: "processing".to_string(),
    })
}

async fn health_check() -> Json<&'static str> {
    Json("✅ Server is healthy")
}

async fn get_stats(State(state): State<AppState>) -> Json<HashMap<String, String>> {
    let mut stats = HashMap::new();

    let cache_size = state.fingerprint_cache.read().await.len();
    let results_size = state.results.lock().await.len();

    stats.insert("memory_cache_size".to_string(), cache_size.to_string());
    stats.insert("results_size".to_string(), results_size.to_string());
    stats.insert("max_concurrent_jobs".to_string(), MAX_CONCURRENT_JOBS.to_string());

    Json(stats)
}

// Audio Fingerprint
fn generate_audio_fingerprint(audio_data: &[f32]) -> String {
    let mut hasher = Hasher::new();

    for (i, &sample) in audio_data.iter().enumerate() {
        if i % 50 == 0 {
            let quantized = (sample * 1000.0) as i32;
            hasher.update(&quantized.to_be_bytes());
        }

        if i > 10000 {
            break;
        }
    }

    let hash = hasher.finalize();
    BASE64.encode(hash.as_bytes())
}

async fn find_similar_fingerprint(state: &AppState, target_fp: &str) -> Option<String> {
    let cache = state.fingerprint_cache.read().await;

    for (stored_fp, job_id) in cache.iter() {
        if calculate_similarity(target_fp, stored_fp) > FINGERPRINT_SIMILARITY_THRESHOLD {
            return Some(job_id.clone());
        }
    }

    None
}

fn calculate_similarity(fp1: &str, fp2: &str) -> f32 {
    let min_len = std::cmp::min(fp1.len(), fp2.len());
    let mut matches = 0;

    for (c1, c2) in fp1.chars().zip(fp2.chars()).take(min_len) {
        if c1 == c2 {
            matches += 1;
        }
    }

    matches as f32 / min_len as f32
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