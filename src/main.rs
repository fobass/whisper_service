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
use symphonia::core::audio::Signal;

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
const TARGET_SAMPLE_RATE: u32 = 16000;

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<Job>,
    results: Arc<Mutex<HashMap<String, String>>>,
    redis: Arc<redis::Client>,
    whisper_ctx: Arc<WhisperContext>,
    fingerprint_cache: Arc<RwLock<HashMap<String, String>>>,
}

#[derive(Debug, Clone)]
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

    let model_path = env::var("WHISPER_MODEL_PATH")
        .unwrap_or_else(|_| "/root/whisper.cpp/models/ggml-small.bin".to_string());

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("failed to load model");


    println!("✅ Whisper model loaded successfully with base model.");

    let (tx, rx) = mpsc::channel::<Job>(QUEUE_CAPACITY);
    let results = Arc::new(Mutex::new(HashMap::new()));
    let concurrency_limiter = Arc::new(Semaphore::new(MAX_CONCURRENT_JOBS));
    let fingerprint_cache = Arc::new(RwLock::new(HashMap::new()));
    let redis_client_arc = Arc::new(redis_client);

    let ctx = Arc::new(ctx);
    let results_clone = results.clone();
    let fingerprint_cache_clone = fingerprint_cache.clone();

    let rx = Arc::new(Mutex::new(rx));

    // Запуск воркеров
    for worker_id in 0..MAX_CONCURRENT_JOBS {
        let ctx = ctx.clone();
        let results = results_clone.clone();
        let redis = redis_client_arc.clone();
        let limiter = concurrency_limiter.clone();
        let fingerprint_cache = fingerprint_cache_clone.clone();
        let rx = rx.clone();

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
                    break;
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
    println!("🚀 Before starting server...");
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
    println!("🚀 After starting server...");

}

async fn process_job(
    job: Job,
    ctx: Arc<WhisperContext>,
    results: Arc<Mutex<HashMap<String, String>>>,
    redis: Arc<redis::Client>,
    fingerprint_cache: Arc<RwLock<HashMap<String, String>>>,
    worker_id: usize
) {
    let start_time = std::time::Instant::now();
    let job_id = job.id.clone();
    println!("👷 Worker {} started processing job {}", worker_id, job_id);

    // Debug: Check audio statistics
    let audio_mean = job.audio.iter().sum::<f32>() / job.audio.len() as f32;
    let audio_max = job.audio.iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let audio_min = job.audio.iter()
        .map(|x| x.abs())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let audio_energy: f32 = job.audio.iter().map(|x| x * x).sum::<f32>() / job.audio.len() as f32;

    println!("📊 Audio analysis:");
    println!("   Samples: {}", job.audio.len());
    println!("   Mean: {:.6}", audio_mean);
    println!("   Max: {:.6}", audio_max);
    println!("   Min: {:.6}", audio_min);
    println!("   Energy: {:.6}", audio_energy);
    println!("   Fingerprint: {}", job.fingerprint);

    // Check if audio is mostly silence
    if audio_energy < 0.0001 {
        println!("⚠️  Very low energy - likely silence or noise");
    }

    let text = tokio::task::spawn_blocking(move || {
        println!("🎯 Running whisper on {} samples", job.audio.len());

        let mut wstate = match ctx.create_state() {
            Ok(state) => state,
            Err(e) => {
                eprintln!("❌ Failed to create state: {:?}", e);
                return None;
            }
        };

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_translate(false);
        params.set_language(Some("ru"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Try without suppression first
        params.set_suppress_blank(false);
        params.set_suppress_non_speech_tokens(false);

        match wstate.full(params, &job.audio) {
            Ok(_) => {
                let num_segments = wstate.full_n_segments().unwrap_or(0);
                println!("📝 Got {} segments", num_segments);

                let mut text = String::new();
                for i in 0..num_segments {
                    match wstate.full_get_segment_text(i) {
                        Ok(segment) => {
                            println!("   Segment {}: '{}'", i, segment);
                            text.push_str(&segment);
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to get segment {}: {:?}", i, e);
                        }
                    }
                }

                Some(text)
            }
            Err(e) => {
                eprintln!("❌ Whisper error: {:?}", e);
                None
            }
        }
    }).await.unwrap_or_else(|e| {
        eprintln!("❌ Task failed: {:?}", e);
        None
    });

    if let Some(text) = text {
        println!("✅ Final result: '{}'", text);

        // Store results
        results.lock().await.insert(job_id.clone(), text.clone());

        match redis.get_async_connection().await {
            Ok(mut conn) => {
                if let Err(e) = conn.set_ex::<&str, &str, ()>(&job_id, &text, REDIS_CACHE_TTL).await {
                    eprintln!("❌ Redis set failed: {:?}", e);
                }
            }
            Err(e) => {
                eprintln!("❌ Redis connection failed: {:?}", e);
            }
        }

        fingerprint_cache.write().await.insert(job.fingerprint, job_id.clone());

        let duration = start_time.elapsed();
        println!("✅ Completed in {:?}", duration);
    } else {
        eprintln!("❌ Processing failed for job {}", job_id);
    }
}

fn decode_wav_to_f32(data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let reader = hound::WavReader::new(cursor)?;
    let spec = reader.spec();

    println!("WAV spec: channels={}, sample_rate={}, bits_per_sample={}",
             spec.channels, spec.sample_rate, spec.bits_per_sample);

    if spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16 {
        let samples: Vec<f32> = reader.into_samples::<i16>()
            .map(|s| s.map(|sample| sample as f32 / 32768.0))
            .collect::<Result<Vec<f32>, _>>()?;

        Ok(samples)
    } else {
        Err("Only 16-bit PCM audio supported".into())
    }
}

fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f32 / to_rate as f32;
    let new_length = (samples.len() as f32 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_length);

    for i in 0..new_length {
        let pos = i as f32 * ratio;
        let index = pos as usize;
        let frac = pos - index as f32;

        if index + 1 < samples.len() {
            // Linear interpolation
            let sample = samples[index] * (1.0 - frac) + samples[index + 1] * frac;
            resampled.push(sample);
        } else {
            resampled.push(samples[samples.len() - 1]);
        }
    }

    resampled
}

fn convert_to_mono(samples: &[f32], num_channels: u16) -> Vec<f32> {
    if num_channels == 1 {
        return samples.to_vec();
    }

    let mut mono = Vec::with_capacity(samples.len() / num_channels as usize);

    for chunk in samples.chunks(num_channels as usize) {
        let sum: f32 = chunk.iter().sum();
        mono.push(sum / num_channels as f32);
    }

    mono
}

fn decode_webm_to_f32(data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // For now, let's just return an error since WebM decoding is complex
    // You might want to use a proper WebM/Opus decoder library
    Err("WebM decoding not implemented".into())
}

// Update the enqueue_transcription function
async fn enqueue_transcription(
    State(state): State<AppState>,
    body: bytes::Bytes,
) -> Json<EnqueueResponse> {
    let start_time = std::time::Instant::now();

    let audio_data = match decode_wav_to_f32(&body) {
        Ok(data) => {
            println!("✅ Decoded audio: {} samples", data.len());
            data
        }
        Err(e) => {
            eprintln!("❌ Failed to decode WAV audio: {}", e);
            return Json(EnqueueResponse {
                job_id: "decode_error".to_string(),
                status: "error".to_string(),
                cached_job_id: None,
            });
        }
    };

    let fingerprint = generate_audio_fingerprint(&audio_data);
    println!("🔑 Generated fingerprint: {}", fingerprint);

    // Check memory cache
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

    // Check for similar fingerprints with debug info
    println!("🔍 Checking for similar fingerprints...");
    if let Some(similar_job_id) = find_similar_fingerprint(&state, &fingerprint).await {
        let duration = start_time.elapsed();
        println!("🔍 Similar audio found! Response time: {:?}", duration);
        return Json(EnqueueResponse {
            job_id: similar_job_id.clone(),
            status: "similar".to_string(),
            cached_job_id: Some(similar_job_id),
        });
    }

    // Check Redis cache
    let mut conn = state.redis.get_async_connection().await.expect("Redis connection failed");
    let redis_key = format!("fingerprint:{}", fingerprint);
    if let Ok(Some(cached_job_id)) = conn.get::<_, Option<String>>(&redis_key).await {
        println!("🗄️ Redis cache hit for fingerprint: {}", fingerprint);
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
    println!("🆕 New job created: {}", job_id);

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

    let _: () = conn.set_ex::<&str, &str, _>(&redis_key, &job_id, REDIS_CACHE_TTL).await.expect("Redis set failed");

    let duration = start_time.elapsed();
    println!("🔄 New job created in {:?}", duration);

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
// Update the generate_audio_fingerprint function to be more unique
fn generate_audio_fingerprint(audio_data: &[f32]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash the audio length first
    audio_data.len().hash(&mut hasher);

    // Hash statistical features
    if !audio_data.is_empty() {
        let mean = audio_data.iter().sum::<f32>() / audio_data.len() as f32;
        let max = audio_data.iter().fold(0.0f32, |acc: f32, &x| acc.max(x.abs()));
        let min = audio_data.iter().fold(f32::INFINITY, |acc: f32, &x| acc.min(x.abs()));

        ((mean * 1000.0).round() as i32).hash(&mut hasher);
        ((max * 1000.0).round() as i32).hash(&mut hasher);
        ((min * 1000.0).round() as i32).hash(&mut hasher);

        // Hash spectral features (simple version)
        let energy: f32 = audio_data.iter().map(|x| x * x).sum();
        ((energy * 1000.0).round() as i32).hash(&mut hasher);
    }

    // Hash a sampling of the audio data
    let step = std::cmp::max(1, audio_data.len() / 500); // Sample more points
    for (i, &sample) in audio_data.iter().enumerate() {
        if i % step == 0 {
            let quantized = (sample * 10000.0).round() as i32; // More precision
            quantized.hash(&mut hasher);
        }

        if i > 20000 { // Limit to reasonable number
            break;
        }
    }

    // Add timestamp for extra uniqueness
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);

    let hash = hasher.finish();
    format!("{:016x}", hash)
}

// Also update the calculate_similarity function to be more accurate
fn calculate_similarity(fp1: &str, fp2: &str) -> f32 {
    if fp1.len() != fp2.len() {
        return 0.0;
    }

    let mut matches = 0;
    let mut total = 0;

    for (c1, c2) in fp1.chars().zip(fp2.chars()) {
        if c1 == c2 {
            matches += 1;
        }
        total += 1;
    }

    matches as f32 / total as f32
}

// And update the find_similar_fingerprint function to be more strict
async fn find_similar_fingerprint(state: &AppState, target_fp: &str) -> Option<String> {
    let cache = state.fingerprint_cache.read().await;

    for (stored_fp, job_id) in cache.iter() {
        let similarity = calculate_similarity(target_fp, stored_fp);
        println!("🔍 Comparing {} vs {}: similarity = {:.3}", target_fp, stored_fp, similarity);

        if similarity > FINGERPRINT_SIMILARITY_THRESHOLD {
            println!("🎯 Found similar audio: {}", job_id);
            return Some(job_id.clone());
        }
    }

    None
}

// fn decode_wav_to_f32(data: &[u8]) -> Vec<f32> {
//     use std::io::Cursor;
//     let reader = hound::WavReader::new(Cursor::new(data)).expect("invalid wav");
//     let spec = reader.spec();
//     assert_eq!(spec.channels, 1, "audio must be mono");
//     assert_eq!(spec.sample_rate, 16_000, "audio must be 16kHz");
//
//     reader
//         .into_samples::<i16>()
//         .map(|s| s.expect("invalid sample") as f32 / 32768.0)
//         .collect()
// }