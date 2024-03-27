
use crate::embeddings::BackendEmbedding;
use semanticsimilarity_rs::{cosine_similarity,manhattan_distance,};
use std::error::Error;
use serde_json::{Value};

pub async fn compute_cosine_similarity<T: BackendEmbedding + Sync>(
    backend: &T,
    text1: &str,
    text2: &str,
    model: &str,
) -> Result<f64, Box<dyn Error>> {
    let embeddings_json1 = backend.run(text1, model).await?;
    let embeddings_json2 = backend.run(text2, model).await?;

    let embeddings1 = parse_openai_embeddings(&embeddings_json1)?;
    let embeddings2 = parse_openai_embeddings(&embeddings_json2)?;

    let similarity = cosine_similarity(&embeddings1, &embeddings2);
    Ok(similarity)
}

fn parse_openai_embeddings(json_str: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let v: Value = serde_json::from_str(json_str)?;
    let embeddings = v["data"][0]["embedding"].as_array()
        .ok_or("Failed to parse embeddings")?
        .iter()
        .map(|val| val.as_f64().unwrap())
        .collect();
    Ok(embeddings)
    // parse ollama openai embeddings
}
