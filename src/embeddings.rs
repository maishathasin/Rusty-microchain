
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use ollama_rs::{Ollama};
use ollama_rs::generation::completion::request::GenerationRequest;
use reqwest::Client;
use std::error::Error;


#[derive(Serialize)]
struct EmbeddingsRequest {
    input: String,
    model: String,
}

#[derive(Deserialize, Serialize)]
struct OpenAIEmbeddingsResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingsUsage,
}

#[derive(Deserialize,Serialize)]
struct EmbeddingData {
    object: String,
    index: u32,
    embedding: Vec<f64>,
}

#[derive(Deserialize,Serialize)]
struct EmbeddingsUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

pub struct OpenAIEmbeddings {
    client: Client,
    api_key: String,
}

impl OpenAIEmbeddings {
    pub fn new(api_key: String) -> Self {
        OpenAIEmbeddings { 
            client: Client::new(), 
            api_key,
        }
    }
}


#[async_trait]
pub trait BackendEmbedding {
    async fn run(&self, request: &str, model: &str) -> Result<String, Box<dyn Error>>;
}

#[async_trait]
impl BackendEmbedding for OpenAIEmbeddings {
    async fn run(&self, request: &str,model: &str) -> Result<String, Box<dyn Error>> {
        let embeddings_request = EmbeddingsRequest {
            input: request.to_string(),
            model: model.to_string(),
        };

        let response = self.client.post("https://api.openai.com/v1/embeddings")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&embeddings_request)
            .send()
            .await?;

        if response.status().is_success() {
            let embeddings_response: OpenAIEmbeddingsResponse = response.json::<OpenAIEmbeddingsResponse>().await?;
            Ok(serde_json::to_string(&embeddings_response)?)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Error: {:?}", response.status()),
            )))
        }
    }
}


// Ollama embeddings 


pub struct OllamaEmbeddings {
    ollama: Ollama,
}

impl OllamaEmbeddings {
    pub fn new(base_url: &str, port: u16) -> Self {
        let ollama = Ollama::new(base_url.to_string(), port);
        OllamaEmbeddings { ollama }
    }
}



#[async_trait]
impl BackendEmbedding for OllamaEmbeddings {
    async fn run(&self, model: &str, request: &str) -> Result<String, Box<dyn Error>> {
        let res = self.ollama.generate_embeddings(model.to_string(), request.to_string(), None).await
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        serde_json::to_string(&res.embeddings).map_err(|e| Box::new(e) as Box<dyn Error>)
    }
}


