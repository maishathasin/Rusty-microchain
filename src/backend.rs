use futures::stream::{Stream, once};
use std::pin::Pin;
use serde_json::{Value};
use serpapi_search_rust::serp_api_search::SerpApiSearch;
use std::collections::HashMap;
use futures::stream::StreamExt; 
use reqwest::Client;
use serde::{Serialize, Deserialize};
use std::process::{Command, Output};
use serde::ser::StdError;
use std::error::Error;
use tera::{Tera, Context};
use async_trait::async_trait;
use serde_json::json;
use semanticsimilarity_rs::{cosine_similarity,manhattan_distance,};








#[async_trait]
pub trait Backend {
    async fn run(&self, request: &str) -> String;
    async fn run_stream(&self, request: &str) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>>;
}


pub struct Google {
    api_key: String,
}

impl Google {
    pub fn new(api_key: String) -> Self {
        Google { api_key }
    }
}

#[async_trait]
impl Backend for Google {
    async fn run(&self, request: &str) -> String {
        let mut params = HashMap::<String, String>::new();
        params.insert("q".to_string(), request.to_string());
        params.insert("google_domain".to_string(), "google.com".to_string());

        let search = SerpApiSearch::google(params, self.api_key.clone());

        match search.json().await {
            Ok(results) => extract_answer_box(results),
            Err(_) => "Failed to fetch results".to_string(),
        }
    }

    async fn run_stream(&self, request: &str) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
        let response = self.run(request).await; // todo: test this 
        Box::pin(once(async move { Ok(response) })) // Wrap the response in ok
    }
}

fn extract_answer_box(results: Value) -> String {
    if let Some(answer_box) = results.get("answer_box") {
        if let Some(answer) = answer_box.get("answer") {
            return answer.as_str().unwrap_or_default().to_string();
        } else if let Some(snippet) = answer_box.get("snippet") {
            return snippet.as_str().unwrap_or_default().to_string();
        } else if let Some(highlighted) = answer_box.get("snippet_highlighted_words") {
            return highlighted[0].as_str().unwrap_or_default().to_string();
        }
    }

    if let Some(organic_results) = results.get("organic_results").and_then(|r| r.get(0)) {
        if let Some(snippet) = organic_results.get("snippet") {
            return snippet.as_str().unwrap_or_default().to_string();
        }
    }

    "No answer found".to_string()
}

/// Perplexity AI 


#[derive(Serialize, Deserialize)]
struct PerplexityMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct PerplexityRequestBody {
    model: String,
    messages: Vec<PerplexityMessage>,
}

#[derive(Serialize,Deserialize)]
struct Choice {
    message: PerplexityMessage,
}

#[derive(Serialize, Deserialize)] 
struct PerplexityResponse {
    choices: Vec<Choice>,
}

pub struct PerplexityAI {
    client: Client,
    api_token: String, 
}

impl PerplexityAI {
    pub fn new(api_token: String) -> Self {
        PerplexityAI { client: Client::new(), api_token }
    }
}

#[async_trait]
impl Backend for PerplexityAI {
    async fn run(&self, request: &str) -> String {
        let body = PerplexityRequestBody {
            model: "mistral-7b-instruct".to_string(),
            messages: vec![
                PerplexityMessage {
                    role: "system".to_string(),
                    content: "Be precise and concise.".to_string(),
                },
                PerplexityMessage {
                    role: "user".to_string(),
                    content: request.to_string(),
                },
            ],
        };

        let res = self.client.post("https://api.perplexity.ai/chat/completions")
            .bearer_auth(&self.api_token) 
            .json(&body)
            .send()
            .await
            .expect("Failed to send request");

        match res.json::<PerplexityResponse>().await {
                Ok(response) => response.choices.get(0)
                    .map_or("No response found".to_string(), |choice| choice.message.content.clone()),
                Err(_) => "Failed to parse response".to_string(),
            }


    }

    async fn run_stream(&self, _request: &str) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
        unimplemented!() //TODO implement for streamning 
    }
}

// OPEN Ai gpt 



// Bash 


pub struct Bash {
    pub capture_stderr: bool,
}

#[async_trait]
impl Backend for Bash {
    async fn run(&self, request: &str) -> String {
        let output = if self.capture_stderr {
            Command::new("sh")
                .arg("-c")
                .arg(request)
                .output()
                .expect("failed to execute process")
        } else {
            Command::new("sh")
                .arg("-c")
                .arg(request)
                .stderr(std::process::Stdio::null())
                .output()
                .expect("failed to execute process")
        };

        String::from_utf8_lossy(&output.stdout).to_string()
    }
    // do test 
    async fn run_stream(&self, _request: &str) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
        unimplemented!() //TODO implement for streamning 
    }
}


// use rust to run the output, 


// use openAI to run the python code

//anthropic 
//cohere 
// implement in builtin logging and evaluation 
// a plugin system for tests 
/// use HUgging face to run
/// ollama 

//candle framework 
// dataloaders 
//text splitters 




#[async_trait]
pub trait Chainable {
    async fn process(&self, template: &str, context: &mut tera::Context) -> Result<String, Box<dyn Error>>;
}

#[async_trait]
impl Chainable for Google {
    async fn process(&self, template: &str, context:  &mut tera::Context) -> Result<String, Box<dyn Error>> {
        let mut tera = Tera::default(); 
        let processed_input = tera.render_str(template, context)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Ok(self.run(&processed_input).await)
    }
}



#[async_trait]
impl Chainable for PerplexityAI {
    async fn process(&self, template: &str, context: &mut Context) -> Result<String, Box<dyn Error>> {
        let mut tera = Tera::default(); 
        let processed_input = tera.render_str(template, context)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Ok(self.run(&processed_input).await)
    }
}



#[async_trait]
impl Chainable for Bash {
    async fn process(&self, template: &str, context: &mut Context) -> Result<String, Box<dyn Error>> {
        let mut tera = Tera::default(); 
        let input = tera.render_str(template, context)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Ok(self.run(&input).await)
    }
}


pub async fn chain_backends(
    backends: Vec<(&dyn Chainable, &str)>, 
    context: &mut Context, 
) -> Result<String, Box<dyn Error>> {
    let mut last_output = String::new();

    for (backend, template) in backends {
        let processed_input = Tera::one_off(template, &context, false)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        last_output = backend.process(&processed_input, context).await?;
        context.insert("input", &last_output); 
    }

    Ok(last_output)
}


/// Test each loader 

#[async_trait]
pub trait Loader {
    async fn load(&self) -> Result<Vec<String>, Box<dyn Error>>;
}


use std::fs;

pub struct TextLoader {
    file_path: String,
}

impl TextLoader {
    pub fn new(file_path: &str) -> Self {
        TextLoader {
            file_path: file_path.to_string(),
        }
    }
}

#[async_trait]
impl Loader for TextLoader {
    async fn load(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let content = fs::read_to_string(&self.file_path)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(vec![content])
    }
}


use select::document::Document;
use select::predicate::Name;

pub struct HtmlLoader {
    file_path: String,
}

impl HtmlLoader {
    pub fn new(file_path: &str) -> Self {
        HtmlLoader {
            file_path: file_path.to_string(),
        }
    }
}
/* 
#[async_trait]
impl Loader for HtmlLoader {
    async fn load(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let content = fs::read_to_string(&self.file_path)?;
        let document = Document::from(content.as_str());

        let texts: Vec<String> = document
            .find(Name("body"))
            .flat_map(|n| n.text().lines())
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        Ok(texts)
    }
}
*/

use std::path::Path;

pub struct PdfLoader {
    file_path: String,
}

impl PdfLoader {
    pub fn new<P: AsRef<Path>>(file_path: P) -> Self {
        PdfLoader {
            file_path: file_path.as_ref().to_str().unwrap().to_string(),
        }
    }
}

#[async_trait]
impl Loader for PdfLoader {
    async fn load(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let output = Command::new("pdftotext")
            .args(&[&self.file_path, "-"])
            .output()?;

        if !output.status.success() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to extract text from PDF",
            )));
        }

        let content = String::from_utf8(output.stdout)?;
        Ok(vec![content])
    }
}

pub struct OpenAI {
    client: Client,
    api_key: String,
    model: Option<String>,
    max_tokens: Option<i32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<i32>,
}

impl OpenAI {
    pub fn new(
        api_key: String,
        model: Option<String>,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        n: Option<i32>,
    ) -> Self {
        OpenAI {
            client: Client::new(),
            api_key,
            model: Some(model.unwrap_or_else(|| "gpt-3.5-turbo".to_string())),
            max_tokens,
            temperature,
            top_p,
            n,
        }
    }

    fn build_body(&self, request: &str) -> serde_json::Value {
        json!({
            "model": self.model.as_deref().unwrap_or("gpt-3.5-turbo"),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": request
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
        })
    }
}

#[async_trait]
impl Backend for OpenAI {
    async fn run(&self, request: &str) -> String {
        let body = self.build_body(request);

        let response = self.client.post("https://api.openai.com/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .json(&body)
            .send()
            .await
            .expect("Failed to send request");

        match response.json::<serde_json::Value>().await {
            Ok(res) => res["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string(),
            Err(_) => "Failed to parse response".to_string(),
        }
    }

    async fn run_stream(&self, _request: &str) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
        unimplemented!()
    }
}

// Implement the Chainable trait for OpenAI
//test this 
#[async_trait]
impl Chainable for OpenAI {
    async fn process(&self, template: &str, context: &mut Context) -> Result<String, Box<dyn Error>> {
        let mut tera = Tera::default();
        let processed_input = tera.render_str(template, context)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let response = self.run(&processed_input).await;
        Ok(response)
    }
}

//test ollama 
//make sure ollama is installed 
use ollama_rs::{Ollama};
use ollama_rs::generation::completion::request::GenerationRequest;
//use ollama_rs::generation::completion::request::GenerationResponseStream;


pub struct OllamaBackend {
    ollama: Ollama,
}

//make sure to install the ollama model 
// ask the ollama model parameters for backend 
impl OllamaBackend {
    pub fn new(base_url: &str, port: u16) -> Self {
        let ollama = Ollama::new(base_url.to_string(), port);
        OllamaBackend { ollama }
    }
}

#[async_trait]
impl Backend for OllamaBackend {
    async fn run(&self, request: &str) -> String {
        let model = "dolphin-phi".to_string();
        let prompt = request.to_string();

        let res = self.ollama.generate(GenerationRequest::new(model, prompt)).await;

        match res {
            Ok(res) => res.response,
            Err(_) => "Failed to generate completion".to_string(),
        }
    }

    async fn run_stream(&self, _request: &str) -> Pin<Box<dyn Stream<Item = Result<String, std::io::Error>> + Send>> {
       unimplemented!()
    }
}

#[async_trait]
impl Chainable for OllamaBackend {
    async fn process(&self, template: &str, context: &mut Context) -> Result<String, Box<dyn Error>> {
        let mut tera = Tera::default(); 
        let processed_input = tera.render_str(template, context)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let model = context.get("model").and_then(|v| v.as_str()).unwrap_or("dolphin-phi");

        let res = self.ollama.generate(GenerationRequest::new(model.to_string(), processed_input)).await;

        match res {
            Ok(res) => Ok(res.response),
            Err(e) => Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn Error>),
        }
    }
}


// BACEND FOR metaphor systems 


// MEMORY 





// evals 
// simple json embedding disntance , and simple embeddings 



// SIMPLE OPENAI embeddings 



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



// Quick tests  using cosine similarity and other similarity of the embeddings 



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
}



// LLM logging 
// a simple logging system 

use std::fs::{OpenOptions, File};
use std::io::{self, Write, Read};

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    id: String,
    prompts: Vec<String>,
    metrics: HashMap<String, f64>, 
}

#[derive(Debug)]
pub struct LlmLogger {
    experiments: Vec<Experiment>,
    file_path: String, 
}


impl LlmLogger {
    pub fn new(file_path: String) -> Self {
        LlmLogger {
            experiments: Vec::new(),
            file_path,
        }
    }

    pub fn load_experiments(&mut self) -> io::Result<()> {
        let mut file = File::open(&self.file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        self.experiments = serde_json::from_str(&contents)?;
        Ok(())
    }

    pub fn save_experiments(&self) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.file_path)?;
        file.write_all(serde_json::to_string(&self.experiments)?.as_bytes())?;
        Ok(())
    }

    pub fn log_experiment(&mut self, id: String, prompt: String, metrics: HashMap<String, f64>) {
        let experiment = Experiment {
            id,
            prompts: vec![prompt],
            metrics,
        };
        self.experiments.push(experiment);
        self.save_experiments().expect("Failed to save experiments");
    }

    //updating experiemtns 

}


impl LlmLogger {

    pub fn compare_experiments(&self, id1: &str, id2: &str) {
        let exp1 = self.experiments.iter().find(|e| e.id == id1);
        let exp2 = self.experiments.iter().find(|e| e.id == id2);

        match (exp1, exp2) {
            (Some(exp1), Some(exp2)) => {
                println!("Comparing Experiment {} with {}", id1, id2);
                // implement
            }
            _ => println!("One or both experiment IDs not found"),
        }
    }
}






