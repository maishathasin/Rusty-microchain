use std::fs;
use crate::backend::Loader;
use std::error::Error;
use async_trait::async_trait;

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