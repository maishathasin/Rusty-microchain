
use std::path::Path;

use crate::backend::Loader;
use std::error::Error;
use async_trait::async_trait;
use std::process::Command;
use std::fs;

pub struct PdfLoader {
    file_path: String,
}

impl PdfLoader {
    pub fn new(file_path: &str) -> Self {
        PdfLoader {
            file_path: file_path.to_string(),
        }
    }
}

#[async_trait]
impl Loader for PdfLoader {
    async fn load(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let bytes = fs::read(&self.file_path).map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let out = pdf_extract::extract_text_from_mem(&bytes).map_err(|e| Box::new(e) as Box<dyn Error>)?;

        Ok(vec![out])
    }
}