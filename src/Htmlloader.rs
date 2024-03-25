
use select::document::Document;
use select::predicate::Name;
use crate::backend::Loader;
use std::error::Error;
use async_trait::async_trait;
use std::fs;
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

#[async_trait]
impl Loader for HtmlLoader {
    async fn load(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let content = fs::read_to_string(&self.file_path)?;
        Ok(vec![content])  //entire HTML content as a single string 
    }
}