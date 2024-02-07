
use crate::backend::Backend;

pub struct MiniChain {
    backends: Vec<Box<dyn Backend + Send + Sync>>,
}


impl MiniChain {
    pub fn new() -> Self {
        MiniChain { backends: Vec::new() }
    }

    pub fn add_backend(&mut self, backend: Box<dyn Backend + Send + Sync>) {
        self.backends.push(backend);
    }


    pub async fn run(&self, request: &str) -> Vec<String> {
        let mut responses: Vec<String> = Vec::new();
        for backend in &self.backends {
            let response = backend.run(request).await;
            responses.push(response);
        }
        responses
    }
}


