// Defines the structure and functionality of prompts.
pub struct Prompt {
    template: String,
}

impl Prompt {
    // Creates a new prompt with the given template.
    pub fn new(template: &str) -> Self {
        Prompt {
            template: template.to_string(),
        }
    }

    // Generates a prompt string, potentially incorporating dynamic elements in the future.
    pub fn generate(&self) -> String {
        self.template.clone()
    }
}
