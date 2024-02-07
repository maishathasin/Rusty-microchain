// Extends the prompt module for chat-specific functionality.
use crate::prompt::Prompt;

pub struct ChatPrompt {
    base_prompt: Prompt,
    conversation_history: Vec<String>,
}

impl ChatPrompt {
    // Initializes a new ChatPrompt with a base prompt template.
    pub fn new(base_prompt: Prompt) -> Self {
        ChatPrompt {
            base_prompt,
            conversation_history: Vec::new(),
        }
    }

    // Adds a message to the conversation history.
    pub fn add_message(&mut self, message: &str) {
        self.conversation_history.push(message.to_string());
    }

    // Generates a chat prompt incorporating the conversation history.
    pub fn generate(&self) -> String {
        let mut full_prompt = self.base_prompt.generate();
        for message in &self.conversation_history {
            full_prompt.push_str("\n");
            full_prompt.push_str(message);
        }
        full_prompt
    }
}
