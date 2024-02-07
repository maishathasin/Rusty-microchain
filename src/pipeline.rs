// Orchestrates sequences of operations involving LLMs.
use crate::{decoder::Decoder, prompt::Prompt};

pub struct PipelineStep {
    pub llm: Box<dyn LLM>,
    pub decoder: Box<dyn Decoder>,
}

pub struct Pipeline {
    steps: Vec<PipelineStep>,
}

impl Pipeline {
    // Initializes a new, empty Pipeline.
    pub fn new() -> Self {
        Pipeline { steps: Vec::new() }
    }

    // Adds a step to the pipeline.
    pub fn add_step(&mut self, llm: Box<dyn LLM>, decoder: Box<dyn Decoder>) {
        self.steps.push(PipelineStep { llm, decoder });
    }

    // Executes the pipeline with the given initial prompt, passing the output of each step as the prompt for the next.
    pub fn execute(&self, initial_prompt: &Prompt) -> String {
        let mut current_output = initial_prompt.generate();
        for step in &self.steps {
            let response = step.llm.query(&Prompt::new(&current_output));
            current_output = step.decoder.decode(&response);
        }
        current_output
    }
}
