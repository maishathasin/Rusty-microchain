// Defines various decoders for processing LLM outputs.
pub trait Decoder {
    fn decode(&self, input: &str) -> String;
}

pub struct JSONDecoder;

impl Decoder for JSONDecoder {
    fn decode(&self, input: &str) -> String {
        format!("Decoded JSON: {}", input)
    }
}

pub struct RegExDecoder {
    pattern: String,
}

impl RegExDecoder {
    pub fn new(pattern: &str) -> Self {
        RegExDecoder {
            pattern: pattern.to_string(),
        }
    }
}

impl Decoder for RegExDecoder {
    fn decode(&self, input: &str) -> String {
        format!("Extracted with pattern {}: {}", self.pattern, input)
    }
}
