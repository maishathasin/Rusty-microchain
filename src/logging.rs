
use std::fs::{OpenOptions, File};
use std::io::{self, Write, Read};
use rand::Rng;
use serde::{Serialize, Deserialize};
use prettytable::{Table, row, cell};
use prettytable::Cell;
use prettytable::Row;
use std::collections::HashMap;



#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub id: String,
    pub prompt: String,
    pub data: HashMap<String, String>,
}

#[derive(Debug)]
pub struct LlmLogger {
    pub experiments: Vec<Experiment>,
    pub file_path: String,
}

impl LlmLogger {
    pub fn new(file_path: String) -> Self {
        LlmLogger {
            experiments: Vec::new(),
            file_path,
        }
    }

    pub fn load_experiments(&mut self) -> io::Result<()> {
        match File::open(&self.file_path) {
            Ok(mut file) => {
                let mut contents = String::new();
                file.read_to_string(&mut contents)?;
                self.experiments = serde_json::from_str(&contents)?;
            }
            Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
                File::create(&self.file_path)?;
            }
            Err(e) => return Err(e),
        }
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

    pub fn log_experiment(&mut self, prompt: String) -> String {
        let id = Self::generate_random_id();
        let experiment = Experiment {
            id: id.clone(),
            prompt,
            data: HashMap::new(),
        };
        self.experiments.push(experiment);
        self.save_experiments().expect("Failed to save experiments");
        id
    }

    pub fn log_data(&mut self, id: &str, column: &str, value: &str) {
        if let Some(experiment) = self.experiments.iter_mut().find(|e| e.id == id) {
            experiment.data.insert(column.to_string(), value.to_string());
            self.save_experiments().expect("Failed to save experiments");
        } else {
            println!("Experiment ID not found");
        }
    }

    fn generate_random_id() -> String {
        let mut rng = rand::thread_rng();
        let id: u32 = rng.gen();
        id.to_string()
    }

    pub fn display_experiments_table(&self) {
        let mut table = Table::new();
    
        let mut keys: Vec<String> = self.experiments
            .iter()
            .flat_map(|exp| exp.data.keys().cloned())
            .collect();
        keys.sort();
        keys.dedup();
    
        let mut header = vec!["ID", "Prompt"];
        header.extend(keys.iter().map(|key| key.as_str())); 
        table.add_row(Row::new(header.iter().map(|&h| Cell::new(h)).collect()));
    
        for exp in &self.experiments {
            let mut row_cells = vec![
                Cell::new(&exp.id.to_string()),
                Cell::new(&exp.prompt),
            ];
            for key in &keys {
                let value = exp.data.get(key).cloned().unwrap_or_else(String::new);
                row_cells.push(Cell::new(&value));
            }
            table.add_row(Row::new(row_cells));
        }
    
        table.printstd();
    }

    //Save the table and the output in cache 
    
}