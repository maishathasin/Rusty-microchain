---
title: "Logger "
description: "Guides lead a user through a specific task they want to accomplish, often with a sequence of steps."
summary: ""
date: 2023-09-07T16:04:48+02:00
lastmod: 2023-09-07T16:04:48+02:00
draft: false
menu:
  docs:
    parent: ""
    identifier: "example-6a1a6be4373e933280d78ea53de6158e"
weight: 810
toc: true
seo:
  title: "" # custom title (optional)
  description: "" # custom description (recommended)
  canonical: "" # custom canonical URL (optional)
  noindex: false # false (default) or true
---


We  provide a simple logger to log your experiments so you can easily view and compare prompts, metrics, templates etc. Each experiment is saved into one json file with a unique ID and you can use this view your runs, you can log any data you like!


#### Logging a new experiment
```
// Create a new LlmLogger instance
let mut logger = LlmLogger::new("experiments.json".to_string());

// Log a new experiment
let prompt = "What is the capital of France?".to_string();
let experiment_id = logger.log_experiment(prompt);
println!("Logged experiment with ID: {}", experiment_id);
```


#### Logging data
```
// Log data for the experiment
logger.log_data(&experiment_id, "answer", "Paris");
logger.log_data(&experiment_id, "accuracy", "0.95");
```

#### Display experiments
```
logger.display_experiments_table();
```

