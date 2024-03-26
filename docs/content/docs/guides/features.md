---
title: "Features"
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

Rust microchain has all the features needed to create an LLM application from the ground up! 

### Overview


#### Search 

* Google Search (Serpson API): Enables integration with Google search, providing comprehensive search capabilities within your application.
* Perplexity: Offers tools to assess the complexity and understandability of text, which is crucial for optimizing content.
* Exa (Metaphor Systems): Advanced search functionalities that enhance the ability to find and interpret complex data patterns.


#### Functions 
* Bash: Incorporates shell scripting capabilities, allowing for versatile script execution and automation within the microchain environment.


#### Templates 
We are using Tera templates, for prompt templates for our chains 

#### Chains
We have implemented chainable, a fucntion that will help you chain vaious LLMS, functions , search together for your application

#### Loaders 
Loaders can be used to load verious type of inputs from files, these are the current loaders implemented:
* Pdf loaders 
* HTML loaders 
* text loaders

#### Large Language models
We have access to a various LLMs by implementin there apis, these solutions can be used to interact with chains and other models etc. 


* OpenAI
* Ollama 
* Anthropic (coming soon)


#### Embeddings 
Embeddings can be used to create numerical representations of words , they can be used to create similairties between sentences, text classifications etc. These are currently the models we support 
* OpenAI
* Ollama 
* Anthropic (coming soon)


#### Vector store 
Vector stores can be used to store vectors. Currently we support these stores 
* Qdrant 


#### Evaluation 
Currently for evaluation we can use the embeddings above to do a simple similarity using cosine 
* Computing cosine similarity using the above embeddings 

#### Logging 
Our framework also provides a simple Logger to log prompts, templates, and any other metric you like !
* simple Logger and to log your prompts, templates and any metrics  

