# Next Word Prediction

This Flask web application utilizes the GPT-2 language model to generate text predictions and sentences based on user input. The app provides an easy-to-use interface for interacting with the GPT-2 model via HTTP requests.

## Overview

The project consists of a Flask web application that serves as an interface to a GPT-2 language model. Users can input text prompts, and the model will generate predictions for the next words or complete sentences based on the provided input.

## Features

- **Predict Next Words**: Users can input a text prompt, and the app will generate predictions for the next words or tokens in the sequence.
- **Predict Next Sentence**: Users can input a text prompt, and the app will generate a complete sentence based on the provided input.
- **HTML Interface**: The web application provides a user-friendly HTML interface for interacting with the model.
- **REST API**: The application exposes REST API endpoints for programmatic access to the model's predictions.

## Getting Started

To run the web application locally, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using pip.
4. Run the Flask app.
5. Access the web application in your web browser at `http://localhost:5000`.

## Usage

### Predict Next Words

To predict the next words or tokens given an input text, send a POST request to the `/api/v1/predict` endpoint with JSON data containing the input text. You can specify the number of predictions to generate (default is 3).

### Predict Next Sentence

To predict the next complete sentence given an input text, send a POST request to the `/api/v1/predict/sentence` endpoint with JSON data containing the input text.

## Dependencies

- Flask
- Flask-CORS
- transformers
- torch
