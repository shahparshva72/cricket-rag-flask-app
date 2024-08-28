# WPL Cricket Data Query API

A Flask-based API for querying Women's Premier League (WPL) T20 cricket match data using natural language processing and pandas data analysis.

## 🚀 Features

- **Natural Language Queries**: Query WPL cricket data using natural language.
- **LLM Integration**: Utilizes LlamaIndex for query processing. Supports any LLM provider (Anthropic Claude, Google Gemini etc.).
- **Formatted Responses**: Returns responses in markdown table format for easy readability.

## 🛠️ Requirements

- **Python**: 3.x
- **Dependencies**:
  - Flask
  - pandas
  - LlamaIndex
- **API Keys**:
  - Anthropic API key: Get yours [here](https://console.anthropic.com/).
  - Gemini API key: Get yours [here](https://aistudio.google.com/app/apikey).

## 📦 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shahparshva72/cricket-rag-flask-app.git
   cd cricket-rag-flask-app
   ```
2. **Create a virtual environment and install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**:
   - `ANTHROPIC_API_KEY='your-anthropic-api-here'`
   - `GOOGLE_API_KEY='your-gemini-api-key-here'`

## 🚀 Usage

1. **Run the Flask Application**:
   ```bash
   flask --app main run
   ```
2. **Query the API**:
   - Send POST requests to the `/query` endpoint with a JSON payload.
   ```json
   {
     "query": "Your natural language query here"
   }
   ```
   - Example query: *"Who has scored the most runs in a single season in WPL?"*

## 📊 Data

The API uses a CSV file named `wpl_bbb.csv` that includes ball-by-ball data from WPL T20 cricket matches for the 2023 and 2024 seasons. You can download the latest dataset from [CricSheet](https://cricsheet.org/downloads/).

## 📚 References

This project draws inspiration from the following resources:

1. [LLMs for Advanced Question-Answering over Tabular/CSV/SQL Data](https://www.youtube.com/watch?v=L1o1VPVfbb0)
2. [Advanced Query Pipelines over Tabular Data](https://colab.research.google.com/drive/1fRkgSn2PSlXSMgLk32beldVnLMLtI1Pc)
