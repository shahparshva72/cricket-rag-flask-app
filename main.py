from flask import Flask, request, jsonify
import pandas as pd
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core import PromptTemplate
from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.gemini  # import Gemini or any other LLM Provider If you want to use Gemini instead of Anthropic
import os

app = Flask(__name__)

os.environ["ANTHROPIC_API_KEY"] = os.environ.get('ANTHROPIC_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

# Load the data
# Ball by ball data from cricsheet.org
df = pd.read_csv("wpl_bbb.csv")

# Define the prompts and components
instruction_str = """
The dataset is from the Women's Premier League (WPL) T20 cricket matches for the 2023 and 2024 seasons. 
It is in a ball-by-ball format with the file named wpl_bbb.csv.
----------------------------

The first row contains headers, and each subsequent row provides details on a single delivery.
The headers in the file are:

  * match_id: Unique identifier for each match.
  * season: The season in which the match was played.
  * start_date: The date when the match started.
  * venue: The location where the match was played.
  * innings: The innings number (1 or 2).
  * ball: The delivery number within the over (e.g., 0.1 for the first ball, 0.2 for the second).
  * batting_team: The team currently batting.
  * bowling_team: The team currently bowling.
  * striker: The batter facing the current delivery.
  * non_striker: The batter at the non-striker's end.
  * bowler: The player delivering the ball.
  * runs_off_bat: Runs scored off the bat (excluding extras).
  * extras: Additional runs given by the bowling side (e.g., wides, no-balls, byes, leg-byes).
  * wides: A type of extra run given when the bowler delivers a ball out of the batter's reach.
  * noballs: A type of extra run given when the bowler oversteps the crease or delivers an illegal ball.
  * byes: Runs scored when the ball passes the batter without hitting the bat and no fielder catches it.
  * legbyes: Runs scored when the ball hits the batter's body without hitting the bat.
  * penalty: Extra runs awarded to the batting team as a penalty against the bowling team.
  * wicket_type: The method by which a batter is dismissed (e.g., bowled, caught, LBW).
  * player_dismissed: The name of the player who was dismissed.
  * other_wicket_type: Any other type of wicket not covered by the primary wicket_type.
  * other_player_dismissed: Another player dismissed in case of multiple dismissals on a ball (e.g., a run-out).

Key Cricketing Terms:
- **Half-century (50)**: A score of 50 or more runs by a batter in a single innings.
- **Century (100)**: A score of 100 or more runs by a batter in a single innings.
- **Strike Rate**: A measure of how quickly a batter scores, calculated as (runs scored/balls faced) * 100.
- **Economy Rate**: A measure of how many runs a bowler concedes per over, calculated as (runs conceded/overs bowled).
- **Powerplay**: The initial overs of the innings where fielding restrictions apply, generally the first 6 overs.
- **Death Overs**: The final overs of the innings, typically the last 4-5 overs, where teams often score quickly.
- **Maiden Over**: An over in which no runs are scored off the bat.
- **Dot Ball**: A delivery that does not result in any runs being scored.
- **Over**: A set of 6 consecutive deliveries bowled by the same bowler.
- **Dismissal**: The event of a batter being out.
- **Partnership**: The number of runs scored by two batters together before a wicket falls.

Instructions:
1. Understand that this data pertains to T20 matches, where key performance indicators like strike rate, economy rate, and performance in powerplay and death overs are significant.
2. If the query does not specify a season, aggregate the data across all seasons available in the dataset. The season contains value of 2022/23 which means the year was 2023 and 2023/24 means year was 2024 and so on. 
3. If the query does not specify a team, aggregate the data across all teams.
4. Convert the query to executable Python code using Pandas.
5. The code should return a DataFrame containing the top 5 results for the query.
6. Include relevant columns such as player names, teams, seasons, and the queried statistic.
7. Sort the results in descending order of the main statistic being queried.
8. The final line of code should be a Python expression that can be called with the `eval()` function.
9. PRINT ONLY THE EXPRESSION.
10. Do not quote the expression.

Example Query: "Who has scored the most runs in a single season?"
Example Expression: df.groupby(['season', 'striker'])['runs_off_bat'].sum().reset_index().sort_values('runs_off_bat', ascending=False).head(5)[['season', 'striker', 'runs_off_bat']]
"""

pandas_prompt_str = """
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression:
"""

response_synthesis_prompt_str = """
Given an input question, synthesize a response from the query results in a tabular format.
Query: {query_str}

Pandas Instructions:
{pandas_instructions}

Pandas Output: {pandas_output}

Instructions for response:
1. Format the Pandas output as a markdown table.
2. Include a brief title or description above the table.
3. Ensure the table has clear column headers.
4. Include up to 5 rows of data in the table.
5. If applicable, add a brief note below the table about what the data represents.

Response (Markdown table format):
"""

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

llm = Anthropic(model="claude-3-5-sonnet-20240620")
# Build Query Pipeline
qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
    ]
)
qp.add_link("response_synthesis_prompt", "llm2")

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_str = data.get('query')

    if not query_str:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = qp.run(query_str=query_str)
        # The response now contains a markdown table
        markdown_table = response.message.content
        return jsonify({"response": markdown_table})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(debug=True)
