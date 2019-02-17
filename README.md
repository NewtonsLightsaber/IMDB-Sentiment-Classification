# IMDB Sentiment Analysis

*Part of Project 2 of COMP 551 Applied Machine Learning - McGill University*

## Prerequisites

### Running Google Colab locally

1. Install and enable the jupyter_http_over_ws jupyter extension (one-time)
    ```
    pip install jupyter_http_over_ws
    jupyter serverextension enable --py jupyter_http_over_ws
    ```

2. Start server and authenticate
    ```
    jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
    ```
