# DeepSeek Scraping with CAPTCHA Bypass

This script allows you to scrape responses from DeepSeek chat by automatically handling "verify you are human" challenges and CAPTCHA verification.

## Features

- Automatic CAPTCHA detection and solving (including reCAPTCHA v2)
- Handles various verification methods (checkboxes, "I am not a robot" buttons)
- Supports both single query and batch query modes
- Configurable wait times and headless browser operation
- Saves responses to files (text for single queries, CSV for batch queries)

## Requirements

Install the required dependencies:

```bash
pip install selenium pandas
```

For audio CAPTCHA solving (optional but recommended):

```bash
pip install SpeechRecognition pydub
```

For undetected browser (optional but recommended):

```bash
pip install undetected-chromedriver
```

## Usage

### Single Query Mode

```bash
python deepseek_scraping.py --query "What are the latest advancements in AI?"
```

### Batch Query Mode

Create a text file with one query per line, then run:

```bash
python deepseek_scraping.py --batch queries.txt --output responses.csv
```

### Additional Options

- `--wait-time`: Time to wait for response in seconds (default: 10)
- `--headless`: Run browser in headless mode
- `--delay`: Delay between queries in batch mode (default: 5)

Example with all options:

```bash
python deepseek_scraping.py --query "Tell me about quantum computing" --wait-time 15 --headless
```

## How It Works

1. The script launches a browser and navigates to DeepSeek chat
2. It automatically detects and attempts to solve any CAPTCHA or verification challenges
3. After verification, it enters the query and waits for a response
4. The response is extracted and saved to a file

## Troubleshooting

- If CAPTCHA solving fails, try running in non-headless mode (remove `--headless` flag)
- Increase `--wait-time` if responses are not being captured
- For batch mode, increase `--delay` if you encounter rate limiting

## Note

This script is for educational purposes only. Please respect DeepSeek's terms of service and rate limits when using this tool.