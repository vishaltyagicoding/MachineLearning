# Web Scraping with CAPTCHA Handling

This package provides tools for web scraping with Selenium, including advanced CAPTCHA handling capabilities.

## Components

### 1. SeleniumScraper

A base class for Selenium-based web scraping with methods for navigation, element finding, scrolling, and data extraction.

### 2. CaptchaHandler

A specialized class for handling "I am not a robot" verifications (CAPTCHAs) during web scraping, with methods for:

- Detecting reCAPTCHA presence
- Solving reCAPTCHA v2 using audio challenges
- Setting up undetected browsers to avoid triggering CAPTCHAs

### 3. Example Scrapers

- `amazon_product_scraper.py`: Scrapes product information from Amazon
- `indeed_job_scraper.py`: Scrapes job listings from Indeed
- `captcha_example.py`: Demonstrates CAPTCHA handling with real websites

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install FFmpeg (required for audio CAPTCHA processing):
   - Windows: Download from https://ffmpeg.org/download.html
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

## Usage

### Basic Scraping with CAPTCHA Handling

```python
from webscraping_for_data import SeleniumScraper, CaptchaHandler

# Set up the CAPTCHA handler and get an undetected browser
handler = CaptchaHandler(None)  # Temporary None driver
driver = handler.setup_undetected_browser()

# Update the handler with the new driver
handler = CaptchaHandler(driver)

try:
    # Navigate to the target URL
    driver.get("https://example.com")
    
    # Check if CAPTCHA is present and solve it
    if handler.is_recaptcha_present():
        success = handler.solve_recaptcha_v2()
        if success:
            print("Successfully bypassed CAPTCHA!")
        else:
            print("Failed to bypass CAPTCHA.")
    
    # Continue with your scraping logic here...
    
finally:
    # Close the browser
    driver.quit()
```

### Using the Example Scripts

Run the example CAPTCHA handling script:

```bash
python -m webscraping_for_data.captcha_example
```

## Documentation

For more detailed information, see:

- `captcha_bypass_guide.md`: Comprehensive guide to CAPTCHA bypass techniques
- Code comments in individual files

## Ethical and Legal Considerations

1. Always check a website's Terms of Service before scraping
2. Implement delays between requests to avoid overloading servers
3. Be aware that bypassing CAPTCHAs may violate a site's terms or applicable laws
4. Only use these techniques for legitimate purposes like research or personal projects

## Dependencies

See `requirements.txt` for a complete list of dependencies.