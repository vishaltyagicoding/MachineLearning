import pandas as pd
import time
from selenium_scraper import SeleniumScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

def deepseek_scrap(search_term="Hello, how can you help me?", wait_time=10, headless=False):
    """
    Scrape DeepSeek chat response for a given search term
    
    Args:
        search_term (str): The query to send to DeepSeek
        wait_time (int): Time to wait for response in seconds
        headless (bool): Whether to run the browser in headless mode
        
    Returns:
        str: The response from DeepSeek, or None if no response found
    """
    url = "https://chat.deepseek.com/"
    selenium_scraper = SeleniumScraper(url, headless=headless)
    response_text = None
    
    try:
        selenium_scraper.navigate_to_url()
        
        # Find and interact with the chat input
        input_element = selenium_scraper.find_element(By.ID, "chat-input")
        if input_element:
            input_element.clear()
            input_element.send_keys(search_term)
            input_element.send_keys(Keys.RETURN)
            
            print(f"Sent query: {search_term}")
            print(f"Waiting {wait_time} seconds for response...")
            
            # Wait for response to appear
            time.sleep(wait_time)
            
            # Try to find the response element
            # Note: You may need to adjust the selector based on the actual page structure
            response_element = selenium_scraper.find_element(By.CSS_SELECTOR, ".markdown-body")
            if response_element:
                response_text = response_element.text
                print("Response received!")
            else:
                print("Could not find response element")
        else:
            print("Could not find chat input element")
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    finally:
        # Always close the browser to prevent resource leaks
        if hasattr(selenium_scraper, 'driver'):
            selenium_scraper.driver.quit()
    
    return response_text


def batch_scrape(queries, wait_time=10, headless=True, output_file="deepseek_responses.csv"):
    """
    Scrape DeepSeek chat responses for multiple queries and save to CSV
    
    Args:
        queries (list): List of queries to send to DeepSeek
        wait_time (int): Time to wait for response in seconds
        headless (bool): Whether to run the browser in headless mode
        output_file (str): Path to save the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing queries and responses
    """
    results = []
    
    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}/{len(queries)}: {query}")
        response = deepseek_scrap(query, wait_time, headless)
        results.append({"query": query, "response": response})
        
        # Add a delay between queries to avoid rate limiting
        if i < len(queries) - 1:
            delay = 5  # 5 seconds between queries
            print(f"Waiting {delay} seconds before next query...")
            time.sleep(delay)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nAll responses saved to {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek Chat Scraper")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--wait", type=int, default=15, help="Wait time for response in seconds")
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode example
        queries = [
            "What are the best practices for web scraping?",
            "How to handle CAPTCHA in web scraping?",
            "What are the legal considerations for web scraping?"
        ]
        df = batch_scrape(queries, wait_time=args.wait, headless=args.headless)
        
        # Display the first response
        if not df.empty and not df.iloc[0]['response'] is None:
            print("\nFirst Response Preview:")
            print("-" * 50)
            print(df.iloc[0]['response'][:500] + "..." if len(df.iloc[0]['response']) > 500 else df.iloc[0]['response'])
            print("-" * 50)
    else:
        # Single query mode
        query = "What are the best practices for web scraping?"
        response = deepseek_scrap(query, wait_time=args.wait, headless=args.headless)
        
        if response:
            print("\nDeepSeek Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Save response to file
            with open("deepseek_response.txt", "w", encoding="utf-8") as f:
                f.write(f"Query: {query}\n\n")
                f.write(response)
            print("\nResponse saved to deepseek_response.txt")
        else:
            print("\nNo response received from DeepSeek")
    
    print("\nDone!")
    
# Usage examples:
# python deepseek_scraping.py                  # Single query mode with visible browser
# python deepseek_scraping.py --headless       # Single query mode with headless browser
# python deepseek_scraping.py --batch          # Batch mode with visible browser
# python deepseek_scraping.py --batch --headless --wait 20  # Batch mode with headless browser and 20s wait time