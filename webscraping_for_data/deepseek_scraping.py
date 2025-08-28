import pandas as pd
import time
from selenium_scraper import SeleniumScraper
from captcha_handler import CaptchaHandler
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
        
        # Initialize the CAPTCHA handler
        captcha_handler = CaptchaHandler(selenium_scraper.driver)
        
        # Check for "verify you are human" challenge
        try:
            # Wait for page to load
            time.sleep(3)
            
            # Check for reCAPTCHA
            if captcha_handler.is_recaptcha_present():
                print("reCAPTCHA detected! Attempting to solve...")
                captcha_solved = captcha_handler.solve_recaptcha_v2(timeout=30)
                if not captcha_solved:
                    print("Failed to solve reCAPTCHA. Continuing anyway...")
            
            # Check for other verification methods (like simple checkbox)
            try:
                verify_checkbox = WebDriverWait(selenium_scraper.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//input[@type='checkbox' and contains(@id, 'verify') or contains(@class, 'verify')]")))  
                verify_checkbox.click()
                print("Clicked on verification checkbox")
                time.sleep(2)
            except (TimeoutException, NoSuchElementException):
                print("No verification checkbox found")
                
            # Check for "I am not a robot" button
            try:
                robot_button = WebDriverWait(selenium_scraper.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'not a robot') or contains(@class, 'robot')]")))  
                robot_button.click()
                print("Clicked on 'I am not a robot' button")
                time.sleep(2)
            except (TimeoutException, NoSuchElementException):
                print("No 'I am not a robot' button found")
        except Exception as e:
            print(f"Error handling verification: {str(e)}")
        
        # Check if there's a login button or continue as guest option
        try:
            time.sleep(3)
            # Try to find login/continue buttons with various selectors
            login_selectors = [
                (By.XPATH, "//button[contains(text(), 'Continue') or contains(text(), 'Start Chat') or contains(text(), 'Get Started')]")
            ]
            
            for selector_type, selector_value in login_selectors:
                try:
                    continue_button = WebDriverWait(selenium_scraper.driver, 5).until(
                        EC.element_to_be_clickable((selector_type, selector_value))
                    )
                    continue_button.click()
                    print("Clicked on continue/start button")
                    time.sleep(2)
                    break
                except:
                    continue
                    
            # Take a screenshot and save page source to debug
            try:
                screenshot_path = "deepseek_screenshot.png"
                selenium_scraper.driver.save_screenshot(screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")
                
                # Save page source
                with open("deepseek_page_source.html", "w", encoding="utf-8") as f:
                    f.write(selenium_scraper.driver.page_source)
                print("Page source saved to deepseek_page_source.html")
                
                # Print some page information
                print(f"Current URL: {selenium_scraper.driver.current_url}")
                print(f"Page title: {selenium_scraper.driver.title}")
            except Exception as ss_error:
                print(f"Failed to save debug information: {str(ss_error)}")
                
        except Exception as e:
            print(f"No login/continue buttons found: {str(e)}")
            print("Continuing without login...")
    
    except Exception as e:
        print(f"Error during initial navigation: {str(e)}")
        return None
        
    # Wait for chat interface to load
    try:
        time.sleep(5)  # Wait for page to load after login
        
        # Find and interact with the chat input
        try:
            # Try different selectors for the chat input field
            input_selectors = [
                (By.ID, "chat-input"),
                (By.XPATH, "//textarea[contains(@placeholder, 'Send a message') or contains(@placeholder, 'Message DeepSeek') or contains(@class, 'chat-input')]"),
                (By.CSS_SELECTOR, ".chat-input, .message-input, .prompt-textarea"),
                (By.XPATH, "//div[contains(@class, 'input-area') or contains(@class, 'chat-input-box')]//textarea"),
                (By.XPATH, "//textarea"),
                (By.XPATH, "//div[@role='textbox' or @contenteditable='true']"),
                (By.CSS_SELECTOR, "[data-testid='chat-input']"),
                (By.CSS_SELECTOR, "[aria-label='Chat input']"),
                (By.XPATH, "//footer//textarea")
            ]
            
            input_element = None
            for selector_type, selector_value in input_selectors:
                try:
                    input_element = WebDriverWait(selenium_scraper.driver, 5).until(
                        EC.presence_of_element_located((selector_type, selector_value))
                    )
                    if input_element and input_element.is_displayed():
                        break
                except:
                    continue
            
            if input_element:
                # Clear any existing text
                try:
                    input_element.clear()
                except:
                    print("Could not clear input element, continuing anyway")
                
                # Type the search term
                try:
                    input_element.send_keys(search_term)
                    print(f"Sent query: {search_term}")
                except Exception as e:
                    print(f"Could not send keys to input element: {str(e)}")
                    # Try JavaScript injection as fallback
                    try:
                        js_script = f"arguments[0].value = '{search_term.replace("'", "\\'")}';"
                        selenium_scraper.driver.execute_script(js_script, input_element)
                        print("Used JavaScript to set input value")
                    except Exception as js_error:
                        print(f"JavaScript injection failed: {str(js_error)}")
                
                # Press Enter or click send button
                send_methods = [
                    # Method 1: Send RETURN key
                    lambda: input_element.send_keys(Keys.RETURN),
                    
                    # Method 2: Find and click send button
                    lambda: selenium_scraper.driver.find_element(
                        By.XPATH, "//button[contains(@class, 'send') or contains(@aria-label, 'Send') or contains(@class, 'submit')]").click(),
                    
                    # Method 3: Try to submit the form
                    lambda: input_element.submit(),
                    
                    # Method 4: Use JavaScript to trigger Enter key
                    lambda: selenium_scraper.driver.execute_script(
                        "arguments[0].dispatchEvent(new KeyboardEvent('keydown', {'key': 'Enter', 'code': 'Enter', 'keyCode': 13, 'which': 13, 'bubbles': true}));", 
                        input_element),
                        
                    # Method 5: Use JavaScript to trigger form submission
                    lambda: selenium_scraper.driver.execute_script(
                        "arguments[0].closest('form').submit();", 
                        input_element)
                ]
                
                message_sent = False
                for i, send_method in enumerate(send_methods):
                    try:
                        send_method()
                        print(f"Successfully sent message using method {i+1}")
                        message_sent = True
                        break
                    except Exception as e:
                        print(f"Method {i+1} failed: {str(e)}")
                
                if not message_sent:
                    print("All methods to send message failed")
                
                print(f"Waiting {wait_time} seconds for response...")
                time.sleep(wait_time)
                
                # Try to find the response element
                response_selectors = [
                    (By.CSS_SELECTOR, ".markdown-body"),
                    (By.XPATH, "//div[contains(@class, 'message') and contains(@class, 'assistant')]"),
                    (By.CSS_SELECTOR, ".assistant-message, .bot-message, .ai-response"),
                    (By.XPATH, "//div[contains(@class, 'chat-message-item') and not(contains(@class, 'user'))]//div[contains(@class, 'content')]"),
                    (By.XPATH, "//div[contains(@class, 'assistant') or contains(@class, 'bot')]//div[contains(@class, 'content')]"),
                    (By.XPATH, "//div[@data-message-author-role='assistant']"),
                    (By.CSS_SELECTOR, "[data-testid='message-content-assistant']"),
                    (By.XPATH, "//div[contains(@class, 'message-content') and not(contains(@class, 'user'))]"),
                    (By.XPATH, "//div[contains(@class, 'prose') or contains(@class, 'response')]"),
                    (By.XPATH, "//div[contains(@class, 'message')]//div[not(contains(@class, 'user'))]//p")
                ]
                
                for selector_type, selector_value in response_selectors:
                    try:
                        response_elements = selenium_scraper.driver.find_elements(selector_type, selector_value)
                        if response_elements:
                            # Get the last (most recent) response
                            response_text = response_elements[-1].text
                            print("Response received!")
                            break
                    except:
                        continue
                
                if not response_text:
                    print("Could not find response element")
            else:
                print("Could not find chat input element")
        except Exception as e:
            print(f"Error interacting with chat: {str(e)}")
    except Exception as e:
        print(f"Error after login: {str(e)}")
    finally:
        # Always close the browser to prevent resource leaks
        if hasattr(selenium_scraper, 'driver'):
            selenium_scraper.driver.quit()
    
    return response_text


def batch_scrape(queries, wait_time=10, headless=False, delay_between_queries=5):
    """
    Scrape DeepSeek chat responses for multiple queries
    
    Args:
        queries (list): List of queries to send to DeepSeek
        wait_time (int): Time to wait for each response in seconds
        headless (bool): Whether to run the browser in headless mode
        delay_between_queries (int): Delay between queries to avoid rate limiting
        
    Returns:
        dict: Dictionary mapping queries to their responses
    """
    results = {}
    
    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}/{len(queries)}: {query}")
        
        # Add delay between queries (except for the first one)
        if i > 0:
            print(f"Waiting {delay_between_queries} seconds before next query...")
            time.sleep(delay_between_queries)
        
        response = deepseek_scrap(query, wait_time, headless)
        results[query] = response
        
        print(f"Query: {query}")
        print(f"Response: {response if response else 'No response received'}")
        
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape DeepSeek chat responses')
    parser.add_argument('--query', type=str, help='Single query to send to DeepSeek')
    parser.add_argument('--batch', type=str, help='Path to CSV file with queries (one per line)')
    parser.add_argument('--output', type=str, default='deepseek_responses.csv', help='Output file path for batch mode')
    parser.add_argument('--wait-time', type=int, default=10, help='Time to wait for response in seconds')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--delay', type=int, default=5, help='Delay between queries in batch mode')
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        print(f"Running single query mode: {args.query}")
        response = deepseek_scrap(args.query, args.wait_time, args.headless)
        
        print("\nQuery:")
        print(args.query)
        print("\nResponse:")
        print(response if response else "No response received")
        
        # Save to file
        with open('deepseek_response.txt', 'w', encoding='utf-8') as f:
            f.write(f"Query: {args.query}\n\nResponse: {response if response else 'No response received'}")
        print("\nResponse saved to deepseek_response.txt")
        
    elif args.batch:
        # Batch mode
        print(f"Running batch mode with queries from {args.batch}")
        try:
            # Read queries from CSV file
            queries = []
            with open(args.batch, 'r', encoding='utf-8') as f:
                for line in f:
                    query = line.strip()
                    if query:  # Skip empty lines
                        queries.append(query)
            
            if not queries:
                print("No queries found in the file.")
                exit(1)
                
            print(f"Loaded {len(queries)} queries")
            
            # Run batch scraping
            results = batch_scrape(queries, args.wait_time, args.headless, args.delay)
            
            # Save results to CSV
            df = pd.DataFrame(list(results.items()), columns=['Query', 'Response'])
            df.to_csv(args.output, index=False)
            print(f"\nResponses saved to {args.output}")
            
        except Exception as e:
            print(f"Error in batch mode: {str(e)}")
    else:
        # No arguments provided, run with default query
        default_query = "What are the latest advancements in AI?"
        print(f"No query specified. Running with default query: {default_query}")
        
        response = deepseek_scrap(default_query, args.wait_time, args.headless)
        
        print("\nQuery:")
        print(default_query)
        print("\nResponse:")
        print(response if response else "No response received")
        
        # Save to file
        with open('deepseek_response.txt', 'w', encoding='utf-8') as f:
            f.write(f"Query: {default_query}\n\nResponse: {response if response else 'No response received'}")
        print("\nResponse saved to deepseek_response.txt")