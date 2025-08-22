import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webscraping_for_data.captcha_handler import CaptchaHandler


def scrape_with_captcha_handling(url, scraping_function):
    """
    Perform web scraping with CAPTCHA handling
    
    Args:
        url (str): The URL to scrape
        scraping_function (callable): Function that performs the actual scraping
            This function should accept a WebDriver instance as its argument
    
    Returns:
        The result from the scraping_function or None if failed
    """
    # Set up the CAPTCHA handler and get an undetected browser
    handler = CaptchaHandler(None)  # Temporary None driver
    driver = handler.setup_undetected_browser()
    
    # Update the handler with the new driver
    handler = CaptchaHandler(driver)
    
    try:
        # Navigate to the target URL
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Add a small delay to let the page load
        time.sleep(3)
        
        # Check if CAPTCHA is present and solve it
        if handler.is_recaptcha_present():
            print("CAPTCHA detected, attempting to solve...")
            success = handler.solve_recaptcha_v2()
            if success:
                print("Successfully bypassed CAPTCHA!")
            else:
                print("Failed to bypass CAPTCHA automatically.")
                print("Continuing anyway, as manual intervention may have occurred.")
        
        # Execute the scraping function
        print("Performing scraping operation...")
        result = scraping_function(driver)
        
        return result
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        return None
    
    finally:
        # Close the browser
        print("Closing browser...")
        driver.quit()


# Example scraping function for a product page
def scrape_product_details(driver):
    """
    Example function to scrape product details
    
    Args:
        driver (WebDriver): The Selenium WebDriver instance
    
    Returns:
        dict: Product details
    """
    # Wait for product elements to load
    wait = WebDriverWait(driver, 10)
    
    try:
        # This is just an example - adjust selectors for your target website
        product_name = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, ".product-title")
        )).text
        
        price = driver.find_element(By.CSS_SELECTOR, ".product-price").text
        
        # Get description
        description = driver.find_element(By.CSS_SELECTOR, ".product-description").text
        
        # Get image URL
        image_url = driver.find_element(By.CSS_SELECTOR, ".product-image img").get_attribute("src")
        
        return {
            "name": product_name,
            "price": price,
            "description": description,
            "image_url": image_url
        }
    
    except Exception as e:
        print(f"Error scraping product details: {e}")
        # Take a screenshot for debugging
        driver.save_screenshot("scraping_error.png")
        print("Screenshot saved as 'scraping_error.png'")
        return None


# Example for scraping Google reCAPTCHA demo page
def scrape_recaptcha_demo(driver):
    """
    Example function to scrape the Google reCAPTCHA demo page
    
    Args:
        driver (WebDriver): The Selenium WebDriver instance
    
    Returns:
        bool: True if verification was successful
    """
    try:
        # Wait for the verification success message
        wait = WebDriverWait(driver, 10)
        success_message = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, ".recaptcha-success")
        ))
        
        return True
    except:
        # If we can't find the success message, try to submit the form
        try:
            submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_button.click()
            
            # Wait again for success message
            time.sleep(2)
            success_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Verification Success')]")
            
            return len(success_elements) > 0
        except:
            return False


if __name__ == "__main__":
    # Example 1: Google reCAPTCHA demo
    print("\n=== Example 1: Google reCAPTCHA Demo ===")
    result = scrape_with_captcha_handling(
        "https://www.google.com/recaptcha/api2/demo",
        scrape_recaptcha_demo
    )
    print(f"CAPTCHA verification successful: {result}")
    
    # Example 2: Amazon product (may or may not have CAPTCHA)
    print("\n=== Example 2: Amazon Product Page ===")
    product = scrape_with_captcha_handling(
        "https://www.amazon.com/dp/B08N5KWB9H",  # Example product URL
        scrape_product_details
    )
    
    if product:
        print("\nProduct Details:")
        for key, value in product.items():
            print(f"{key}: {value}")
    else:
        print("Failed to scrape product details.")