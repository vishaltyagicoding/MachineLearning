import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class SeleniumScraper:
    def __init__(self, url, headless=False):
        """
        Initialize the Selenium web scraper
        
        Args:
            url (str): The URL to scrape
            headless (bool): Whether to run the browser in headless mode
        """
        self.url = url
        self.data = []
        
        # Set up Chrome options
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        # Add user agent to avoid detection
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Initialize the Chrome driver
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def navigate_to_url(self):
        """
        Navigate to the specified URL
        """
        self.driver.get(self.url)
        # Wait for the page to load
        time.sleep(2)
    
    def find_element(self, by, value, wait=True):
        """
        Find an element on the page
        
        Args:
            by (By): The method to locate the element
            value (str): The value to search for
            wait (bool): Whether to wait for the element to be present
            
        Returns:
            WebElement: The found element
        """
        try:
            if wait:
                return self.wait.until(EC.presence_of_element_located((by, value)))
            else:
                return self.driver.find_element(by, value)
        except (TimeoutException, NoSuchElementException):
            print(f"Element not found: {by}={value}")
            return None
    
    def find_elements(self, by, value, wait=True):
        """
        Find multiple elements on the page
        
        Args:
            by (By): The method to locate the elements
            value (str): The value to search for
            wait (bool): Whether to wait for the elements to be present
            
        Returns:
            list: The found elements
        """
        try:
            if wait:
                return self.wait.until(EC.presence_of_all_elements_located((by, value)))
            else:
                return self.driver.find_elements(by, value)
        except (TimeoutException, NoSuchElementException):
            print(f"Elements not found: {by}={value}")
            return []
    
    def scroll_down(self, amount=None):
        """
        Scroll down the page
        
        Args:
            amount (int): The amount to scroll down. If None, scrolls to the bottom.
        """
        if amount:
            self.driver.execute_script(f"window.scrollBy(0, {amount});")
        else:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
    
    def extract_data(self, elements, attribute=None, text=True):
        """
        Extract data from elements
        
        Args:
            elements (list): The elements to extract data from
            attribute (str): The attribute to extract. If None, extracts text.
            text (bool): Whether to extract text or attribute
            
        Returns:
            list: The extracted data
        """
        data = []
        for element in elements:
            try:
                if text:
                    data.append(element.text.strip())
                elif attribute:
                    data.append(element.get_attribute(attribute))
            except Exception as e:
                print(f"Error extracting data: {e}")
                data.append(None)
        return data
    
    def save_to_csv(self, data, filename, columns=None):
        """
        Save data to a CSV file
        
        Args:
            data (list or dict): The data to save
            filename (str): The filename to save to
            columns (list): The column names
        """
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    def close(self):
        """
        Close the browser
        """
        if self.driver:
            self.driver.quit()


# Example usage: Scraping product information from an e-commerce website
def scrape_products(url, num_pages=1):
    """
    Scrape product information from an e-commerce website
    
    Args:
        url (str): The URL to scrape
        num_pages (int): The number of pages to scrape
        
    Returns:
        DataFrame: The scraped product data
    """
    scraper = SeleniumScraper(url)
    
    try:
        scraper.navigate_to_url()
        
        all_products = []
        
        for page in range(num_pages):
            print(f"Scraping page {page + 1}...")
            
            # Find product elements
            product_elements = scraper.find_elements(By.CSS_SELECTOR, ".product-item")
            
            for product in product_elements:
                try:
                    # Extract product information
                    name = product.find_element(By.CSS_SELECTOR, ".product-name").text.strip()
                    price = product.find_element(By.CSS_SELECTOR, ".product-price").text.strip()
                    
                    # Try to get rating if available
                    try:
                        rating = product.find_element(By.CSS_SELECTOR, ".product-rating").text.strip()
                    except NoSuchElementException:
                        rating = "N/A"
                    
                    # Try to get image URL if available
                    try:
                        image_url = product.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                    except NoSuchElementException:
                        image_url = "N/A"
                    
                    all_products.append({
                        "name": name,
                        "price": price,
                        "rating": rating,
                        "image_url": image_url
                    })
                except Exception as e:
                    print(f"Error extracting product data: {e}")
            
            # Check if there's a next page button and click it
            if page < num_pages - 1:
                try:
                    next_button = scraper.find_element(By.CSS_SELECTOR, ".pagination .next")
                    if next_button and next_button.is_enabled():
                        next_button.click()
                        time.sleep(3)  # Wait for the next page to load
                    else:
                        print("No more pages available")
                        break
                except Exception as e:
                    print(f"Error navigating to next page: {e}")
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_products)
        return df
    
    finally:
        scraper.close()


# Example usage: Scraping job listings
def scrape_job_listings(url, job_title, location=None, num_pages=1):
    """
    Scrape job listings from a job search website
    
    Args:
        url (str): The URL of the job search website
        job_title (str): The job title to search for
        location (str): The location to search in
        num_pages (int): The number of pages to scrape
        
    Returns:
        DataFrame: The scraped job data
    """
    scraper = SeleniumScraper(url)
    
    try:
        scraper.navigate_to_url()
        
        # Find and fill the search form
        job_input = scraper.find_element(By.ID, "keywords-input")
        job_input.send_keys(job_title)
        
        if location:
            location_input = scraper.find_element(By.ID, "location-input")
            location_input.clear()
            location_input.send_keys(location)
        
        # Submit the search form
        job_input.send_keys(Keys.RETURN)
        time.sleep(3)  # Wait for results to load
        
        all_jobs = []
        
        for page in range(num_pages):
            print(f"Scraping page {page + 1}...")
            
            # Find job listing elements
            job_elements = scraper.find_elements(By.CSS_SELECTOR, ".job-card")
            
            for job in job_elements:
                try:
                    # Extract job information
                    title = job.find_element(By.CSS_SELECTOR, ".job-title").text.strip()
                    company = job.find_element(By.CSS_SELECTOR, ".company-name").text.strip()
                    
                    # Try to get location if available
                    try:
                        job_location = job.find_element(By.CSS_SELECTOR, ".job-location").text.strip()
                    except NoSuchElementException:
                        job_location = "N/A"
                    
                    # Try to get salary if available
                    try:
                        salary = job.find_element(By.CSS_SELECTOR, ".job-salary").text.strip()
                    except NoSuchElementException:
                        salary = "N/A"
                    
                    # Try to get job URL
                    try:
                        job_url = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    except NoSuchElementException:
                        job_url = "N/A"
                    
                    all_jobs.append({
                        "title": title,
                        "company": company,
                        "location": job_location,
                        "salary": salary,
                        "url": job_url
                    })
                except Exception as e:
                    print(f"Error extracting job data: {e}")
            
            # Check if there's a next page button and click it
            if page < num_pages - 1:
                try:
                    next_button = scraper.find_element(By.CSS_SELECTOR, ".pagination .next")
                    if next_button and next_button.is_enabled():
                        next_button.click()
                        time.sleep(3)  # Wait for the next page to load
                    else:
                        print("No more pages available")
                        break
                except Exception as e:
                    print(f"Error navigating to next page: {e}")
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_jobs)
        return df
    
    finally:
        scraper.close()


# Example: Run the job scraper
if __name__ == "__main__":
    # Example 1: Scrape products from an e-commerce site
    # products_df = scrape_products("https://example-ecommerce.com", num_pages=2)
    # products_df.to_csv("products.csv", index=False)
    
    # Example 2: Scrape job listings
    # jobs_df = scrape_job_listings("https://example-jobs.com", "Data Scientist", "New York", num_pages=2)
    # jobs_df.to_csv("jobs.csv", index=False)
    
    # Example 3: Simple scraping using the class directly
    scraper = SeleniumScraper("https://quotes.toscrape.com")
    
    try:
        scraper.navigate_to_url()
        
        # Find all quote elements
        quote_elements = scraper.find_elements(By.CSS_SELECTOR, ".quote")
        
        quotes_data = []
        
        for quote_element in quote_elements:
            # Extract quote text
            text = quote_element.find_element(By.CSS_SELECTOR, ".text").text.strip()
            # Extract author
            author = quote_element.find_element(By.CSS_SELECTOR, ".author").text.strip()
            # Extract tags
            tags = [tag.text for tag in quote_element.find_elements(By.CSS_SELECTOR, ".tag")]
            
            quotes_data.append({
                "text": text,
                "author": author,
                "tags": ", ".join(tags)
            })
        
        # Save to CSV
        quotes_df = pd.DataFrame(quotes_data)
        quotes_df.to_csv("quotes.csv", index=False)
        print(f"Scraped {len(quotes_data)} quotes and saved to quotes.csv")
        
    finally:
        scraper.close()