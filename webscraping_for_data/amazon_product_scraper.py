import time
import pandas as pd
from selenium_scraper import SeleniumScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

def scrape_amazon_products(search_term, num_pages=1):
    """
    Scrape product information from Amazon based on a search term
    
    Args:
        search_term (str): The product to search for
        num_pages (int): The number of pages to scrape
        
    Returns:
        DataFrame: The scraped product data
    """
    # Initialize the scraper with Amazon's URL
    scraper = SeleniumScraper("https://www.amazon.com")
    
    try:
        # Navigate to Amazon
        scraper.navigate_to_url()
        
        # Find and fill the search box
        search_box = scraper.find_element(By.ID, "twotabsearchtextbox")
        search_box.clear()
        search_box.send_keys(search_term)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for search results to load
        time.sleep(3)
        
        all_products = []
        
        for page in range(num_pages):
            print(f"Scraping page {page + 1} for {search_term}...")
            
            # Find all product containers
            product_containers = scraper.find_elements(
                By.XPATH, 
                "//div[contains(@class, 's-result-item') and contains(@class, 's-asin')]"
            )
            
            for product in product_containers:
                try:
                    # Extract product name
                    try:
                        name_element = product.find_element(By.XPATH, ".//span[@class='a-size-medium a-color-base a-text-normal']")
                    except NoSuchElementException:
                        try:
                            name_element = product.find_element(By.XPATH, ".//span[@class='a-size-base-plus a-color-base a-text-normal']")
                        except NoSuchElementException:
                            continue
                    
                    name = name_element.text.strip()
                    
                    # Extract price
                    try:
                        price = product.find_element(By.XPATH, ".//span[@class='a-price']/span[@class='a-offscreen']").get_attribute("textContent")
                    except NoSuchElementException:
                        price = "N/A"
                    
                    # Extract rating
                    try:
                        rating = product.find_element(By.XPATH, ".//span[@class='a-icon-alt']").get_attribute("textContent")
                    except NoSuchElementException:
                        rating = "N/A"
                    
                    # Extract number of reviews
                    try:
                        reviews = product.find_element(By.XPATH, ".//span[@class='a-size-base s-underline-text']").text.strip()
                    except NoSuchElementException:
                        reviews = "0"
                    
                    # Extract product URL
                    try:
                        url = product.find_element(By.XPATH, ".//a[@class='a-link-normal s-no-outline']").get_attribute("href")
                    except NoSuchElementException:
                        url = "N/A"
                    
                    # Extract image URL
                    try:
                        img_url = product.find_element(By.XPATH, ".//img[@class='s-image']").get_attribute("src")
                    except NoSuchElementException:
                        img_url = "N/A"
                    
                    # Add to our list
                    all_products.append({
                        "name": name,
                        "price": price,
                        "rating": rating,
                        "reviews": reviews,
                        "url": url,
                        "image_url": img_url
                    })
                    
                except Exception as e:
                    print(f"Error extracting product data: {e}")
            
            # Check if there's a next page and navigate to it
            if page < num_pages - 1:
                try:
                    next_button = scraper.find_element(By.XPATH, "//a[contains(@class, 's-pagination-next')]")
                    if next_button:
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


# Example usage
if __name__ == "__main__":
    # Define search terms
    search_terms = ["laptop", "smartphone", "headphones"]
    
    # Create an empty DataFrame to store all products
    all_products_df = pd.DataFrame()
    
    # Scrape products for each search term
    for term in search_terms:
        print(f"\nScraping products for: {term}")
        products_df = scrape_amazon_products(term, num_pages=1)
        
        # Add a column to identify the search term
        products_df["search_term"] = term
        
        # Append to the main DataFrame
        all_products_df = pd.concat([all_products_df, products_df], ignore_index=True)
    
    # Save all products to CSV
    filename = "amazon_products.csv"
    all_products_df.to_csv(filename, index=False)
    print(f"\nScraped a total of {len(all_products_df)} products and saved to {filename}")
    
    # Display some statistics
    print("\nProducts per search term:")
    print(all_products_df["search_term"].value_counts())
    
    print("\nAverage price per search term (excluding N/A):")
    # Convert price strings to numeric values, removing currency symbols
    all_products_df["numeric_price"] = all_products_df["price"].replace("N/A", None)
    all_products_df["numeric_price"] = all_products_df["numeric_price"].str.replace("$", "").str.replace(",", "").astype(float)
    
    # Calculate average price per search term
    avg_price = all_products_df.groupby("search_term")["numeric_price"].mean()
    print(avg_price)