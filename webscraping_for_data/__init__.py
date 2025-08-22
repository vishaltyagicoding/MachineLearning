# Import classes for easy access
from .selenium_scraper import SeleniumScraper
from .captcha_handler import CaptchaHandler


import time
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webscraping_for_data.captcha_handler import CaptchaHandler

from webscraping_for_data import CaptchaHandler

# Set up an undetected browser
handler = CaptchaHandler(None)
driver = handler.setup_undetected_browser()
handler = CaptchaHandler(driver)

# Navigate to a site with CAPTCHA
driver.get("https://www.indeed.com")

# Check if CAPTCHA is present and solve it
if handler.is_recaptcha_present():
    success = handler.solve_recaptcha_v2()
    if success:
        print("Successfully bypassed CAPTCHA!")

# Continue with your scraping logic...

def scrape_indeed_jobs(job_title, location, num_pages=1):
    """
    Scrape job listings from Indeed based on job title and location
    
    Args:
        job_title (str): The job title to search for
        location (str): The location to search in
        num_pages (int): The number of pages to scrape
        
    Returns:
        DataFrame: The scraped job data
    """
    # Initialize the scraper with Indeed's URL
    scraper = SeleniumScraper("https://www.indeed.com")
    
    try:
        # Navigate to Indeed
        scraper.navigate_to_url()
        
        # Find and fill the job title input
        job_input = scraper.find_element(By.ID, "text-input-what")
        job_input.clear()
        job_input.send_keys(job_title)
        
        # Find and fill the location input
        location_input = scraper.find_element(By.ID, "text-input-where")
        location_input.clear()
        location_input.send_keys(location)
        
        # Submit the search form
        job_input.send_keys(Keys.RETURN)
        
        # Wait for search results to load
        time.sleep(3)
        
        all_jobs = []
        
        for page in range(num_pages):
            print(f"Scraping page {page + 1} for {job_title} in {location}...")
            
            # Find all job cards
            job_cards = scraper.find_elements(By.CSS_SELECTOR, ".job_seen_beacon")
            
            if not job_cards:
                print("No job cards found. The selector might have changed.")
                break
            
            for job in job_cards:
                try:
                    # Extract job title
                    try:
                        title_element = job.find_element(By.CSS_SELECTOR, "h2.jobTitle")
                        title = title_element.text.strip()
                    except NoSuchElementException:
                        continue  # Skip if we can't find the title
                    
                    # Extract company name
                    try:
                        company = job.find_element(By.CSS_SELECTOR, "span.companyName").text.strip()
                    except NoSuchElementException:
                        company = "N/A"
                    
                    # Extract location
                    try:
                        job_location = job.find_element(By.CSS_SELECTOR, "div.companyLocation").text.strip()
                    except NoSuchElementException:
                        job_location = "N/A"
                    
                    # Extract salary if available
                    try:
                        salary = job.find_element(By.CSS_SELECTOR, "div.salary-snippet-container").text.strip()
                    except NoSuchElementException:
                        try:
                            salary = job.find_element(By.CSS_SELECTOR, "div.estimated-salary").text.strip()
                        except NoSuchElementException:
                            salary = "N/A"
                    
                    # Extract job description snippet
                    try:
                        description = job.find_element(By.CSS_SELECTOR, "div.job-snippet").text.strip()
                    except NoSuchElementException:
                        description = "N/A"
                    
                    # Extract posting date
                    try:
                        date = job.find_element(By.CSS_SELECTOR, "span.date").text.strip()
                    except NoSuchElementException:
                        date = "N/A"
                    
                    # Extract job URL
                    try:
                        url = title_element.find_element(By.XPATH, "./..")
                        job_url = url.get_attribute("href")
                    except (NoSuchElementException, AttributeError):
                        job_url = "N/A"
                    
                    # Add to our list
                    all_jobs.append({
                        "title": title,
                        "company": company,
                        "location": job_location,
                        "salary": salary,
                        "description": description,
                        "date": date,
                        "url": job_url
                    })
                    
                except Exception as e:
                    print(f"Error extracting job data: {e}")
            
            # Check if there's a next page and navigate to it
            if page < num_pages - 1:
                try:
                    # Look for the "Next" button
                    next_button = scraper.find_element(By.CSS_SELECTOR, "a[data-testid='pagination-page-next']", wait=False)
                    if next_button:
                        next_button.click()
                        time.sleep(3)  # Wait for the next page to load
                    else:
                        print("No more pages available")
                        break
                except (NoSuchElementException, TimeoutException) as e:
                    print(f"Error navigating to next page: {e}")
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_jobs)
        return df
    
    finally:
        scraper.close()


# Example usage
if __name__ == "__main__":
    # Define job search parameters
    job_searches = [
        {"title": "Data Scientist", "location": "New York, NY"},
        {"title": "Software Engineer", "location": "San Francisco, CA"},
        {"title": "Machine Learning Engineer", "location": "Remote"}
    ]
    
    # Create an empty DataFrame to store all jobs
    all_jobs_df = pd.DataFrame()
    
    # Scrape jobs for each search
    for search in job_searches:
        print(f"\nScraping jobs for: {search['title']} in {search['location']}")
        jobs_df = scrape_indeed_jobs(search['title'], search['location'], num_pages=1)
        
        # Add columns to identify the search
        jobs_df["search_title"] = search['title']
        jobs_df["search_location"] = search['location']
        
        # Append to the main DataFrame
        all_jobs_df = pd.concat([all_jobs_df, jobs_df], ignore_index=True)
    
    # Save all jobs to CSV
    filename = "indeed_jobs.csv"
    all_jobs_df.to_csv(filename, index=False)
    print(f"\nScraped a total of {len(all_jobs_df)} jobs and saved to {filename}")
    
    # Display some statistics
    print("\nJobs per search:")
    print(all_jobs_df.groupby(["search_title", "search_location"]).size())
    
    print("\nTop companies hiring:")
    print(all_jobs_df["company"].value_counts().head(10))
    
    # Check for remote jobs
    remote_jobs = all_jobs_df[all_jobs_df["location"].str.contains("Remote", case=False, na=False)]
    print(f"\nNumber of remote jobs found: {len(remote_jobs)}")
