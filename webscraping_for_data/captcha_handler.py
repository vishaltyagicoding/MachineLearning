import time
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Optional imports for audio CAPTCHA solving
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    AUDIO_CAPTCHA_AVAILABLE = True
except ImportError:
    AUDIO_CAPTCHA_AVAILABLE = False
    print("Warning: speech_recognition and/or pydub modules not found.")
    print("Audio CAPTCHA solving will not be available.")
    print("Install with: pip install SpeechRecognition pydub")


class CaptchaHandler:
    """
    A class to handle different types of CAPTCHAs in Selenium web scraping
    """
    
    def __init__(self, driver):
        """
        Initialize the CAPTCHA handler
        
        Args:
            driver (WebDriver): The Selenium WebDriver instance
        """
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 10)
    
    def is_recaptcha_present(self):
        """
        Check if reCAPTCHA is present on the page
        
        Returns:
            bool: True if reCAPTCHA is present, False otherwise
        """
        try:
            iframe = self.driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
            return True
        except NoSuchElementException:
            return False
    
    def solve_recaptcha_v2(self, timeout=30):
        """
        Attempt to solve reCAPTCHA v2 using audio challenge
        
        Args:
            timeout (int): Maximum time to wait for CAPTCHA to be solved
            
        Returns:
            bool: True if CAPTCHA was solved, False otherwise
        """
        try:
            print("Detecting reCAPTCHA...")
            
            # Switch to the reCAPTCHA iframe
            recaptcha_iframe = self.wait.until(EC.frame_to_be_available_and_switch_to_it(
                (By.XPATH, '//iframe[@title="reCAPTCHA"]')
            ))
            
            # Click on the checkbox
            checkbox = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//div[@class="recaptcha-checkbox-border"]')
            ))
            checkbox.click()
            
            # Switch back to the main content
            self.driver.switch_to.default_content()
            
            # Check if CAPTCHA is already solved (sometimes it passes with just a click)
            time.sleep(2)
            if self._is_captcha_solved():
                print("CAPTCHA solved with checkbox click!")
                return True
            
            # If audio CAPTCHA solving is available, try that approach
            if AUDIO_CAPTCHA_AVAILABLE:
                # If not solved, try audio challenge
                print("Attempting audio challenge...")
                
                # Switch to the CAPTCHA challenge iframe
                challenge_iframe = self.wait.until(EC.frame_to_be_available_and_switch_to_it(
                    (By.XPATH, '//iframe[contains(@src, "api2/bframe")]')
                ))
                
                # Click on audio challenge button
                audio_button = self.wait.until(EC.element_to_be_clickable(
                    (By.ID, "recaptcha-audio-button")
                ))
                audio_button.click()
                
                # Get the audio source
                time.sleep(2)
                audio_src = self.driver.find_element(By.ID, "audio-source").get_attribute("src")
                
                # Download and convert the audio file
                audio_file_path = self._download_audio(audio_src)
                
                if audio_file_path:
                    # Transcribe the audio
                    transcribed_text = self._transcribe_audio(audio_file_path)
                    
                    if transcribed_text:
                        # Enter the transcribed text
                        audio_response = self.driver.find_element(By.ID, "audio-response")
                        audio_response.send_keys(transcribed_text)
                        
                        # Click verify
                        verify_button = self.driver.find_element(By.ID, "recaptcha-verify-button")
                        verify_button.click()
                        
                        # Switch back to the main content
                        self.driver.switch_to.default_content()
                        
                        # Check if CAPTCHA is solved
                        time.sleep(2)
                        if self._is_captcha_solved():
                            print("CAPTCHA solved with audio challenge!")
                            return True
            else:
                print("Audio CAPTCHA solving is not available. Install required packages.")
                print("Skipping audio challenge and proceeding to manual intervention.")
            
            # If we get here, CAPTCHA wasn't solved automatically
            print("Failed to solve CAPTCHA automatically.")
            
            # Ask for manual intervention
            print("\nMANUAL INTERVENTION REQUIRED")
            print("Please solve the CAPTCHA manually in the browser window.")
            print(f"Waiting up to {timeout} seconds for manual completion...")
            
            # Wait for manual solving
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._is_captcha_solved():
                    print("CAPTCHA solved manually!")
                    return True
                time.sleep(1)
            
            print("Timeout waiting for manual CAPTCHA solution.")
            return False
            
        except Exception as e:
            print(f"Error solving reCAPTCHA: {e}")
            return False
    
    def _download_audio(self, url):
        """
        Download audio file from URL
        
        Args:
            url (str): URL of the audio file
            
        Returns:
            str: Path to the downloaded audio file, or None if download failed
        """
        if not AUDIO_CAPTCHA_AVAILABLE:
            print("Audio CAPTCHA solving is not available. Install required packages.")
            return None
            
        try:
            response = requests.get(url)
            if response.status_code == 200:
                mp3_path = os.path.join(os.getcwd(), "captcha_audio.mp3")
                with open(mp3_path, "wb") as f:
                    f.write(response.content)
                
                # Convert to WAV format for speech recognition
                wav_path = os.path.join(os.getcwd(), "captcha_audio.wav")
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(wav_path, format="wav")
                
                return wav_path
            else:
                print(f"Failed to download audio: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None
    
    def _transcribe_audio(self, audio_file_path):
        """
        Transcribe audio file to text using Google Speech Recognition
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text, or None if transcription failed
        """
        if not AUDIO_CAPTCHA_AVAILABLE:
            print("Audio CAPTCHA solving is not available. Install required packages.")
            return None
            
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
    
    def _is_captcha_solved(self):
        """
        Check if CAPTCHA has been solved
        
        Returns:
            bool: True if CAPTCHA is solved, False otherwise
        """
        try:
            # This is a generic check - you may need to adapt it for specific websites
            # For example, check if a certain element appears after CAPTCHA is solved
            return not self.is_recaptcha_present()
        except Exception:
            return False
    
    def setup_undetected_browser(self):
        """
        Set up an undetected browser to avoid CAPTCHA triggers
        
        Returns:
            WebDriver: The undetected WebDriver instance
        """
        # Try to use undetected_chromedriver if available
        try:
            # Import here to avoid dependency issues if not installed
            import undetected_chromedriver as uc
            print("Using undetected_chromedriver for better CAPTCHA avoidance")
            
            options = uc.ChromeOptions()
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            # Add a realistic user agent
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/91.0.4472.124 Safari/537.36")
            
            driver = uc.Chrome(options=options)
            return driver
        except ImportError:
            print("undetected_chromedriver not found. Install with: pip install undetected-chromedriver")
            print("Falling back to regular Chrome driver with anti-detection measures")
        except Exception as e:
            print(f"Error setting up undetected browser: {e}")
            print("Falling back to regular Chrome driver with anti-detection measures")
        
        # Fall back to regular Chrome with anti-detection measures
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            
            # Add a realistic user agent
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/91.0.4472.124 Safari/537.36")
            
            driver = webdriver.Chrome(options=options)
            
            # Execute CDP commands to prevent detection
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
                """
            })
            
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Trying to use basic Chrome driver without anti-detection measures")
            
            # Last resort: basic Chrome driver
            options = webdriver.ChromeOptions()
            driver = webdriver.Chrome(options=options)
            return driver


# Example usage
def example_usage():
    # Set up an undetected browser
    handler = CaptchaHandler(None)  # Temporary None driver
    driver = handler.setup_undetected_browser()
    
    # Update the handler with the new driver
    handler = CaptchaHandler(driver)
    
    try:
        # Navigate to a site with CAPTCHA
        driver.get("https://www.google.com/recaptcha/api2/demo")
        
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


if __name__ == "__main__":
    # Run the example
    example_usage()