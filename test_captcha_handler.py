import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the classes
from webscraping_for_data import SeleniumScraper, CaptchaHandler

def main():
    print("Successfully imported CaptchaHandler!")
    print("\nHere's how to use the CaptchaHandler to bypass 'I am not a robot' verification:")
    print("\n1. Create a CaptchaHandler instance with an undetected browser:")
    print("   handler = CaptchaHandler(None)")
    print("   driver = handler.setup_undetected_browser()")
    print("   handler = CaptchaHandler(driver)")
    
    print("\n2. Navigate to a website and check for CAPTCHA:")
    print("   driver.get(\"https://example.com\")")
    print("   if handler.is_recaptcha_present():")
    print("       success = handler.solve_recaptcha_v2()")
    
    print("\n3. Continue with your scraping after CAPTCHA is solved")
    
    print("\nNote: For full functionality, install these packages:")
    print("- pip install undetected-chromedriver")
    print("- pip install SpeechRecognition pydub")
    print("- Install FFmpeg for audio processing")

if __name__ == "__main__":
    main()