import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib3
import os

class StudyGuideDownloader:
    def __init__(self, base_url: str, download_dir: str = "docs", verify_ssl: bool = True):
        self.base_url = base_url
        self.download_dir = download_dir
        self.verify_ssl = verify_ssl

        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        os.makedirs(self.download_dir, exist_ok=True)

    def fetch_html(self) -> str:
        response = requests.get(self.base_url, verify=self.verify_ssl)
        response.raise_for_status()
        return response.text

    def extract_pdf_url(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        target_li = soup.find("li", class_="item-1860")
        if not target_li:
            raise Exception("Could not find <li class='item-1860'>")

        pdf_tag = target_li.find("a")
        if not pdf_tag or not pdf_tag.get("href"):
            raise Exception("PDF link not found in item-1860")

        return urljoin(self.base_url, pdf_tag["href"])

    def hash_md5(self, content: bytes) -> str:
        return hashlib.md5(content).hexdigest()

    def check_if_same(self, existing_path: str, new_content: bytes) -> bool:
        if not os.path.exists(existing_path):
            return False

        with open(existing_path, "rb") as f:
            existing_content = f.read()

        return self.hash_md5(existing_content) == self.hash_md5(new_content)

    def download_pdf(self, pdf_url: str) -> str:
        filename = pdf_url.split("/")[-1]
        full_path = os.path.join(self.download_dir, filename)

        print(f"Downloading: {pdf_url} → {full_path}")
        response = requests.get(pdf_url, verify=self.verify_ssl)
        response.raise_for_status()

        if self.check_if_same(full_path, response.content):
            print("⏩ PDF is the same as the existing one. Skipping download.")
            return None

        with open(full_path, "wb") as f:
            f.write(response.content)

        print("✅ Download complete.")
        return full_path

    def download_first_study_guide(self) -> str:
        html = self.fetch_html()
        pdf_url = self.extract_pdf_url(html)
        return self.download_pdf(pdf_url)
