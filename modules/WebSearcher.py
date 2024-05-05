from serpapi import GoogleSearch
from enum import Enum
import os

class WebSearchProviders(Enum):
    SERP = 'serp'
    TAVILY = 'tavily'

class UnsupportedProvider(Exception):
    """
    Exception raised when selecting an unsupported web search provider.
    """
    def __init__(self, message="Web search provider not yet supported."):
        self.message = message
        super().__init__(self.message)

class WebSearcher:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.count_organic_results = 5 # Get the first 5 results by default
        if not self.api_key:
            SERP_API_KEY = os.getenv('SERP_API_KEY')
            assert SERP_API_KEY, 'SERP_API_KEY is not set in environment variables'
            self.api_key = SERP_API_KEY

    def set_count_of_organic_results(self, count: int):
        self.count_organic_results = count

    def search_web(self, query: str, service_provider: WebSearchProviders = WebSearchProviders.SERP):
        if service_provider == WebSearchProviders.SERP:
            return self.serp_search(query)

        raise UnsupportedProvider()

    def serp_search(self, query: str, location: str = 'Philippines', num_organic_results: int = 0):
        params = {
            "q": query,
            "location": location,
            "api_key": self.api_key
        }
        search = GoogleSearch(params)
        if search:
            result = search.get_dict()

        curated_result = {}
        if num_organic_results <= 0:
            num_organic_results = self.count_organic_results

        if "search_parameters" in result:
            curated_result["search_parameters"] = result["search_parameters"]

        if "organic_results" in result:
            curated_result["organic_results"] = result["organic_results"][:num_organic_results]
        
        if "related_questions" in result:
            curated_result["related_questions"] = result["related_questions"][:num_organic_results]

        if "inline_images" in result:
            curated_result["inline_images"] = result["inline_images"][:num_organic_results]

        if "inline_videos" in result:
            curated_result["inline_videos"] = result["inline_videos"][:num_organic_results]

        return curated_result
    
