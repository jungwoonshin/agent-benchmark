"""General API requester for multiple external APIs."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseAPIRequester(ABC):
    """Base class for API requesters."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.session = requests.Session()
        self.session.headers.update(self._get_default_headers())

    @abstractmethod
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        pass

    @abstractmethod
    def _get_base_url(self) -> str:
        """Get base URL for the API."""
        pass

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {'User-Agent': 'Agent-System/1.0'}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request with error handling."""
        url = f'{self.base_url}{endpoint}'
        request_headers = {**self.session.headers}
        if headers:
            request_headers.update(headers)

        try:
            if method.upper() == 'GET':
                response = self.session.get(
                    url, params=params, headers=request_headers, timeout=timeout
                )
            elif method.upper() == 'POST':
                response = self.session.post(
                    url,
                    params=params,
                    json=json_data,
                    headers=request_headers,
                    timeout=timeout,
                )
            else:
                raise ValueError(f'Unsupported HTTP method: {method}')

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.warning(f'API request failed for {url}: {e}')
            return None


class GitHubAPIRequester(BaseAPIRequester):
    """GitHub API requester for repositories, issues, commits."""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('GITHUB_API_KEY') or os.getenv('GITHUB_TOKEN')

    def _get_base_url(self) -> str:
        return 'https://api.github.com'

    def _get_default_headers(self) -> Dict[str, str]:
        headers = super()._get_default_headers()
        if self.api_key:
            headers['Authorization'] = f'token {self.api_key}'
        headers['Accept'] = 'application/vnd.github.v3+json'
        return headers

    def search_issues(
        self,
        repo: Optional[str] = None,
        state: str = 'all',
        labels: Optional[List[str]] = None,
        sort: str = 'created',
        order: str = 'asc',
        per_page: int = 100,
        q: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search issues in a repository.

        Args:
            repo: Repository in format "owner/repo" (required if q is not provided)
            state: Issue state (open, closed, all)
            labels: List of label names
            sort: Sort field (created, updated, comments)
            order: Sort order (asc, desc)
            per_page: Results per page (max 100)
            q: Raw GitHub search query string (if provided, repo/state/labels are ignored)

        Returns:
            List of issue dictionaries
        """
        if q:
            # Use raw query if provided
            query = q
        else:
            # Construct query from structured parameters
            if not repo:
                raise ValueError("Either 'repo' or 'q' parameter must be provided")
            query_parts = [f'repo:{repo}', f'state:{state}']
            if labels:
                for label in labels:
                    query_parts.append(f'label:{label}')
            query = ' '.join(query_parts)

        params = {
            'q': query,
            'sort': sort,
            'order': order,
            'per_page': min(per_page, 100),
        }

        result = self._make_request('GET', '/search/issues', params=params)
        if result:
            return result.get('items', [])
        return []

    def get_issue_events(
        self,
        repo: str,
        issue_number: int,
    ) -> List[Dict[str, Any]]:
        """Get events for a specific issue (including label additions)."""
        endpoint = f'/repos/{repo}/issues/{issue_number}/events'
        result = self._make_request('GET', endpoint)
        if result:
            return result if isinstance(result, list) else []
        return []

    def get_issue(
        self,
        repo: str,
        issue_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific issue."""
        endpoint = f'/repos/{repo}/issues/{issue_number}'
        return self._make_request('GET', endpoint)

    def get_repository_commit(
        self,
        repo: str,
        ref: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific commit from repository."""
        endpoint = f'/repos/{repo}/commits/{ref}'
        return self._make_request('GET', endpoint)

    def get_repository_contents(
        self,
        repo: str,
        path: str = '',
        ref: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get repository contents at a path."""
        endpoint = f'/repos/{repo}/contents/{path}'
        params = {}
        if ref:
            params['ref'] = ref
        return self._make_request('GET', endpoint, params=params)


class WikipediaAPIRequester:
    """Wikipedia API requester using mwclient."""

    def __init__(self, logger: logging.Logger, language: str = 'en'):
        self.logger = logger
        self.language = language
        try:
            import mwclient

            self.site = mwclient.Site(f'{language}.wikipedia.org')
        except ImportError:
            self.site = None
            logger.warning(
                'mwclient not available. Install with: uv pip install mwclient'
            )

    def get_page(
        self,
        title: str,
        revision_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get Wikipedia page content."""
        if not self.site:
            return None

        try:
            page = self.site.pages[title]

            # Check if page exists
            if not page.exists:
                self.logger.warning(f'Wikipedia page "{title}" does not exist')
                return None

            # Handle redirects - follow redirect to get actual content
            if page.redirect:
                redirect_target = page.redirects_to()
                if redirect_target:
                    self.logger.info(
                        f'Page "{title}" is a redirect to "{redirect_target.name}"'
                    )
                    page = redirect_target
                    if not page.exists:
                        self.logger.warning(
                            f'Redirect target "{redirect_target.name}" does not exist'
                        )
                        return None

            # Get page content
            page_text = page.text()
            if page_text is None:
                self.logger.warning(f'Wikipedia page "{title}" returned None content')
                return None

            if revision_id:
                # Get specific revision with content
                rev = page.revisions(
                    prop='ids|timestamp|content|user', limit=1, startid=revision_id
                )
                rev_data = next(rev, None)
                if rev_data:
                    # Get text from the revision data
                    rev_text = rev_data.get('*')  # '*' contains the revision text
                    if rev_text is None:
                        # Fallback: try to get current page text
                        self.logger.warning(
                            f'Revision {revision_id} for page "{title}" did not include content, using current page text'
                        )
                        rev_text = page_text

                    return {
                        'title': page.name,
                        'revision_id': revision_id,
                        'content': rev_text,
                        'timestamp': rev_data.get('timestamp'),
                    }
                else:
                    self.logger.warning(
                        f'Revision {revision_id} not found for page "{title}"'
                    )
                    return None
            else:
                return {
                    'title': page.name,
                    'content': page_text,
                    'revision_id': page.revision,
                }
        except Exception as e:
            self.logger.warning(
                f'Failed to get Wikipedia page {title}: {e}', exc_info=True
            )
            return None

    def search_pages(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search Wikipedia pages."""
        if not self.site:
            return []

        try:
            results = []
            for page in self.site.search(query, limit=limit):
                results.append(
                    {
                        'title': page.name,
                        'url': f'https://{self.language}.wikipedia.org/wiki/{page.name.replace(" ", "_")}',
                    }
                )
            return results
        except Exception as e:
            self.logger.warning(f'Wikipedia search failed: {e}')
            return []

    def get_page_revisions(
        self,
        title: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get page revision history."""
        if not self.site:
            return []

        try:
            page = self.site.pages[title]
            revisions = []
            for rev in page.revisions(
                prop='ids|timestamp|user|comment|tags', limit=limit
            ):
                timestamp = rev.get('timestamp', '')
                # Handle both tuple and string formats from mwclient
                if isinstance(timestamp, tuple):
                    # Convert tuple (year, month, day, hour, minute, second) to string
                    rev_date = (
                        f'{timestamp[0]:04d}-{timestamp[1]:02d}-{timestamp[2]:02d}'
                    )
                elif isinstance(timestamp, str):
                    rev_date = timestamp[:10]  # YYYY-MM-DD
                else:
                    # Skip if timestamp format is unexpected
                    continue

                if start_date and rev_date < start_date:
                    break
                if end_date and rev_date > end_date:
                    continue
                revisions.append(rev)
            return revisions
        except Exception as e:
            self.logger.warning(f'Failed to get revisions for {title}: {e}')
            return []


class YouTubeAPIRequester(BaseAPIRequester):
    """YouTube Data API requester."""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('YOUTUBE_API_KEY')

    def _get_base_url(self) -> str:
        return 'https://www.googleapis.com/youtube/v3'

    def get_video_info(
        self,
        video_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get video information."""
        if not self.api_key:
            self.logger.warning('YouTube API key not set')
            return None

        params = {
            'key': self.api_key,
            'id': video_id,
            'part': 'snippet,contentDetails,statistics',
        }
        result = self._make_request('GET', '/videos', params=params)
        if result and 'items' in result:
            return result['items'][0] if result['items'] else None
        return None

    def search_videos(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for videos."""
        if not self.api_key:
            self.logger.warning('YouTube API key not set')
            return []

        params = {
            'key': self.api_key,
            'q': query,
            'part': 'snippet',
            'type': 'video',
            'maxResults': min(max_results, 50),
        }
        result = self._make_request('GET', '/search', params=params)
        if result and 'items' in result:
            return result['items']
        return []


class TwitterAPIRequester(BaseAPIRequester):
    """Twitter/X API v2 requester."""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('TWITTER_BEARER_TOKEN')

    def _get_base_url(self) -> str:
        return 'https://api.twitter.com/2'

    def _get_default_headers(self) -> Dict[str, str]:
        headers = super()._get_default_headers()
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 10,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get tweets from a user."""
        if not self.api_key:
            self.logger.warning('Twitter API key not set')
            return []

        # First get user ID
        user_result = self._make_request(
            'GET',
            f'/users/by/username/{username}',
        )
        if not user_result or 'data' not in user_result:
            return []

        user_id = user_result['data']['id']

        # Get tweets
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,text,public_metrics',
        }
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time

        result = self._make_request(
            'GET',
            f'/users/{user_id}/tweets',
            params=params,
        )
        if result and 'data' in result:
            return result['data']
        return []


class RedditAPIRequester(BaseAPIRequester):
    """Reddit API requester (uses OAuth2)."""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('REDDIT_CLIENT_ID')

    def _get_base_url(self) -> str:
        return 'https://oauth.reddit.com'

    def _authenticate(self) -> bool:
        """Authenticate with Reddit API."""
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'Agent-System/1.0')

        if not client_id or not client_secret:
            self.logger.warning('Reddit API credentials not set')
            return False

        auth_url = 'https://www.reddit.com/api/v1/access_token'
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        data = {'grant_type': 'client_credentials'}
        headers = {'User-Agent': user_agent}

        try:
            response = requests.post(auth_url, auth=auth, data=data, headers=headers)
            response.raise_for_status()
            token = response.json()['access_token']
            self.session.headers['Authorization'] = f'bearer {token}'
            self.session.headers['User-Agent'] = user_agent
            return True
        except Exception as e:
            self.logger.warning(f'Reddit authentication failed: {e}')
            return False

    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self._authenticate()

    def get_user_posts(
        self,
        username: str,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Get posts from a Reddit user."""
        endpoint = f'/user/{username}/submitted'
        params = {'limit': min(limit, 100)}
        result = self._make_request('GET', endpoint, params=params)
        if result and 'data' in result and 'children' in result['data']:
            return [item['data'] for item in result['data']['children']]
        return []

    def search_posts(
        self,
        subreddit: str,
        query: str,
        limit: int = 25,
        sort: str = 'relevance',
    ) -> List[Dict[str, Any]]:
        """Search posts in a subreddit."""
        endpoint = f'/r/{subreddit}/search'
        params = {
            'q': query,
            'limit': min(limit, 100),
            'sort': sort,
            'restrict_sr': 'true',
        }
        result = self._make_request('GET', endpoint, params=params)
        if result and 'data' in result and 'children' in result['data']:
            return [item['data'] for item in result['data']['children']]
        return []


class WaybackMachineAPIRequester:
    """Wayback Machine API requester using waybackpy."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        try:
            from waybackpy import WaybackMachineCDXServerAPI, WaybackMachineSaveAPI

            self.waybackpy_available = True
        except ImportError:
            self.waybackpy_available = False
            logger.warning(
                'waybackpy not available. Install with: uv pip install waybackpy'
            )

    def get_archived_url(
        self,
        url: str,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Get archived URL from Wayback Machine."""
        if not self.waybackpy_available:
            return None

        try:
            from waybackpy import WaybackMachineCDXServerAPI

            cdx = WaybackMachineCDXServerAPI(url, user_agent='Agent-System/1.0')
            snapshots = cdx.snapshots()

            if timestamp:
                # Find closest snapshot to timestamp
                for snapshot in snapshots:
                    if snapshot.timestamp.startswith(timestamp):
                        return snapshot.archive_url
            else:
                # Get most recent
                if snapshots:
                    return snapshots[0].archive_url

            return None
        except Exception as e:
            self.logger.warning(f'Wayback Machine request failed: {e}')
            return None


class GoogleMapsAPIRequester(BaseAPIRequester):
    """Google Maps/Street View API requester."""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('GOOGLE_MAPS_API_KEY')

    def _get_base_url(self) -> str:
        return 'https://maps.googleapis.com/maps/api'

    def get_place_details(
        self,
        place_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get place details."""
        if not self.api_key:
            self.logger.warning('Google Maps API key not set')
            return None

        params = {
            'key': self.api_key,
            'place_id': place_id,
            'fields': 'name,formatted_address,geometry,photos',
        }
        result = self._make_request('GET', '/place/details/json', params=params)
        if result and 'result' in result:
            return result['result']
        return None

    def get_street_view_image(
        self,
        location: str,
        size: str = '600x400',
        heading: Optional[int] = None,
        pitch: Optional[int] = None,
        fov: Optional[int] = None,
    ) -> Optional[str]:
        """Get Street View image URL."""
        if not self.api_key:
            self.logger.warning('Google Maps API key not set')
            return None

        params = {
            'key': self.api_key,
            'location': location,
            'size': size,
        }
        if heading:
            params['heading'] = heading
        if pitch:
            params['pitch'] = pitch
        if fov:
            params['fov'] = fov

        # Street View Static API returns image, not JSON
        url = f'{self.base_url}/streetview'
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.url  # Return the image URL
        except Exception as e:
            self.logger.warning(f'Street View request failed: {e}')

        return None


class GovernmentAPIRequester(BaseAPIRequester):
    """Base class for government APIs (USGS, Census, BLS, FRED, etc.)."""

    def _get_api_key(self) -> Optional[str]:
        # Most government APIs don't require keys, but some do
        return os.getenv('CENSUS_API_KEY') or os.getenv('FRED_API_KEY')

    def _get_base_url(self) -> str:
        # Override in subclasses
        return ''

    def get_usgs_data(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get data from USGS API."""
        url = f'https://nas.er.usgs.gov/api/v1.1/{endpoint}'
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.warning(f'USGS API request failed: {e}')
            return None

    def get_census_data(
        self,
        dataset: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get data from Census API."""
        url = f'https://api.census.gov/data/{dataset}'
        if self.api_key:
            params['key'] = self.api_key
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.warning(f'Census API request failed: {e}')
            return None


class UnifiedAPIRequester:
    """Unified API requester that routes to appropriate API."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.github = GitHubAPIRequester(logger)
        self.wikipedia = WikipediaAPIRequester(logger)
        self.youtube = YouTubeAPIRequester(logger)
        self.twitter = TwitterAPIRequester(logger)
        self.reddit = RedditAPIRequester(logger)
        self.wayback = WaybackMachineAPIRequester(logger)
        self.google_maps = GoogleMapsAPIRequester(logger)
        self.gov = GovernmentAPIRequester(logger)

    def request(
        self,
        api_name: str,
        method: str,
        **kwargs,
    ) -> Any:
        """
        Unified request method that routes to appropriate API.

        Args:
            api_name: Name of API (github, wikipedia, youtube, twitter, reddit, arxiv, wayback, google_maps, usgs, census)
            method: Method name to call
            **kwargs: Arguments for the method

        Returns:
            API response
        """
        api_map = {
            'github': self.github,
            'wikipedia': self.wikipedia,
            'youtube': self.youtube,
            'twitter': self.twitter,
            'reddit': self.reddit,
            'wayback': self.wayback,
            'google_maps': self.google_maps,
            'usgs': self.gov,
            'census': self.gov,
        }

        if api_name == 'arxiv':
            # Handle arxiv separately
            from ..utils.arxiv_utils import (
                extract_arxiv_id_from_url,
                get_arxiv_metadata,
            )

            if method == 'get_metadata':
                paper_id = kwargs.get('paper_id')
                download_pdf = kwargs.get('download_pdf', False)
                tool_belt = kwargs.get('tool_belt', None)
                return get_arxiv_metadata(
                    paper_id,
                    self.logger,
                    download_pdf=download_pdf,
                    tool_belt=tool_belt,
                )
            elif method == 'extract_id_from_url':
                url = kwargs.get('url')
                return extract_arxiv_id_from_url(url)
            return None

        api = api_map.get(api_name.lower())
        if not api:
            self.logger.warning(f'Unknown API: {api_name}')
            return None

        if not hasattr(api, method):
            self.logger.warning(f'Method {method} not found in {api_name} API')
            return None

        try:
            return getattr(api, method)(**kwargs)
        except Exception as e:
            self.logger.error(f'Error calling {api_name}.{method}: {e}', exc_info=True)
            return None
