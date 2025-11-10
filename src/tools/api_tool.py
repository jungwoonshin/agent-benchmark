"""API tool for accessing external APIs."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..utils.api_requester import UnifiedAPIRequester


class APITool:
    """Tool for accessing external APIs."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_requester = UnifiedAPIRequester(logger)
    
    def detect_api_from_url(
        self,
        url: str,
        problem: Optional[str] = None,
        subtask_description: Optional[str] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Detect which API should be used based on URL and context.
        
        Args:
            url: URL to analyze
            problem: Optional problem description
            subtask_description: Optional subtask description
        
        Returns:
            Tuple of (api_name, api_params) if API should be used, None otherwise
        """
        if not url:
            return None
        
        url_lower = url.lower()
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # GitHub detection
        if "github.com" in domain:
            # Extract repo and issue number
            github_match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
            if github_match:
                owner = github_match.group(1)
                repo = github_match.group(2)
                
                # Check if it's an issue
                issue_match = re.search(r'/issues/(\d+)', url)
                if issue_match:
                    issue_number = int(issue_match.group(1))
                    return ("github", {
                        "method": "get_issue",
                        "repo": f"{owner}/{repo}",
                        "issue_number": issue_number,
                    })
                
                # Check if it's a commit
                commit_match = re.search(r'/commit/([a-f0-9]+)', url)
                if commit_match:
                    commit_sha = commit_match.group(1)
                    return ("github", {
                        "method": "get_repository_commit",
                        "repo": f"{owner}/{repo}",
                        "ref": commit_sha,
                    })
                
                # Check if it's repository contents
                if "/blob/" in url or "/tree/" in url:
                    path_match = re.search(r'/(?:blob|tree)/[^/]+/(.+)', url)
                    if path_match:
                        file_path = path_match.group(1)
                        return ("github", {
                            "method": "get_repository_contents",
                            "repo": f"{owner}/{repo}",
                            "path": file_path,
                        })
        
        # Wikipedia detection
        if "wikipedia.org" in domain:
            # Extract page title
            wiki_match = re.search(r'/wiki/([^?#]+)', url)
            if wiki_match:
                title = wiki_match.group(1).replace('_', ' ')
                # Check for revision in query params
                revision_id = None
                if parsed.query:
                    rev_match = re.search(r'oldid=(\d+)', parsed.query)
                    if rev_match:
                        revision_id = int(rev_match.group(1))
                
                return ("wikipedia", {
                    "method": "get_page",
                    "title": title,
                    "revision_id": revision_id,
                })
        
        # YouTube detection
        if "youtube.com" in domain or "youtu.be" in domain:
            video_id = None
            if "youtube.com" in domain:
                video_match = re.search(r'[?&]v=([^&]+)', url) or re.search(r'/watch/([^/?]+)', url)
                if video_match:
                    video_id = video_match.group(1)
            elif "youtu.be" in domain:
                video_match = re.search(r'youtu\.be/([^/?]+)', url)
                if video_match:
                    video_id = video_match.group(1)
            
            if video_id:
                return ("youtube", {
                    "method": "get_video_info",
                    "video_id": video_id,
                })
        
        # Twitter/X detection
        if "twitter.com" in domain or "x.com" in domain:
            # Extract username from URL
            user_match = re.search(r'/(?:twitter|x)\.com/([^/?#]+)', url)
            if user_match:
                username = user_match.group(1)
                if username and not username.startswith('i/'):  # Skip special pages
                    return ("twitter", {
                        "method": "get_user_tweets",
                        "username": username,
                        "max_results": 10,
                    })
        
        # Reddit detection
        if "reddit.com" in domain:
            # Extract subreddit or username
            reddit_match = re.search(r'reddit\.com/r/([^/]+)', url)
            if reddit_match:
                subreddit = reddit_match.group(1)
                return ("reddit", {
                    "method": "search_posts",
                    "subreddit": subreddit,
                    "query": "",  # Will need to extract from context
                    "limit": 25,
                })
            
            user_match = re.search(r'reddit\.com/user/([^/]+)', url)
            if user_match:
                username = user_match.group(1)
                return ("reddit", {
                    "method": "get_user_posts",
                    "username": username,
                    "limit": 25,
                })
        
        # arXiv detection
        if "arxiv.org" in domain:
            arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/([\d.]+)', url)
            if arxiv_match:
                paper_id = arxiv_match.group(1)
                return ("arxiv", {
                    "method": "get_metadata",
                    "paper_id": paper_id,
                })
        
        # Wayback Machine detection
        if "web.archive.org" in domain or "archive.org" in domain:
            # Extract original URL from Wayback Machine URL
            wayback_match = re.search(r'web\.archive\.org/web/(\d+)/(.+)', url)
            if wayback_match:
                timestamp = wayback_match.group(1)
                original_url = wayback_match.group(2)
                return ("wayback", {
                    "method": "get_archived_url",
                    "url": original_url,
                    "timestamp": timestamp[:8],  # YYYYMMDD
                })
        
        return None
    
    def detect_api_from_context(
        self,
        problem: str,
        subtask_description: str,
        url: Optional[str] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Detect which API should be used based on problem/subtask context.
        
        Args:
            problem: Problem description
            subtask_description: Subtask description
            url: Optional URL from search result
        
        Returns:
            Tuple of (api_name, api_params) if API should be used, None otherwise
        """
        text = f"{problem} {subtask_description}".lower()
        
        # GitHub context detection
        if "github" in text or "repository" in text or "issue" in text or "commit" in text:
            # Try to extract repo name
            repo_match = re.search(r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)', text)
            if repo_match:
                owner = repo_match.group(1)
                repo = repo_match.group(2)
                
                # Check for issue search
                if "issue" in text and ("label" in text or "closed" in text or "open" in text):
                    labels = []
                    label_match = re.search(r'label[:\s]+([^\s,]+)', text, re.IGNORECASE)
                    if label_match:
                        labels.append(label_match.group(1))
                    
                    state = "all"
                    if "closed" in text:
                        state = "closed"
                    elif "open" in text:
                        state = "open"
                    
                    sort = "created"
                    order = "asc"
                    if "oldest" in text:
                        sort = "created"
                        order = "asc"
                    elif "newest" in text:
                        sort = "created"
                        order = "desc"
                    
                    return ("github", {
                        "method": "search_issues",
                        "repo": f"{owner}/{repo}",
                        "state": state,
                        "labels": labels,
                        "sort": sort,
                        "order": order,
                    })
        
        # Wikipedia context detection
        if "wikipedia" in text or "wiki" in text:
            # Try to extract page title
            title_match = re.search(r'wikipedia.*?page.*?for\s+([^,\.]+)', text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                return ("wikipedia", {
                    "method": "get_page",
                    "title": title,
                })
            
            # Check for revision history
            if "revision" in text or "edit" in text:
                title_match = re.search(r'page.*?for\s+([^,\.]+)', text, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    return ("wikipedia", {
                        "method": "get_page_revisions",
                        "title": title,
                    })
        
        # YouTube context detection
        if "youtube" in text or "video" in text:
            # Try to extract video ID from URL if provided
            if url:
                video_match = re.search(r'[?&]v=([^&]+)', url) or re.search(r'youtu\.be/([^/?]+)', url)
                if video_match:
                    video_id = video_match.group(1)
                    return ("youtube", {
                        "method": "get_video_info",
                        "video_id": video_id,
                    })
        
        # arXiv context detection
        if "arxiv" in text:
            # Try to extract paper ID
            arxiv_match = re.search(r'arxiv[:\s]+([\d]{4}\.[\d]{5})', text, re.IGNORECASE)
            if arxiv_match:
                paper_id = arxiv_match.group(1)
                return ("arxiv", {
                    "method": "get_metadata",
                    "paper_id": paper_id,
                })
        
        # Twitter context detection
        if "twitter" in text or "tweet" in text:
            # Try to extract username
            user_match = re.search(r'@?([a-zA-Z0-9_]+).*?twitter', text, re.IGNORECASE)
            if user_match:
                username = user_match.group(1)
                return ("twitter", {
                    "method": "get_user_tweets",
                    "username": username,
                    "max_results": 10,
                })
        
        # Reddit context detection
        if "reddit" in text:
            # Try to extract subreddit
            subreddit_match = re.search(r'r/([a-zA-Z0-9_]+)', text)
            if subreddit_match:
                subreddit = subreddit_match.group(1)
                return ("reddit", {
                    "method": "search_posts",
                    "subreddit": subreddit,
                    "query": "",  # Will need to extract from context
                    "limit": 25,
                })
        
        return None
    
    def call_api(
        self,
        api_name: str,
        method: str,
        **kwargs,
    ) -> Any:
        """
        Call an external API.
        
        Args:
            api_name: Name of API (github, wikipedia, youtube, twitter, reddit, arxiv, wayback, google_maps)
            method: Method name to call
            **kwargs: Arguments for the method
        
        Returns:
            API response
        """
        self.logger.info(f"Calling {api_name}.{method} with args: {kwargs}")
        result = self.api_requester.request(api_name, method, **kwargs)
        return result
    
    def try_api_for_search_result(
        self,
        search_result_url: str,
        problem: str,
        subtask_description: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to use an API for a search result instead of web scraping.
        
        Args:
            search_result_url: URL from search result
            problem: Problem description
            subtask_description: Subtask description
        
        Returns:
            API response if API was used, None otherwise
        """
        # First try URL-based detection
        api_info = self.detect_api_from_url(search_result_url, problem, subtask_description)
        
        # If not found, try context-based detection
        if not api_info:
            api_info = self.detect_api_from_context(problem, subtask_description, search_result_url)
        
        if not api_info:
            return None
        
        api_name, api_params = api_info
        method = api_params.pop("method")
        
        try:
            result = self.call_api(api_name, method, **api_params)
            if result:
                self.logger.info(f"Successfully retrieved data from {api_name} API for {search_result_url}")
                return {
                    "api_name": api_name,
                    "api_method": method,
                    "data": result,
                    "source_url": search_result_url,
                }
        except Exception as e:
            self.logger.warning(f"API call failed for {api_name}.{method}: {e}")
        
        return None


