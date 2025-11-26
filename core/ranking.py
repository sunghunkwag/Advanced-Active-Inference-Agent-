from abc import ABC, abstractmethod
from typing import List
from .models import Post

class RankingStrategy(ABC):
    @abstractmethod
    def calculate_score(self, post: Post) -> float:
        pass

    def rank(self, posts: List[Post]) -> List[Post]:
        return sorted(posts, key=lambda p: self.calculate_score(p), reverse=True)

class RecencyRankingStrategy(RankingStrategy):
    def calculate_score(self, post: Post) -> float:
        # Higher score for more recent posts
        return post.timestamp.timestamp()

class EngagementRankingStrategy(RankingStrategy):
    def calculate_score(self, post: Post) -> float:
        # Score based on likes and comments
        return post.likes + post.comments * 2 # Weight comments more
