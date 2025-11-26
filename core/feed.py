from typing import List
from .models import User, Post
from .ranking import RankingStrategy

class FeedGenerator:
    def __init__(self, ranking_strategy: RankingStrategy):
        self.ranking_strategy = ranking_strategy

    def generate_feed(self, user: User, all_posts: List[Post]) -> List[Post]:
        followed_posts = [
            post for post in all_posts if post.user_id in user.followed_users
        ]

        return self.ranking_strategy.rank(followed_posts)
