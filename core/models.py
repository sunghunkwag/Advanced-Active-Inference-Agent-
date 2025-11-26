from datetime import datetime
from typing import List

class User:
    def __init__(self, user_id: int, username: str):
        self.user_id = user_id
        self.username = username
        self.followed_users: List[int] = []

    def follow(self, user_id: int):
        if user_id not in self.followed_users:
            self.followed_users.append(user_id)

    def unfollow(self, user_id: int):
        if user_id in self.followed_users:
            self.followed_users.remove(user_id)

class Post:
    def __init__(self, post_id: int, user_id: int, content: str, timestamp: datetime = None):
        self.post_id = post_id
        self.user_id = user_id
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.likes = 0
        self.comments = 0
