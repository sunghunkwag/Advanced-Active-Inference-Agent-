import unittest
from datetime import datetime, timedelta
from core.models import User, Post
from core.ranking import RecencyRankingStrategy, EngagementRankingStrategy
from core.feed import FeedGenerator

class TestRankingStrategies(unittest.TestCase):
    def test_recency_ranking(self):
        post1 = Post(1, 1, "Post 1", datetime.now() - timedelta(hours=1))
        post2 = Post(2, 1, "Post 2", datetime.now())
        posts = [post1, post2]

        strategy = RecencyRankingStrategy()
        ranked_posts = strategy.rank(posts)

        self.assertEqual(ranked_posts[0].post_id, 2)
        self.assertEqual(ranked_posts[1].post_id, 1)

    def test_engagement_ranking(self):
        post1 = Post(1, 1, "Post 1")
        post1.likes = 10
        post1.comments = 5

        post2 = Post(2, 1, "Post 2")
        post2.likes = 20
        post2.comments = 2

        posts = [post1, post2]

        strategy = EngagementRankingStrategy()
        ranked_posts = strategy.rank(posts)

        self.assertEqual(ranked_posts[0].post_id, 2)
        self.assertEqual(ranked_posts[1].post_id, 1)

class TestFeedGenerator(unittest.TestCase):
    def test_feed_generation(self):
        user = User(1, "testuser")
        user.follow(2)

        post1 = Post(1, 2, "Followed user's post")
        post2 = Post(2, 3, "Unfollowed user's post")
        all_posts = [post1, post2]

        strategy = RecencyRankingStrategy()
        feed_generator = FeedGenerator(strategy)
        feed = feed_generator.generate_feed(user, all_posts)

        self.assertEqual(len(feed), 1)
        self.assertEqual(feed[0].post_id, 1)

if __name__ == '__main__':
    unittest.main()
