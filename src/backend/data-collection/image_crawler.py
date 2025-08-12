from icrawler.builtin import GoogleImageCrawler
import os

players = [
    "Alphonso Davies",
    "Joshua Kimmich",
    "Leon Goretzka",
    "Michael Olise",
    "Thomas Mueller"
]

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw-images'))

for player in players:
    player_dir = os.path.join(OUTPUT_DIR, f"{player.lower().replace(' ', '-')}-raw-images-new")
    os.makedirs(player_dir, exist_ok=True)
    crawler = GoogleImageCrawler(storage={'root_dir': player_dir})
    crawler.crawl(
        keyword=f"{player} portrait",
        max_num=300,
        min_size=(400, 400),  # good resolution
        file_idx_offset=0
    )
