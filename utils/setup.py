import urllib.request
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError

from tqdm import tqdm


default_models = {
    "config.json": ("https://drive.google.com/uc?export=download&id=19TW1HlvNcBtwnPdz99LYlQS-OC5bQajB", 1126),
    "Level0.pt": ("https://drive.google.com/uc?export=download&id=18eNp2SoB9njColgdCzkT9A5rxSdpI1aW", 13877330),
    "Level1.pt": ("https://drive.google.com/uc?export=download&id=1P-WAkx0_t_of-IA9UxHyStqXqMZHDZ7u", 13877860),
    "Level2.pt": ("https://drive.google.com/uc?export=download&id=1CjKWBL_9gVxeBMl78akhnRvMbLbjxEih", 13877860),
    "Level3.pt": ("https://drive.google.com/uc?export=download&id=1H_TLz3ylYHGFLyvNYwYUVUh3d6p9C0Hm", 13877860),
    "Level4.pt": ("https://drive.google.com/uc?export=download&id=1AyK3YLFnvEChDztBMVlTOGk0-nk-MesB", 13877860),
    "Level5.pt": ("https://drive.google.com/uc?export=download&id=19euMDztGDNs4qEXtJyt5iU7AY9_NtWbe", 13877860),
    "Level6.pt": ("https://drive.google.com/uc?export=download&id=14ApvG0frhP4qNwGSpywAEj0VEji4qlm8", 13877860),
    "Level7.pt": ("https://drive.google.com/uc?export=download&id=1l2dHv0a5DqpL2nAjNEcRz5YfuVQ7tNJj", 13877860),
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, target: Path, bar_pos=0):
    target.parent.mkdir(exist_ok=True, parents=True)

    desc = f"Downloading {target.name}"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False) as t:
        try:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
        except HTTPError:
            return


def check_base_models(models_dir: Path):
    # Define download tasks
    jobs = []
    for model_name, (url, size) in default_models.items():
        target_path = Path(f"{models_dir}/{model_name}")
        if target_path.exists():
            if target_path.stat().st_size != size:
                print(f"File {target_path} is not of expected size, redownloading...")
            else:
                continue

        thread = Thread(target=download, args=(url, target_path, len(jobs)))
        thread.start()
        jobs.append((thread, target_path, size))

    # Run and join threads
    for thread, target_path, size in jobs:
        thread.join()

        assert target_path.exists() and target_path.stat().st_size == size, \
            f"Download for {target_path.name} failed. You may download models manually instead.\n" \
            f"https://drive.google.com/drive/folders/1Y8mkbvzsmv7hq6oOEWef7VO83kyTRW4R?usp=drive_link"