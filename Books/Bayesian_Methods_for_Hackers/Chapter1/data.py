from os import makedirs
makedirs("data", exist_ok=True)

from urllib.request import urlretrieve

urlretrieve("https://git.io/vXTVC", "data/txtdata.csv")