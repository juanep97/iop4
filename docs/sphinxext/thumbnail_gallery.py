"""
Sphinx plugin to run generate a gallery for notebooks

Adapted from [pymc](https://github.com/pymc-devs/pymc-examples/blob/d5e25ca283ab015c8eded9f82c3df5f68de84314/sphinxext/gallery_generator.py):
    Modified from the seaborn project, which modified the mpld3 project.
"""

import base64
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image
from glob import glob
from pathlib import Path

DOC_SRC = Path(__file__).parent.parent.resolve().absolute()
DEFAULT_IMG_LOC = DOC_SRC / "_static" / "jupyter.png"
THUMBNAIL_DIR = DOC_SRC / "_build" / "html" / "_thumbnails"

if not THUMBNAIL_DIR.exists():
    THUMBNAIL_DIR.mkdir(parents=True)

def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """Overwrites `infile` with a new file of the given size"""
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cx * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)
    return fig


class NotebookGenerator:
    """Tools for generating an example page from a file"""

    def __init__(self, filename):
        self.basename = os.path.basename(filename)
        stripped_name = os.path.splitext(self.basename)[0]
        self.image_dir = THUMBNAIL_DIR
        self.png_path = os.path.join(self.image_dir, f"{stripped_name}.png")
        try:
            with open(filename) as fid:
                self.json_source = json.load(fid)
        except FileNotFoundError:
            print(f"File {filename} not found")
        self.default_image_loc = DEFAULT_IMG_LOC

    def extract_preview_pic(self):
        """By default, just uses the last image in the notebook."""
        pic = None
        for cell in self.json_source["cells"]:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", []):
                    pic = output["data"]["image/png"]
        if pic is not None:
            return base64.b64decode(pic)
        return None

    def gen_previews(self):
        preview = self.extract_preview_pic()
        if preview is not None:
            with open(self.png_path, "wb") as buff:
                buff.write(preview)
        else:
            print(f"I can't find a thumbnail for {self.png_path}")
            shutil.copy(self.default_image_loc, self.png_path)
        create_thumbnail(self.png_path)


def main(app, e):

    nb_paths = glob("recipes/*.py")
    nb_names = [Path(nb_path).with_suffix('.ipynb').name for nb_path in nb_paths]
    ipynb_paths = [str(DOC_SRC / "_build" / "jupyter_execute" / "recipes" / nb_name) for nb_name in nb_names]

    for ipynb_path in ipynb_paths:
        nbg = NotebookGenerator(ipynb_path)
        nbg.gen_previews()


def setup(app):
    app.connect("build-finished", main)