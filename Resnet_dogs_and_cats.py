from fastai.vision.learner import cnn_learner
from fastai.vision.models import resnet34
from fastai.vision.utils import untar_data
from fastai.vision.all import URLs, ImageDataLoaders, get_image_files, Resize
from fastai.metrics import error_rate

path = untar_data(URLs.PETS) / 'images'


def is_cat(x):
    return x[0].isupper()


dis = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    bs=32,
    item_tfms=Resize(224))

learn = cnn_learner(dis, resnet34, metrics=error_rate)
learn.fine_tune(1)
