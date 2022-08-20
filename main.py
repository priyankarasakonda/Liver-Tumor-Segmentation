import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
#from flask_session import Session
import os
import numpy as np
import nibabel as nib
import keras
from tensorflow import keras
from tensorflow.python import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.utils import load_img, img_to_array
import constant
import sys
from tqdm.notebook import tqdm
from PIL import Image
import fastai
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'AJDJRJS24$($(#$$33--'  # <--- SECRET_KEY must be set in


@app.route('/')
def main():
    session['secrrt'] = 'sec'
    return render_template('index.html', btn_name="Upload CTScan")


@app.route("/about")
def about_page():
    return "Help your own by test own"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        f = request.files['uploads']
        f.save(secure_filename(f.filename))
        flash(secure_filename(f.filename))
        fl = open("Filepaths.txt", "r+")
        lines = fl.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            if index == 0:
                ctf = line.strip().split("=")[1]
            else:
                segvar = line.strip().split("=")[1]
        fl.close()
        if ctf == "":
            lines = "ctf=" + str(secure_filename(f.filename))
            fl = open("Filepaths.txt", "w+")
            fl.writelines(lines)
            fl.close()
            return render_template('index.html', btn_name="Upload Segmentation")
        else:
            fl = open("Filepaths.txt", "a+")
            lines = "segvar=" + secure_filename(f.filename)
            fl.newlines
            fl.writelines(lines)
            fl.close()
            test = generate_results()


# function used to read nii files and convert into a numpy array
def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return (array)


class TensorCTScan(TensorImageBW): _show_args = {'cmap': 'bone'}


@patch
def freqhist_bins(self: Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                   tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()


@patch
def hist_scaled(self: Tensor, brks=None):
    "Scales a tensor using `freqhist_bins` to values between 0 and 1"
    if self.device.type == 'cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0: res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))


@patch
def save_jpg(x: (Tensor), path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    im.save(fn, quality=quality)


@patch
def windowed(self: Tensor, w, l):
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)


def get_x(fname: Path):
    return fname


def label_func(x):
    return path / 'train_masks' / f'{x.stem}_mask.png'


def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def cust_foreground_acc(inp, targ):  # # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3,
                          axis=1)  # 3 is a dummy value to include the background which is 0


def nii_tfm(fn, wins):
    test_nii = read_nii(fn)
    curr_dim = test_nii.shape[2]  # 512, 512, curr_dim
    slices = []

    #     for curr_slice in range(curr_dim):
    #         data = tensor(test_nii[...,curr_slice].astype(np.float32))
    #         data = (data.to_nchan(wins)*255).byte()
    #         slices.append(TensorImage(data))

    #     return slices
    data = tensor(test_nii[..., 74].astype(np.float32))
    data = (data.to_nchan(wins) * 255).byte()
    slices.append(TensorImage(data))
    #     data = tensor(test_nii[...,351].astype(np.float32))
    #     data = (data.to_nchan(wins)*255).byte()
    #     slices.append(TensorImage(data))
    #print(slices)
    return slices

def generate_results():
    dicom_windows = types.SimpleNamespace(
        brain=(80, 40),
        subdural=(254, 100),
        stroke=(8, 32),
        brain_bone=(2800, 600),
        brain_soft=(375, 40),
        lungs=(1500, -600),
        mediastinum=(350, 50),
        abdomen_soft=(400, 50),
        liver=(150, 30),
        spine_soft=(250, 50),
        spine_bone=(1800, 400),
        custom=(200, 60)
    )
    GENERATE_JPG_FILES = True  # warning: generation takes ~ 1h
    slice_sum = 0
    if (GENERATE_JPG_FILES):
        fl = open("Filepaths.txt", "r+")
        lines = fl.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            if index == 0:
                ctf = line.strip().split("=")[1]
                ctname = str(ctf[:-6].split(".")[0])
                ctf = "C:/flaskproject/flaskproject/" + ctf[:-6]
                segvar = "C:/flaskproject/flaskproject/" + line.strip().split("=")[2]
        path = Path(".")

        os.makedirs('train_images', exist_ok=True)
        os.makedirs('train_masks', exist_ok=True)

        curr_ct = read_nii(ctf)
        curr_mask = read_nii(segvar)
        curr_file_name = str(ctname)
        curr_dim = curr_ct.shape[2]  # 512, 512, curr_dim
        slice_sum = slice_sum + curr_dim

        for curr_slice in range(0, curr_dim, 1):  # export every 2nd slice for training
            data = tensor(curr_ct[..., curr_slice].astype(np.float32))
            mask = Image.fromarray(curr_mask[..., curr_slice].astype('uint8'), mode="L")
            data.save_jpg(f"train_images/{curr_file_name}_slice_{curr_slice}.jpg",
                          [dicom_windows.liver, dicom_windows.custom])
            mask.save(f"train_masks/{curr_file_name}_slice_{curr_slice}_mask.png")

    else:

        path = Path("../input/liver-segmentation-with-fastai-v2")  # read jpg from saved kernel output
        print(slice_sum)

    # loading the tensor flow model
    # Load saved model
    bs = 16
    im_size = 128

    # the labels used for the classes
    # When predicting the model predicts it in terms of indices (ie 0 --> background, 1 --> liver ...)
    codes = np.array(["background", "liver", "tumor"])

    # the default pathb
    path = './'
    tfms = [Resize(im_size), IntToFloatTensor(), Normalize()]
    learn0 = load_learner("C:/flaskproject/flaskproject/Liver_segmentation", cpu=False)

    learn0.dls.transform = tfms

    # test number
    #tst = 3

    # slice number
    test_slice_idx = 74

    test_nii = read_nii(ctf)
    test_mask = read_nii(segvar)
    #print(test_nii.shape)

    #sample_slice = tensor(test_nii[..., test_slice_idx].astype(np.float32))

    # Prepare a nii test file for prediction

    test_files = nii_tfm(ctf,[dicom_windows.liver, dicom_windows.custom])
    test_dl = learn0.dls.test_dl(test_files)
    preds, y = learn0.get_preds(dl=test_dl)

    predicted_mask = np.argmax(preds, axis=1)
    print(predicted_mask[0], file=sys.stderr)
    plt.imsave(predicted_mask[0],"result_"+ctf.split(".")[0]+".jpg")

    #print("Number of test slices: ", len(test_files))


if __name__ == '__main__':
    app.run(port=12000, debug=True)
