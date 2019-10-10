import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metric import evaluate
from util import html
import pdb
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
score_list = []
for i, data in enumerate(dataset):
    #if i >= opt.how_many:
    #    break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

    if opt.model == 'cycle_gan':
        gt, pred = (model.gt_A.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255).astype(np.uint8), (model.fake_A.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        gt, pred = (model.gt_B.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255).astype(np.uint8), (model.fake_B.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255).astype(np.uint8)

    score_list.append(evaluate(gt, pred))

webpage.save()

ans = {}
for score in score_list:
    for k, v in score.items():
        if k not in ans:
            ans[k] = 0
        ans[k] += v

for k, v in ans.items():
    ans[k] /= len(score_list)


message = ''

for k, v in ans.items():
    message += '%s: %.3f\n' % (k, v)
log_name = os.path.join(web_dir, 'score.txt')
print(log_name)

with open(log_name, "a") as log_file:
    log_file.write('%s' % message)

print(message)
