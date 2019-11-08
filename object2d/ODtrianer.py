import argparse
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import test4  # import test.py to get mAP after each epoch
from models import *
from .utils.datasets import *
from .utils.utils import init_seeds, plot_images
from tqdm import tnrange, tqdm_notebook,tqdm
from time import sleep
import time
import warnings
from .utils import torch_utils 
import random
# warnings.filterwarnings("ignore")
from IPython import display
import os
results_file = 'results.txt'
import numpy as np
import glob
import math



def train(opt=None,status={}):
    if opt==None:
        raise Exception("No config parameter is passed into trainer")
    cfg = opt.cfg
    hyp = opt.hyp
    # data =opt.model
    img_size = opt.img_size
    #TODO:tv_img_path 为test validation result iamge path

    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.model_dir  # initial training weights
    device = torch_utils.select_device(opt.device)
    if 'pw' not in opt.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

    # Initialize
    init_seeds()
    multi_scale = False

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = opt.model
    # data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    lb_lists = data_dict['lb_list']
    nc = int(data_dict['classes'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=opt.arc).to(device)

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.

    status['load_weights'] = False
    status['load_optimizer'] = False
    status['transfer'] = False

    if weights.endswith('.pt'):  # pytorch format
        # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)
        status['load_weights']  = True
        # load model
        if opt.transfer==1:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
            status['transfer'] = True
        else:
           model.load_state_dict(chkpt['model'])

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
            status['load_optimizer'] = True
            # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        # start_epoch = chkpt['epoch'] + 1
        start_epoch = 0
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights)

    if opt.transfer or opt.prebias:  # transfer learning edge (yolo) layers
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

        if opt.prebias:
            for p in optimizer.param_groups:
                # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                p['lr'] *= 100  # lr gain
                if p.get('momentum') is not None:  # for SGD but not Adam
                    p['momentum'] *= 0.9

        for p in model.parameters():
            if opt.prebias and p.numel() == nf:  # train (yolo biases)
                p.requires_grad = True
            elif opt.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                p.requires_grad = True
            else:  # freeze layer
                p.requires_grad = False

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)

    # scheduler.last_epoch = start_epoch - 1

    scheduler.last_epoch = 1 - 1

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=False,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  image_weights=None,
                                  cache_labels=True if epochs > 10 else False,
                                  cache_images=False if opt.prebias else opt.cache_images, labels=lb_lists)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=1,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Starting %s for %g epochs...' % ('prebias' if opt.prebias else 'training', epochs))
    ss = [[],[],[],[]]
    testss = [[],[],[],[],[],[],[]]


    for epoch in tqdm(range(start_epoch, epochs), desc='train in epochs'):

        status['epoch'] = epoch
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = True
        if freeze_backbone and epoch < epochs:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                   
                    p.requires_grad = False

        mloss = torch.zeros(4).to(device)  # mean losses

        status['NewSavedModel'] = ''
        saved_bestn = 0
        status['Intermediate results chpt'] = []

        inter_plot_folder = os.path.join(opt.model_dir.split('.')[0], 'intermediate_results')
        tvimg_plot_folder = os.path.join(opt.model_dir.split('.')[0], 'tv_img')
        wdir = os.path.join(opt.model_dir.split('.')[0],'trainOutPt')
        last = os.path.join(wdir, 'last.pt')
        best = os.path.join(wdir, 'best.pt')

        for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch -------------------------------------------------------------

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device)
            targets = targets.to(device)
            status['batch_num'] = i
            if multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes

            if ni %10== 0:
                train_batch_name = os.path.join(inter_plot_folder,'train_batch%g.jpg' % i)
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=train_batch_name)

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)

            #TODO :验证是否有问题
            status['batch_num'] = loss.item()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64

            # loss *= batch_size / 64

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            # pbar.set_description(s)
            status['mem'] = mem
            status['targets_num'] = len(targets)
            status['img_size'] = img_size
            status['GIoU']  = mloss[0]
            status['Objectness']= mloss[1]
            status['Classification']= mloss[2]
            status['Train loss'] = mloss[3]
            for idx in range(4):
                ss[idx].append(mloss[idx])
            realTimePlotResults(ss,tvimg_plot_folder)
            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if opt.prebias:
            print_model_biases(model)
        else:
            # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
            #TODO: 测试标准的iou_Map
            if not (opt.notest or (opt.nosave and epoch < 10)) or final_epoch:
                with torch.no_grad():
                    results, maps = test4.test(cfg,
                                              data_dict,
                                              batch_size=batch_size,
                                              img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.5,  # 0.1 for speed
                                              save_json=False,
                                               valOrTest='test')
        # ss = mem + lbox, lobj, lcls, loss + targets + img_size + mp, mr, map, mf1, 3*loss

        for idx in range(7):
            testss[idx].append(results[idx])
        realTimePlotResults(testss,tvimg_plot_folder)
        status['Precision'] = testss[0]
        status['Recall'] = testss[1]
        status['mAP'] = testss[2]
        status['F1'] = testss[3]
        status['GIou'] = testss[4]
        status['valObjectness'] = testss[5]
        status['valClassification'] = testss[6]
        # final_epoch and epoch > 0 and 'coco.data' in data
        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Update best mAP
        fitness = results[2]  # mAP
        if fitness > best_fitness:
            best_fitness = fitness
        # display.clear_output(wait=True)
        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve) or opt.prebias

        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if best_fitness == fitness:
                saved_bestn=saved_bestn+1
                status['NewSavedBestModel'] = saved_bestn
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)

            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, os.path.join(wdir,'backup{}_trianloss{}_testmAP{}.pt'.format(epoch,loss.item(),status['mAP'])))
                status['Intermediate results chpt'].append('backup%g.pt' % epoch)
            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------


    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


class ODtrianer():
    def __init__(self,status,opt):
        self.status = status
        self.opt= opt
        self._init()

    def _initParameter(self):
        self.cfg = self.opt.cfg
        self.hyp = self.opt.hyp
        # data =opt.model
        self.img_size = self.opt.img_size
        self.epochs = self.opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
        self.batch_size = self.opt.batch_size
        self.accumulate = self.opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
        self.weights = self.opt.model_dir  # initial training weights
        self.device = torch_utils.select_device(self.opt.device)
        if 'pw' not in self.opt.arc:  # remove BCELoss positive weights
            self.hyp['cls_pw'] = 1.
            self.hyp['obj_pw'] = 1.

        # Initialize
        init_seeds()
        multi_scale = False

        if multi_scale:
            img_sz_min = round(self.img_size / 32 / 1.5) + 1
            img_sz_max = round(self.img_size / 32 * 1.5) - 1
            self.img_size = img_sz_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (img_sz_min * 32, self.img_size))

        # Configure run
        self.data_dict = self.opt.model
        # data_dict = parse_data_cfg(data)
        self.train_path = self.data_dict['train']
        self.lb_lists = self.data_dict['lb_list']
        self.nc = int(self.data_dict['classes'])  # number of classes

        self.inter_plot_folder = os.path.join(self.opt.model_dir.split('.')[0], 'intermediate_results')
        self.tvimg_plot_folder = os.path.join(self.opt.model_dir.split('.')[0], 'tv_img')
        self.wdir = os.path.join(self.opt.model_dir.split('.')[0], 'trainOutPt')
        self.last = os.path.join(self.wdir, 'last.pt')
        self.best = os.path.join(self.wdir, 'best.pt')
        self.best_fitness = 0
        self.saved_bestn = 0
        self.epoch = 0

    def _init(self):
        self._initParameter()
        self.init_model()
        self.initOptim()
        self._transfer()
        self._initLr('LambdaLR')

    def init_model(self):
        self.model = Darknet(self.cfg, arc=self.opt.arc).to(self.device)


    def initOptim(self):
        pg0, pg1 = [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if 'Conv2d.weight' in k:
                pg1 += [v]  # parameter group 1 (apply weight_decay)
            else:
                pg0 += [v]  # parameter group 0

        if self.opt.adam:
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        del pg0, pg1

    def _transfer(self):
        if self.weights.endswith('.pt'):  # pytorch format
            # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(self.weights, map_location=self.device)
            self.status['load_weights'] = True
            # load model
            if self.opt.transfer == 1:
                chkpt['model'] = {k: v for k, v in chkpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
                self.model.load_state_dict(chkpt['model'], strict=False)
                self.status['transfer'] = True
            else:
                self.model.load_state_dict(chkpt['model'])

            # load optimizer
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_fitness = chkpt['best_fitness']
                self.status['load_optimizer'] = True

                # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            self.start_epoch = chkpt['epoch'] + 1
            # start_epoch = 0
            del chkpt

        elif len(self.weights) > 0:  # darknet format
            # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            self.cutoff = load_darknet_weights(self.model, self.weights)

        if self.opt.transfer or self.opt.prebias:  # transfer learning edge (yolo) layers
            nf = int(self.model.module_defs[self.model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

            if self.opt.prebias:
                for p in self.optimizer.param_groups:
                    # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                    p['lr'] *= 100  # lr gain
                    if p.get('momentum') is not None:  # for SGD but not Adam
                        p['momentum'] *= 0.9

            for p in self.model.parameters():
                if self.opt.prebias and p.numel() == nf:  # train (yolo biases)
                    p.requires_grad = True
                elif self.opt.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                    p.requires_grad = True
                else:  # freeze layer
                    p.requires_grad = False

    def _fit_one(self, save=True):

        mloss = torch.zeros(4).to(self.device)
        for i, (imgs, targets, paths, _) in enumerate(self.dataloader):
            # batch -------------------------------------------------------------
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            self.status['batch_num'] = i

            # Plot images with bounding boxes

            if i % 10 == 0:
                train_batch_name = os.path.join(self.inter_plot_folder, 'train_batch%g.jpg' % i)
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=train_batch_name)

            # Run model
            pred = self.model(imgs)
            # Compute loss
            loss, loss_items = compute_loss(pred, targets, self.model)

            # TODO :验证是否有问题
            self.status['batch_loss'] = loss.item()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                raise Exception('WARNING: non-finite loss, ending training')

            # Scale loss by nominal batch_size of 64

            # loss *= batch_size / 64

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if i % self.accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

            for idx in range(4):
                self.ss[idx].append(mloss[idx])

            if i%10==0:
                realTimePlotResults(self.ss, self.tvimg_plot_folder)

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)

            self.status['mem'] = mem
            self.status['targets_num'] = len(targets)
            self.status['img_size'] = self.img_size
            self.status['GIoU'] = mloss[0]
            self.status['Objectness'] = mloss[1]
            self.status['Classification'] = mloss[2]
            self.status['Train loss'] = mloss[3]

        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                self.chkpt = {'epoch': 0,
                         'best_fitness': 0,
                         'training_results': f.read(),
                         'model': self.model.module.state_dict() if type(
                             self.model) is nn.parallel.DistributedDataParallel else self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()}

            # Save last checkpoint
            torch.save(self.chkpt, self.last)

        return mloss

    def _initLr(self,typeOfLr):
        # Scheduler https://github.com/ultralytics/yolov3/issues/238
        self.typeOfLr = ['LambdaLR','MultiStepLR']
        assert typeOfLr in self.typeOfLr , 'Input Learing rater is out of field'

        if typeOfLr=='MultiStepLR':
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                      milestones=[round(self.opt.epochs * x) for x in [0.8, 0.9]],
                                                      gamma=0.1)

        elif typeOfLr=='LambdaLR':

            lf = lambda x: 1 - x / self.epochs  # linear ramp to zero
            lf = lambda x: 10 ** (self.hyp['lrf'] * x / self.epochs)  # exp ramp
            lf = lambda x: 1 - 10 ** (self.hyp['lrf'] * (1 - x / self.epochs))  # inverse exp ramp
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        if self.opt.transfer:
            self.scheduler.last_epoch = self.start_epoch - 1
        else:
            self.scheduler.last_epoch = 0

    def fit_n(self,start_epoch,epochs):
        self.clear_result_list()
        self.clear_test_reult_list()
        for epoch in tqdm(range(start_epoch, epochs), desc='train in epochs'):
            self.epoch = epoch
            self.status['epoch'] = epoch
            self.model.train()
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

            self.freeze()

            self.status['NewSavedModel'] = ''

            self.status['Intermediate results chpt'] = []
            self.mloss = self._fit_one()

            # realTimePlotResults(self.ss, self.tvimg_plot_folder)
            self.scheduler.step()
            results = self.test(epoch)

            self.save_(results)


    def save_(self,results):
        fitness = results[2]  # mAP
        if fitness > self.best_fitness:
            self.best_fitness = fitness

        if self.best_fitness == fitness:
            self.saved_bestn = self.saved_bestn + 1
            self.status['NewSavedBestModel'] = self.saved_bestn
            torch.save(self.chkpt, self.best)

        # Save backup every 10 epochs (optional)

        if self.epoch > 0 and self.epoch % 10 == 0:
            torch.save(self.chkpt, os.path.join(self.wdir, 'backup{}_trianloss{}_testmAP{}.pt'.format(self.epoch, self.mloss[3],
                                                                                            self.status['mAP'])))
            self.status['Intermediate results chpt'].append('backup%g.pt' % self.epoch)
        # Delete checkpoint
        del self.chkpt


    def test(self,epoch=None):
        if epoch==None:
            final_epoch = True
        else :
            final_epoch = epoch + 1 == self.epochs

        # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
        # TODO: 测试标准的iou_Map
        if not final_epoch:
            with torch.no_grad():
                results, maps = test4.test(self.cfg,
                                           self.data_dict,
                                           batch_size=self.batch_size,
                                           img_size=self.opt.img_size,
                                           model=self.model,
                                           conf_thres=0.5,  # 0.1 for speed
                                           save_json=False,
                                           valOrTest='test')
        # ss = mem + lbox, lobj, lcls, loss + targets + img_size + mp, mr, map, mf1, 3*loss

        for idx in range(7):
            self.testss[idx].append(results[idx])
        realTimePlotResults(self.testss, self.tvimg_plot_folder)
        self.status['Precision'] = self.testss[0]
        self.status['Recall'] = self.testss[1]
        self.status['mAP'] = self.testss[2]
        self.status['F1'] = self.testss[3]
        self.status['GIou'] = self.testss[4]
        self.status['valObjectness'] = self.testss[5]
        self.status['valClassification'] = self.testss[6]
        return results

    def freeze(self, freeze_epoch=3, freeze_backbone=True):
        '''
        :param epoch:
        :param epochs:
        :return:
        '''
        # freeze_backbone = True
        if freeze_backbone and self.epoch < freeze_epoch:
            for name, p in self.model.named_parameters():
                if int(name.split('.')[1]) < self.cutoff:  # if layer < 75
                    p.requires_grad = False

    def fitOne(self):
        self.clear_result_list()
        self._fit_one()

    def clear_result_list(self):
        self.ss = [[],[],[],[]]

    def clear_test_reult_list(self):
        self.testss = [[],[],[],[],[],[],[]]

    def _distributeTrain(self):
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model.yolo_layers = self.model.module.yolo_layers

    def _datasetInit(self):
        self.dataset = LoadImagesAndLabels(self.train_path,
                                      self.img_size,
                                      self.batch_size,
                                      augment=False,
                                      hyp=self.hyp,  # augmentation hyperparameters
                                      rect=False,  # rectangular training
                                      image_weights=None,
                                      cache_labels=True if self.epochs > 10 else False,
                                      cache_images=False if self.opt.prebias else self.opt.cache_images, labels=self.lb_lists)

        # Dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=1,
                                                 shuffle=not self.opt.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=self.dataset.collate_fn)

        self.model.nc = self.nc  # attach number of classes to model
        self.model.arc = self.opt.arc  # attach yolo architecture
        self.model.hyp = self.hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        model_info(self.model, report='summary')  # 'full' or 'summary'
        self.nb = len(self.dataloader)