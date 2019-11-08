import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from IPython import display
# cfg / yolov3 - xray.cfg
from tqdm import tnrange, tqdm_notebook

def test(cfg='cfg/yolov3-xray.cfg',
         data =None,
         weights='/data1/wanglonglong/01workspace/yolov3_orig/yolov3-xray-chest/weights/last.pt',
         batch_size=8,
         img_size=512,
         iou_thres=0.5,
         conf_thres=0.01,
         nms_thres=0.5,
         save_json=False,
         model = None,
         test_result_path='./results',
         valOrTest='val'
         ):
    # Initialize/load model and set device
    # opt = opt
    if model is None:
        device = torch_utils.select_device('0')
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    # data = data
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    lb_lists = data['lb_list']
    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size,labels = lb_lists)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()


    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')

    a_s = np.linspace(0.4, 0.75, 8)
    aps = []

    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, ap, ap_class = [], [], []
    stats= [[],[],[],[],[],[],[],[]]
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):#迭代每一张图像
        #将图像输入到争取的device中
        # display.clear_output(wait=True)
        targets = targets.to(device)
        # print(targets)
        imgs = imgs.to(device)
        bs, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model
        # inf_out输出了检测的结果，trian_out用于输出计算loss
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
        # 可视化预测结果----------------开始
        if batch_i%10==0:#没10个batch输出validation结果
            ns = np.ceil(bs ** 0.5).astype(int)
            fig, _axs = plt.subplots(nrows=2, ncols=4,figsize=(10, 10))
            axs = _axs.flatten()
            for idx, boxes in enumerate(output):
                # 输出的结构是xyxy objconf clsconf cls_pred
                axs[idx].imshow((imgs[idx]).permute(1, 2, 0).cpu())
                axs[idx].set_title('the {}th img'.format(idx))
                for bbox in boxes:
                    x1 = bbox[0]
                    y1 = bbox[1]
                    w = (bbox[0]+bbox[2])/2
                    h = (bbox[1] + bbox[3]) / 2
                    bboxp = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='green', facecolor="none")
                # Add the bbox to the plot
                    axs[idx].add_patch(bboxp)
            fig.tight_layout()
            # plt.show()
            if not os.path.exists(test_result_path):
                os.makedirs(test_result_path)
            if valOrTest=="test":
                fname = os.path.join(test_result_path,'testPredImg.jpg')
            else:
                fname = os.path.join(test_result_path, 'valPredImg.jpg')
            fig.savefig(fname, dpi=200)
            # display.clear_output(wait=True)
        # 可视化预测结果----------------结束

        # Statistics per image shape = 【batch,num,8】8 = [lable,bbox,obj_conf,class_conf]
        for idx,iou_threshold in enumerate(a_s):
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                # 如果没有目标就在统计列表中：加入一个空的列表和空的张量，并将分类lables加入到其中，
                if pred is None:
                    if nl:
                        stats[idx].append(([], torch.Tensor(), torch.Tensor(), tcls))
                    continue

                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = [0] * len(pred)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= width
                    tbox[:, [1, 3]] *= height

                    # Search for correct predictions
                    # 对于每一个预测
                    # 都要找到一个目标框

                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                        # Break if all targets already located in image
                        if len(detected) == nl:
                            break

                        # Continue if predicted class not among image classes
                        if pcls.item() not in tcls:
                            continue

                        # Best iou, index between pred and targets
                        # 得到同一类的下标
                        m = (pcls == tcls_tensor).nonzero().view(-1)
                        # 与同一类的tbbox进行iou的计算，得到当前predbbox匹配的tbox的下标和iou
                        iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                        # If iou > threshold and class is correct mark as correct
                        # 如果iou大于设定阈值，同时没有被匹配过，就将该coorect置1，标记当前perd已经匹配了
                        if iou > iou_threshold and m[bi] not in detected:  # and pcls == tcls[bi]:
                            correct[i] = 1
                            detected.append(m[bi])

                # Append statistics (correct, conf, pcls, tcls)
                stats[idx].append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

######################
    # Compute statistics
    l=[]
    for i in range(len(a_s)):
        l.append(list(zip(*stats[i])))

    # 先将每一个列变成一个列表，然后使用numpy变换成一个列向量，最终输出的是四个列向量
    stats=[]

    for i in range(len(a_s)):
        stats.append([np.concatenate(x, 0) for x in l[i]]) # to numpy
    # correct中为1的预测值 以及tcls是一一对应的
    for i in range(len(a_s)):
        if len(stats[i]):
            p, r, ap, f1, ap_class = ap_per_class1(*stats[i])
            print(ap)
            if a_s[i]==0.5:
                mp, mr, mf1, ap_class,ap= p.mean(), r.mean(),  f1.mean(),ap_class, ap
            map = ap.mean()
            aps.append(ap[0])
            nt = np.bincount(stats[i][3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    aps = np.array(aps)
    # print(aps)
    map = aps.mean()
    print(map)
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-xray.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='./data/pneumonia.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='/data1/wanglonglong/01workspace/yolov3_orig/yolov3-xray-chest/weights/backup30.pt', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)
