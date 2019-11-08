'''
@Author: feynmanwill
@Date: 2019-11-06 14:42:38
@LastEditTime: 2019-11-07 11:58:49
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /dw_research_platform/DeepLearning/object2d/ODmaster.py
'''
import sys

import test4
from detect import detector

sys.path.append('../')
sys.path.append('../../')
from ..d0_AIMaster import AIMaster
from dw_sql_operator import SqlOperator
from d0_datafinder import DcmAnnoLoader
import traceback
import os
from utils import get_task_logger, pickle_load, pickle_save
from .utils.utils import outcfg
from kmeans import kmeans_out
from ODtrianer import train
class object2d(AIMaster):
    def init(self, task_queue, done_queue, running_queue, lock, worker_id, conf, logger, gpu_id):

        self.task_queue = task_queue
        self.done_queue = done_queue
        self.running_queue = running_queue
        self.lock = lock
        self.worker_id = worker_id
        self.conf = conf
        self.logger = logger

        self.dcmanno_loader = DcmAnnoLoader(logger, conf)
        self.sql = SqlOperator(logger, conf)

        self.logger.info("DW Processor init success!")

        self.model_type_dict = {4: "cls", 5: "det", 6: "seg"}
        self.res = {}
        self.gpu_id = gpu_id
        self.opter = opter()
        return True

    def main_course(self,task):
        self.task = task
        self.sql.update_to_sql(table='project', key='id', key_value=task['id'], **{'status': 15})

        self.progress =   {"stage": {"获取数据": 0, "数据整理": 0, "模型训练": 0, "模型测试": 0, "模型保存": 0}, 
                    "progress_bar": {"visible": False, 
                                    "cur_epoch": 0, "total_epoch": 0, 
                                    "iteration": "5.00%", "time_consumed": "0", 
                                    "time_remains": "0","end": ''}, 
                                    "history": ""}
        
        self.training = int(task['type']) == 0
        model_info = self.sql.get_from_sql(table='model_detail', key='id', key_value=task['modelId'])[0]
        self.opter.model_dir = model_info['modelPath']

        print('model_dir : ', self.opter.model_dir)
        
        model_type = self.model_type_dict[task['modelType']]
        
        self.opter.modelType = model_type

        assert model_type in ['cls', 'seg','det']

        self.local_logger, self.log_file = get_task_logger(task, operation_type=f'DL_{model_type}')
        self.parse(task)
        try:
            if self.training:
                self.local_logger.info('Start Training')
                ret = self.train_model(task)
            else:
                self.local_logger.info('Start Infering')
                inferenceMode = task['inferenceMode']
                ret = self.inference(inferenceMode,task)
        except:
            print(self.log_file, 'is file:', os.path.isfile(self.log_file))
            # file = open(self.log_file, 'a')
            traceback.print_exc()
            traceback.print_exc(file=open(self.log_file, 'a'))
            ret = False
        self.local_logger.info('[ %s task ] [ ID : %d ] [ Success : %s ]' % (task['task_type'], task['id'], str(ret)))
        return ret

    def parse(self, task):
        # 超参数部分
        self.opter.hyp = task['hyperparameter']

        # 数据集路径
        self.opter.ds_dir = task['trainParameter']['model']['ds_path']
        # 模型数据集输入路径
        task['trainParameter']['model']['train'] = os.path.join(self.opter.ds_dir,'labellist/train.txt')
        task['trainParameter']['model']['test'] = os.path.join(self.opter.ds_dir,'labellist/test.txt')
        task['trainParameter']['model']['lb_list'] = os.path.join(self.opter.ds_dir,'labels')
        task['trainParameter']['model']['names'] = os.path.join(self.opter.ds_dir,'names.txt')
        self.opter.model = task['trainParameter']['model']

        # 训练配置选项
        self.opter.epochs = task['trainParameter']['n_epochs']
        self.opter.batch_size = task['trainParameter']['batch_size']
        self.opter.acumulate = task['trainParameter']['acumulate']
        self.opter.transfer = task['trainParameter']['transfer']
        self.opter.img_size = task['trainParameter']['img_size']
        self.opter.resume = task['trainParameter']['resume']
        self.opter.cache_images = task['trainParameter']['cache_images']
        self.opter.notest = task['trainParameter']['notest']
        self.opter.arc = task['trainParameter']['loss_func']
        self.opter.optimizer = task['trainParameter']['optimizer']
        self.opter.augment = task['trainParameter']['augment']
        self.opter.map_type = task['trainParameter']['map_type']
        self.opter.device = task['trainParameter']['device']
        #数据增强
        self.opter.brightEnhancement = task['brightEnhancement']
        self.opter.spaceEnhancemenet = task['spaceEnhancemenet']

        '''
        在解析参数的时候就开始运行kemeans对数据集anchor box 进行聚类
        配置模型cfg文件的获取
        '''
        ds_path = self.opter.ds_dir
        shellpath = os.path.join(ds_path, 'create_custom_model.sh')
        # 模型的配置
        if opter.kmeans == True:
            parant_path = self.opter.model['paranet_path']
            kmeans_out(parant_path)
            kmeans_path = os.path.join(parant_path,'kmeans.txt')
            outcfg(shellpath,nc = self.opter.model['classes'],kmeans_path=kmeans_path)
        else :
            outcfg(shellpath,self.opter.model['classes'])

        self.opter.cfg = os.path.join(ds_path,'yolov3_nc{}.cfg'.format(task['trainParameter']['classes']))

        # inference 部分
        self.opter.source = task['inference']['source']
        self.opter.output_folder = task['inference']['output_folder']
        self.opter.conf_thres= task['inference']['conf_thres']
        self.opter.nms_thres= task['inference']['nms_thres']
        self.opter.device= task['inference']['device']
        self.opter.view_img= task['inference']['view_img']
        self.opter.test_result_path = task['inference']['test_result_path']


    def train_model(self,task):
        # stauts_train 为训练状态字典
        self.stauts_train = {}
        train(self.opter,self.status)
        return True

    def inference(self,inferenceMode, task):
        if inferenceMode =='test':
            #TODO:test4 输出各项指标
            cfg = self.opter.cfg
            data_dict = self.opter.model
            batch_size = self.opter.batch_size
            img_size = self.opter.img_size
            conf_thres = self.opter.conf_thres
            test_result_path = os.path.join(self.opter.ds_dir ,'test_result')
            if self.opter.model['classes']==1 & self.opter.map_type=='iouMap':# 如果是肺炎的测试
                results, maps = test4.test(cfg,
                           data_dict,
                           batch_size=batch_size,
                           img_size=img_size,
                           model=None,
                           conf_thres=conf_thres,  # 0.1 for speed
                           valOrTest='val',
                           test_result_path=test_result_path)
                #TODO: results, maps 输出的网络页面上
            elif self.opter.map_type=='Map':# Map测试方法为普通的Map
                pass
        else :
            # 将图片的结果直接输出到指定的文件夹内
            detector(self.opter)

        return True

    
class opter():
    def __init__(self,var=0,epochs=10,batch_size=8,accumulate=2,cfg = 'cfg/yolov3-xray.cfg',
                 data = './data/pneumonia.data',multi_scale = False,img_size=608,
                 rect = False,resume=False,transfer=False,notest = False,
                 cache_images=False,
                 arc = 'defaultpw', device='0',adam=True,tv_img_path='./results'):
        self.var = var
        self.epochs=epochs
        self.batch_size=batch_size
        self.accumulate= accumulate
        self.cfg = cfg
        self.data = data
        self.multi_scale=multi_scale
        self.img_size = img_size
        self.rect= rect
        self.resume=resume
        self.transfer=transfer
        self.notest=notest
        self.cache_images=cache_images
        self.arc = arc
        self.device=device
        self.adam = adam