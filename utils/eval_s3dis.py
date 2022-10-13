import os
import numpy as np
from scipy import stats

class S3DIS_Instance_evaluator:
    

    def __init__(self, logger) -> None:
        self.logger = logger 

        self.NUM_CLASSES = 13 

        self.total_gt_ins = np.zeros(self.NUM_CLASSES)

        self.IoU_threshold = 0.5

        self.ins_tp_num = [ [] for _ in range(self.NUM_CLASSES) ]
        self.ins_fp_num = [ [] for _ in range(self.NUM_CLASSES) ]

        self.all_mean_cov = [ [] for _ in range(self.NUM_CLASSES) ]
        self.all_mean_weighted_cov = [ [] for _ in range(self.NUM_CLASSES) ]

    
    def process(self, pred_info, gt_info):


        conf = pred_info['conf'] # (M, ) M = the num of proposals
        sem_label = pred_info['sem_label'] # (M, ) 
        sem_label = sem_label - 1

        pred_mask = pred_info['mask'] # (M, N) M = the num of proposals  N = the num of points

        sem_gt = gt_info['sem_gt'] # (N, ) semantic labels
        ins_gt = gt_info['ins_gt'] # (N, ) instance labels

        
        oneScene_pred_ins_mask = [ [] for _ in range(self.NUM_CLASSES) ]
        for sem_id, ins_mask in zip(sem_label, pred_mask):
            ins_mask = ins_mask.astype('bool')
            oneScene_pred_ins_mask[sem_id] += [ins_mask]

        oneScene_GT_ins_mask = [ [] for _ in range(self.NUM_CLASSES) ]
        for ins_id in np.unique(ins_gt):
            GT_ins_mask = (ins_gt == ins_id)
            GT_ins_sem = int(stats.mode(sem_gt[GT_ins_mask])[0])
            oneScene_GT_ins_mask[GT_ins_sem] += [GT_ins_mask]

        
        # calculate mCov mWCov
        for sem_id in range(self.NUM_CLASSES):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            all_ins_gt_point_n = 0

            # Find the proposal with the same semantics and the highest IoU
            for GT_ins_mask in oneScene_GT_ins_mask[sem_id]:
                IoU_max = 0.
                one_ins_gt_point_n = np.sum(GT_ins_mask)
                all_ins_gt_point_n += one_ins_gt_point_n # The total number of GT points, To calculate the weights in wCov
                
                for i_pred, pred_ins_mask in enumerate(oneScene_pred_ins_mask[sem_id]):
                    union = (GT_ins_mask | pred_ins_mask)
                    intersect = (GT_ins_mask & pred_ins_mask)
                    IoU = float(np.sum(intersect)) / np.sum(union)

                    IoU_max = max(IoU_max, IoU)
                
                sum_cov += IoU_max
                mean_weighted_cov += one_ins_gt_point_n * IoU_max
            
            # 
            if len(oneScene_GT_ins_mask[sem_id]) != 0:
                mean_cov = sum_cov / len(oneScene_GT_ins_mask[sem_id])
                self.all_mean_cov[sem_id].append(mean_cov)

                mean_weighted_cov /= all_ins_gt_point_n
                self.all_mean_weighted_cov[sem_id].append(mean_weighted_cov)
        
        # cal precision recall
        for sem_id in range(self.NUM_CLASSES):

            tp = [0.] * len(oneScene_pred_ins_mask[sem_id])
            fp = [0.] * len(oneScene_pred_ins_mask[sem_id])

            gtflag = np.zeros(len(oneScene_GT_ins_mask[sem_id])).astype('bool') 
            self.total_gt_ins[sem_id] += len(oneScene_GT_ins_mask[sem_id])

            for pred_i, pred_ins_mask in enumerate(oneScene_pred_ins_mask[sem_id]):
                IoU_max = -1.
                IoU_max_GT_ind = 0
                for GT_i, GT_ins_mask in enumerate(oneScene_GT_ins_mask[sem_id]):
                    union = (GT_ins_mask | pred_ins_mask)
                    intersect = (GT_ins_mask & pred_ins_mask)
                    IoU = float(np.sum(intersect)) / np.sum(union)

                    if IoU > IoU_max:
                        IoU_max = IoU
                        IoU_max_GT_ind = GT_i

                

                # https://github.com/WXinlong/ASIS/blob/d71c3d60e985f5bebe620c8e1a1cb0042fb2f5f6/models/ASIS/eval_iou_accuracy.py#L138
                if IoU_max > self.IoU_threshold: 
                # if IoU_max > self.IoU_threshold and gtflag[IoU_max_GT_ind]==False:
                    tp[pred_i] = 1
                    gtflag[IoU_max_GT_ind] = True
                else:
                    fp[pred_i] = 1
            
            self.ins_tp_num[sem_id] += tp
            self.ins_fp_num[sem_id] += fp
    


    def evaluate(self):

        MUCov = np.zeros(self.NUM_CLASSES)
        MWCov = np.zeros(self.NUM_CLASSES)
        for sem_id in range(self.NUM_CLASSES):
            MUCov[sem_id] = np.mean(self.all_mean_cov[sem_id])
            MWCov[sem_id] = np.mean(self.all_mean_weighted_cov[sem_id])


        precision = np.zeros(self.NUM_CLASSES)
        recall = np.zeros(self.NUM_CLASSES)

        for sem_id in range(self.NUM_CLASSES):
            tp = np.asarray(self.ins_tp_num[sem_id]).astype(np.float)
            fp = np.asarray(self.ins_fp_num[sem_id]).astype(np.float)
            tp = np.sum(tp)
            fp = np.sum(fp)
            rec = tp / self.total_gt_ins[sem_id]
            prec = tp / (tp + fp)

            precision[sem_id] = prec
            recall[sem_id] = rec

        
        self.logger.info('Instance Segmentation MUCov: {}'.format(MUCov))
        self.logger.info('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov)))
        self.logger.info('Instance Segmentation MWCov: {}'.format(MWCov))
        self.logger.info('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov)))
        self.logger.info('Instance Segmentation Precision: {}'.format(precision))
        self.logger.info('Instance Segmentation mPrecision: {}'.format(np.mean(precision)))
        self.logger.info('Instance Segmentation Recall: {}'.format(recall))
        self.logger.info('Instance Segmentation mRecall: {}'.format(np.mean(recall)))









        
