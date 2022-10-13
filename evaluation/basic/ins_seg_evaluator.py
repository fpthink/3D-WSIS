import sys
from typing import Sequence

import numpy as np

from .instances import VertInstance
from .evaluator import DatasetEvaluator

import utils

class InstanceEvaluator(DatasetEvaluator):
    """
    Evaluate instance segmentation metrics.
    """
    # ---------- Evaluation params ---------- #
    # overlaps for evaluation
    OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    # minimum region size for evaluation [verts]
    MIN_REGION_SIZES = np.array([100])
    # distance thresholds [m]
    DISTANCE_THRESHES = np.array([float("inf")])
    # distance confidences
    DISTANCE_CONFS = np.array([-float("inf")])
    def __init__(self,
                 class_labels: Sequence[str],
                 class_ids: Sequence[int],
                 logger=None,
                 **kwargs,):
        """
        Args:
            ignore_label: deprecated argument
        """
        super().__init__(
            class_labels=class_labels,
            class_ids=class_ids,
            logger=logger)
        self.reset()

    def reset(self):
        # initialize precision-recall container to calculate AP
        self.matches = {}
        self.prec_recall_total = {}
        self.ap_scores = np.zeros((len(self.DISTANCE_THRESHES),
                                   len(self.class_labels),
                                   len(self.OVERLAPS)), np.float)
        self.avgs = {}


    def assign_instances_for_scan(self, scene_name, pred_info, gt_ids):
        # get gt instances
        gt_instances = VertInstance.get_instances(gt_ids,
                                                  self.class_ids,
                                                  self.class_labels,
                                                  self.id_to_label)

        # associate
        gt2pred = gt_instances.copy()
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt["matched_pred"] = []
        pred2gt = {}
        for label in self.class_labels:
            pred2gt[label] = []
        num_pred_instances = 0
        # mask of void labels in the groundtruth
        bool_void = np.logical_not(np.in1d(gt_ids // 1000, self.class_ids))
        # go thru all prediction masks
        nMask = pred_info["label_id"].shape[0]

        for i in range(nMask):
            label_id = int(pred_info["label_id"][i])
            conf = pred_info["conf"][i]
            if not label_id in self.id_to_label:
                continue
            label_name = self.id_to_label[label_id]
            # read the mask
            pred_mask = pred_info["mask"][i]  # [N], long
            if len(pred_mask) != len(gt_ids):
                sys.stderr.write(
                    f"ERROR: wrong number of lines in mask#{i}: "
                    f"({len(pred_mask)}) vs #points ({len(gt_ids)})\n")
                sys.exit(2)
            # convert to binary
            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.MIN_REGION_SIZES[0]:
                continue  # skip if empty

            # record pred instance as dict
            pred_instance = {}
            pred_instance["filename"] = f"{scene_name}_{num_pred_instances:03d}"
            pred_instance["pred_id"] = num_pred_instances
            pred_instance["label_id"] = label_id
            pred_instance["instance_count"] = num
            pred_instance["confidence"] = conf
            pred_instance["pred_mask"] = pred_mask
            pred_instance["void_intersection"] = np.count_nonzero(
                np.logical_and(bool_void, pred_mask))

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(
                    np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy["intersection"] = intersection
                    pred_copy["intersection"] = intersection
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
            pred_instance["matched_gt"] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt


    def assign(self, scene_name, pred_info, gt_ids):
        gt2pred, pred2gt = self.assign_instances_for_scan(scene_name, pred_info, gt_ids)
        self.matches[scene_name] = {
            "instance_pred": pred2gt,
            "instance_gt": gt2pred
        }


    def evaluate(self, ap=True, prec_rec=True):
        if ap:
            self.evaluate_matches()
            self.print_results()
        if prec_rec:
            self.print_prec_recall()

    def print_results(self):
        haeders = ["class", "AP", "AP_50%", "AP_25%"]
        results = []
        max_length = max(15, max(map(lambda x: len(x), self.class_labels)))
        self.logger.info("Evaluation average precision(AP) for instance segmentation:")
        for (li, class_label) in enumerate(self.class_labels):
            ap_avg = self.avgs["classes"][class_label]["ap"]
            ap_50o = self.avgs["classes"][class_label]["ap50%"]
            ap_25o = self.avgs["classes"][class_label]["ap25%"]
            results.append((class_label.ljust(max_length, " "), ap_avg, ap_50o, ap_25o))
        
        ap_table = utils.table(
            results,
            headers=haeders,
            stralign="left"
        )
        for line in ap_table.split("\n"):
            self.logger.info(line)

        # print mean results
        self.logger.info("Mean average precision(AP):")
        mean_ap_dict = {}
        for idx, metric in zip(["all_ap", "all_ap_50%", "all_ap_25%"], ["AP", "AP_50%", "AP_25%"]):
            mean_ap_dict[metric] = self.avgs[idx]
            
        acc_table = utils.create_small_table(mean_ap_dict)
        for line in acc_table.split("\n"):
            self.logger.info(line)
        self.logger.info("")


    def evaluate_matches(self):
        # results: class x overlap
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(self.MIN_REGION_SIZES, self.DISTANCE_THRESHES, self.DISTANCE_CONFS)):
            for oi, overlap_th in enumerate(self.OVERLAPS):
                self.prec_recall_total[overlap_th] = {}
                pred_visited = {}
                for m in self.matches:
                    for p in self.matches[m]["instance_pred"]:
                        for label_name in self.class_labels:
                            for p in self.matches[m]["instance_pred"][label_name]:
                                if "filename" in p:
                                    pred_visited[p["filename"]] = False
                for li, label_name in enumerate(self.class_labels):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in self.matches:
                        pred_instances = self.matches[m]["instance_pred"][label_name]
                        gt_instances = self.matches[m]["instance_gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt for gt in gt_instances
                            if gt["instance_id"] >= 0 and gt["instance_count"] >=
                            min_region_size and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                        # collect matches
                        for (gti, gt) in enumerate(gt_instances):
                            found_match = False
                            num_pred = len(gt["matched_pred"])
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["filename"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["instance_count"] + pred["instance_count"] -
                                    pred["intersection"])
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["filename"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match == True]
                        cur_score = cur_score[cur_match == True]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["instance_count"] + pred["instance_count"] -
                                    gt["intersection"])
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    # group?
                                    if gt["instance_id"] < 1000:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if gt["instance_count"] < min_region_size or gt[
                                            "med_dist"] > distance_thresh or gt[
                                                "dist_conf"] < distance_conf:
                                        num_ignore += gt["intersection"]
                                proportion_ignore = float(
                                    num_ignore) / pred["instance_count"]
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted,
                                                                return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        if (len(y_true_sorted_cumsum) == 0):
                            num_true_examples = 0
                        else:
                            num_true_examples = y_true_sorted_cumsum[-1]
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall[-1] = 0.

                        self.prec_recall_total[overlap_th][label_name] = [precision, recall]

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0],
                                                    recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5],
                                                "valid")
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    self.ap_scores[di, li, oi] = ap_current

        # average calculation
        d_inf = 0
        o50 = np.where(np.isclose(self.OVERLAPS, 0.5))
        o25 = np.where(np.isclose(self.OVERLAPS, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.OVERLAPS, 0.25)))
        self.avgs["all_ap"] = np.nanmean(self.ap_scores[d_inf, :, oAllBut25])
        self.avgs["all_ap_50%"] = np.nanmean(self.ap_scores[d_inf, :, o50])
        self.avgs["all_ap_25%"] = np.nanmean(self.ap_scores[d_inf, :, o25])
        self.avgs["classes"] = {}
        for (li, label_name) in enumerate(self.class_labels):
            self.avgs["classes"][label_name] = {}
            self.avgs["classes"][label_name]["ap"] = \
                np.average(self.ap_scores[d_inf, li, oAllBut25])
            self.avgs["classes"][label_name]["ap50%"] = \
                np.average(self.ap_scores[d_inf, li, o50])
            self.avgs["classes"][label_name]["ap25%"] = \
                np.average(self.ap_scores[d_inf, li, o25])


    # modify from https://github.com/Yang7879/3D-BoNet/blob/master/main_eval.py
    def print_prec_recall(self, threshold: float=0.5):
        # init the confusion matrix dict
        TP_FP_Total = {}
        for class_id in self.class_ids:
            TP_FP_Total[class_id] = {}
            TP_FP_Total[class_id]["TP"] = 0
            TP_FP_Total[class_id]["FP"] = 0
            TP_FP_Total[class_id]["Total"] = 0

        for m in utils.track(self.matches):
            # pred ins
            ins_pred_by_sem = {}
            ins_gt_by_sem = {}
            for class_id, class_label in zip(self.class_ids, self.class_labels):
                # pred ins
                ins_pred_by_sem[class_id] = []
                pred_instances = self.matches[m]["instance_pred"][class_label]
                for pred in pred_instances:
                    ins_pred_by_sem[class_id].append(pred["pred_mask"])
                # gt ins
                ins_gt_by_sem[class_id] = []
                gt_instances = self.matches[m]["instance_gt"][class_label]
                for gt in gt_instances:
                    ins_gt_by_sem[class_id].append(gt["gt_mask"])

            # to associate
            for class_id, class_label in zip(self.class_ids, self.class_labels):
                ins_pred_tp = ins_pred_by_sem[class_id] # [num_pred, N]
                ins_gt_tp = ins_gt_by_sem[class_id] # [num_gt, N]

                flag_pred = np.zeros(len(ins_pred_tp), dtype=np.int8)

                for i_p, ins_p in enumerate(ins_pred_tp):
                    for i_g, ins_g in enumerate(ins_gt_tp):
                        u = ins_g | ins_p
                        i = ins_g & ins_p
                        iou_tp = float(np.sum(i)) / (np.sum(u) + 1e-8)
                        if iou_tp > threshold:
                            flag_pred[i_p] = 1
                            break

                # fullfil
                TP_FP_Total[class_id]["TP"] += np.sum(flag_pred)
                TP_FP_Total[class_id]["FP"] += len(flag_pred) - np.sum(flag_pred)
                TP_FP_Total[class_id]["Total"] += len(ins_gt_tp)

        # build precision-recall table
        pre_all = []
        rec_all = []
        haeders = ["class", "precision", "recall"]
        results = []
        self.logger.info("Evaluation precision and recall for instance segmentation:")
        max_length = max(15, max(map(lambda x: len(x), self.class_labels)))
        for class_id, class_label in zip(self.class_ids, self.class_labels):
            TP = TP_FP_Total[class_id]["TP"]
            FP = TP_FP_Total[class_id]["FP"]
            Total = TP_FP_Total[class_id]["Total"]
            pre = float(TP) / (TP + FP + 1e-8)
            rec = float(TP) / (Total + 1e-8)
            results.append((class_label.ljust(max_length, " "), pre, rec))
            pre_all.append(pre)
            rec_all.append(rec)

        pre_rec_table = utils.table(
            results,
            headers=haeders,
            stralign="left"
        )
        for line in pre_rec_table.split("\n"):
            self.logger.info(line)
        
        # print mean results
        self.logger.info("Mean precision and recall:")
        mean_pr_dict = {}
        for metric, value in zip(["precision", "recall"], [np.mean(pre_all), np.mean(rec_all)]):
            mean_pr_dict[metric] = f"{value:.3f}"
            
        acc_table = utils.create_small_table(mean_pr_dict)
        for line in acc_table.split("\n"):
            self.logger.info(line)
        self.logger.info("")

