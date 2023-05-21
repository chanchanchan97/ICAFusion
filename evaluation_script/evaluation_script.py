"""Evaluate performance on multispectral pedestrian detection benchmark

This script evalutes multispectral detection performance.
We adopt [cocoapi](https://github.com/cocodataset/cocoapi)
and apply minor modification for KAISTPed benchmark.

"""
from collections import defaultdict
import argparse
import copy
import datetime
import json
import matplotlib
import numpy as np
import os
import pdb
import sys
import tempfile
import traceback

# matplotlib.use('Agg')
# from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from evaluation_script.coco import COCO
from evaluation_script.cocoeval import COCOeval, Params

font = {'size': 22}
matplotlib.rc('font', **font)


class KAISTPedEval(COCOeval):

    def __init__(self, kaistGt=None, kaistDt=None, iouType='segm', method='unknown'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        super().__init__(kaistGt, kaistDt, iouType)

        self.params = KAISTParams(iouType=iouType)   # parameters
        self.method = method

    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gbox = gt['bbox']
            gt['ignore'] = 1 \
                if gt['height'] < self.params.HtRng[id_setup][0] \
                or gt['height'] > self.params.HtRng[id_setup][1] \
                or gt['occlusion'] not in self.params.OccRng[id_setup] \
                or gbox[0] < self.params.bndRng[0] \
                or gbox[1] < self.params.bndRng[1] \
                or gbox[0] + gbox[2] > self.params.bndRng[2] \
                or gbox[1] + gbox[3] > self.params.bndRng[3] \
                else gt['ignore']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval = {}                      # accumulated evaluation results

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            #print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        OccRng = self.params.OccRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, OccRng, maxDet)
                         for catId in catIds
                         for imgId in p.imgIds]

        self._paramsEval = copy.deepcopy(self.params)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d, g, iscrowd)
        return ious

    def iou(self, dts, gts, pyiscrowd):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2, gx2) - max(dx1, gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2, gy2) - max(dy1, gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t) / unionarea
        return ious

    def evaluateImg(self, imgId, catId, hRng, oRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        try:
            p = self.params
            if p.useCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
            
            if len(gt) == 0 and len(dt) == 0:
                return None

            for g in gt:
                if g['ignore']:
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0

            # sort dt highest score first, sort gt ignore last
            gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
            gt = [gt[i] for i in gtind]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:maxDet]]

            if len(dt) == 0:
                return None

            # load computed ious        
            ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = ious[:, gtind]

            T = len(p.iouThrs)
            G = len(gt)
            D = len(dt)
            gtm = np.zeros((T, G))
            dtm = np.zeros((T, D))
            gtIg = np.array([g['_ignore'] for g in gt])
            dtIg = np.zeros((T, D))

            if not len(ious) == 0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t, 1 - 1e-10])
                        bstOa = iou
                        bstg = -2
                        bstm = -2
                        for gind, g in enumerate(gt):
                            m = gtm[tind, gind]
                            # if this gt already matched, and not a crowd, continue
                            if m > 0:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if bstm != -2 and gtIg[gind] == 1:
                                break
                            # continue to next gt unless better match made
                            if ious[dind, gind] < bstOa:
                                continue
                            # if match successful and best so far, store appropriately
                            bstOa = ious[dind, gind]
                            bstg = gind
                            if gtIg[gind] == 0:
                                bstm = 1
                            else:
                                bstm = -1

                        # if match made store id of match for both dt and gt
                        if bstg == -2:
                            continue
                        dtIg[tind, dind] = gtIg[bstg]
                        dtm[tind, dind] = gt[bstg]['id']
                        if bstm == 1:
                            gtm[tind, bstg] = d['id']

        except Exception:

            ex_type, ex_value, ex_traceback = sys.exc_info()            

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

            sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
            sys.stderr.write("[Error] Exception message : %s \n" % ex_value)
            for trace in stack_trace:
                sys.stderr.write("[Error] (Stack trace) %s\n" % trace)

            pdb.set_trace()

        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'hRng': hRng,
            'oRng': oRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        if not self.evalImgs:
            pass
            #print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        K = len(p.catIds) if p.useCats else 1
        M = len(p.maxDets)
        ys = -np.ones((T, R, K, M))     # -1 for the precision of absent categories

        xx_graph = []
        yy_graph = []

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1]                    # _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort')

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                inds = np.where(dtIg == 0)[1]
                tps = tps[:, inds]
                fps = fps[:, inds]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
            
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp) / I0
                    nd = len(tp)
                    recall = tp / npig
                    q = np.zeros((R,))

                    xx_graph.append(fppi)
                    yy_graph.append(1 - recall)

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except Exception:
                        pass
                    ys[t, :, k, m] = np.array(q)
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP': ys,
            'xx': xx_graph,
            'yy': yy_graph
        }

    @staticmethod
    def draw_figure(ax, eval_results, methods, colors):
        """Draw figure"""
        assert len(eval_results) == len(methods) == len(colors)

        for eval_result, method, color in zip(eval_results, methods, colors):
            mrs = 1 - eval_result['TP']
            mean_s = np.log(mrs[mrs < 2])
            mean_s = np.mean(mean_s)
            mean_s = float(np.exp(mean_s) * 100)

            xx = eval_result['xx']
            yy = eval_result['yy']

            ax.plot(xx[0], yy[0], color=color, linewidth=3, label=f'{mean_s:.2f}%, {method}')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

        yt = [1, 5] + list(range(10, 60, 10)) + [64, 80]
        yticklabels = ['.{:02d}'.format(num) for num in yt]

        yt += [100]
        yt = [yy / 100.0 for yy in yt]
        yticklabels += [1]
        
        ax.set_yticks(yt)
        ax.set_yticklabels(yticklabels)
        ax.grid(which='major', axis='both')
        ax.set_ylim(0.01, 1)
        ax.set_xlim(2e-4, 50)
        ax.set_ylabel('miss rate')
        ax.set_xlabel('false positives per image')

    def summarize(self, id_setup, res_file=None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100):
            OCC_TO_TEXT = ['none', 'partial_occ', 'heavy_occ']

            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
            titleStr = 'Average Miss Rate'
            typeStr = '(MR)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[' + '+'.join(['{:s}'.format(OCC_TO_TEXT[occ]) for occ in p.OccRng[id_setup]]) + ']'

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1 - s[:, :, :, mind]

            if len(mrs[mrs < 2]) == 0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs < 2] + 1e-5)
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)

            if res_file:
                res_file.write(iStr.format(titleStr, typeStr, setupStr, iouStr, heightStr, occlStr, mean_s * 100))
                res_file.write('\n')
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        return _summarize(iouThr=.5, maxDets=1000)


class KAISTParams(Params):
    """Params for KAISTPed evaluation api"""

    def setDetParams(self):
        super().setDetParams()

        # Override variables for KAISTPed benchmark
        self.iouThrs = np.array([0.5])
        self.maxDets = [1000]

        # KAISTPed specific settings
        self.fppiThrs = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])
        #self.HtRng = [[55, 1e5 ** 2], [50, 75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
        self.HtRng = [[55, 1e5 ** 2], [115, 1e5 ** 2], [45, 115], [1, 45], [1, 1e5 ** 2], [1, 1e5 ** 2], [1, 1e5 ** 2]] # jifengshen
        #self.OccRng = [[0, 1], [0, 1], [2], [0, 1, 2]]
        self.OccRng = [[0, 1], [0], [0], [0], [0], [1], [2]]
        #self.SetupLbl = ['Reasonable', 'Reasonable_small', 'Reasonable_occ=heavy', 'All']
        self.SetupLbl = ['Reasonable', 'scale=near', 'scale=medium', 'scale=far', 'occ=none', 'occ=partial', 'occ=heavy', 'All']
        
        self.bndRng = [5, 5, 635, 507]  # discard bbs outside this pixel range


class KAIST(COCO):

    def txt2json(self, txt):
        """
        Convert txt file to coco json format
        Arguments:
            `txt`: Path to annotation file that txt
        """
        predict_result = []
        f = open(txt, 'r')
        #print(f)
        lines = f.readlines()
        for line in lines:
            json_format = {}
            pred_info = [float(ll) for ll in line.split(',')]
            json_format["image_id"] = pred_info[0] - 1                                      # image id
            json_format["category_id"] = 1                                                  # pedestrian
            json_format["bbox"] = [pred_info[1], pred_info[2], pred_info[3], pred_info[4]]  # bbox
            json_format["score"] = pred_info[5]

            predict_result.append(json_format)
        return predict_result

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """

        # If resFile is a text file, convert it to json
        if type(resFile) == str and resFile.endswith('.txt'):
            anns = self.txt2json(resFile)
            _resFile = next(tempfile._get_candidate_names())
            with open(_resFile, 'w') as f:
                json.dump(anns, f, indent=4)
            res = super().loadRes(_resFile)
            os.remove(_resFile)
        elif type(resFile) == str and resFile.endswith('.json'):
            res = super().loadRes(resFile)
        else:
            raise Exception('[Error] Exception extension : %s \n' % resFile.split('.')[-1]) 

        return res


def evaluate(test_annotation_file: str, user_submission_file: str, phase_codename: str = 'Multispectral', plot=False):
    """Evaluates the submission for a particular challenge phase and returns score

    Parameters
    ----------
    test_annotations_file: str
        Path to test_annotation_file on the server
    user_submission_file: str
        Path to file submitted by the user
    phase_codename: str
        Phase to which submission is made

    Returns
    -------
    Dict
        Evaluated/Accumulated KAISTPedEval objects for All/Day/Night
    """
    kaistGt = KAIST(test_annotation_file)
    kaistDt = kaistGt.loadRes(user_submission_file)

    imgIds = sorted(kaistGt.getImgIds())
    method = os.path.basename(user_submission_file).split('_')[0]
    kaistEval = KAISTPedEval(kaistGt, kaistDt, 'bbox', method)

    kaistEval.params.catIds = [1]

    eval_result = {
        'all': copy.deepcopy(kaistEval),
        'day': copy.deepcopy(kaistEval),
        'night': copy.deepcopy(kaistEval),
        'near': copy.deepcopy(kaistEval),
        'medium': copy.deepcopy(kaistEval),
        'far': copy.deepcopy(kaistEval),
        'none': copy.deepcopy(kaistEval),
        'partial': copy.deepcopy(kaistEval),
        'heavy': copy.deepcopy(kaistEval),
    }

    eval_result['all'].params.imgIds = imgIds
    eval_result['all'].evaluate(0)
    eval_result['all'].accumulate()
    MR_all = eval_result['all'].summarize(0)

    eval_result['day'].params.imgIds = imgIds[:1455]
    eval_result['day'].evaluate(0)
    eval_result['day'].accumulate()
    MR_day = eval_result['day'].summarize(0)

    eval_result['night'].params.imgIds = imgIds[1455:]
    eval_result['night'].evaluate(0)
    eval_result['night'].accumulate()
    MR_night = eval_result['night'].summarize(0)
    
    eval_result['near'].params.imgIds = imgIds
    eval_result['near'].evaluate(1)
    eval_result['near'].accumulate()
    MR_near = eval_result['near'].summarize(1)
    
    eval_result['medium'].params.imgIds = imgIds
    eval_result['medium'].evaluate(2)
    eval_result['medium'].accumulate()
    MR_medium = eval_result['medium'].summarize(2)
    
    eval_result['far'].params.imgIds = imgIds
    eval_result['far'].evaluate(3)
    eval_result['far'].accumulate()
    MR_far = eval_result['far'].summarize(3)
    
    eval_result['none'].params.imgIds = imgIds
    eval_result['none'].evaluate(4)
    eval_result['none'].accumulate()
    MR_none = eval_result['none'].summarize(4)
    
    eval_result['partial'].params.imgIds = imgIds
    eval_result['partial'].evaluate(5)
    eval_result['partial'].accumulate()
    MR_partial = eval_result['partial'].summarize(5)
    
    eval_result['heavy'].params.imgIds = imgIds
    eval_result['heavy'].evaluate(6)
    eval_result['heavy'].accumulate()
    MR_heavy = eval_result['heavy'].summarize(6)

    recall_all = 1 - eval_result['all'].eval['yy'][0][-1]

    if plot:
        msg = f'\n########## Method: {method} ##########\n' \
            + f'MR_all: {MR_all * 100:.2f}\n' \
            + f'MR_day: {MR_day * 100:.2f}\n' \
            + f'MR_night: {MR_night * 100:.2f}\n' \
            + f'MR_near: {MR_near * 100:.2f}\n' \
            + f'MR_medium: {MR_medium * 100:.2f}\n' \
            + f'MR_far: {MR_far * 100:.2f}\n' \
            + f'MR_none: {MR_none * 100:.2f}\n' \
            + f'MR_partial: {MR_partial * 100:.2f}\n' \
            + f'MR_heavy: {MR_heavy * 100:.2f}\n' \
            + f'recall_all: {recall_all * 100:.2f}\n' \
            + '######################################\n\n'
        print(msg)

    return eval_result


def draw_all(eval_results, filename='figure.jpg'):
    """Draw all results in a single figure as Miss rate versus false positive per-image (FPPI) curve

    Parameters
    ----------
    eval_results: List of Dict
        Aggregated evaluation results from evaluate function.
        Dictionary contains KAISTPedEval objects for All/Day/Night
    filename: str
        Filename of figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(45, 10))

    methods = [res['all'].method for res in eval_results]
    colors = [plt.cm.get_cmap('Paired')(ii)[:3] for ii in range(len(eval_results))]

    eval_results_all = [res['all'].eval for res in eval_results]
    KAISTPedEval.draw_figure(axes[0], eval_results_all, methods, colors)
    axes[0].set_title('All')

    eval_results_day = [res['day'].eval for res in eval_results]
    KAISTPedEval.draw_figure(axes[1], eval_results_day, methods, colors)
    axes[1].set_title('Day')

    eval_results_night = [res['night'].eval for res in eval_results]
    KAISTPedEval.draw_figure(axes[2], eval_results_night, methods, colors)
    axes[2].set_title('Night')

    filename += '' if filename.endswith('.jpg') or filename.endswith('.png') else '.jpg'
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--annFile', type=str, default='D:/Project/MLPD-Multi-Label-Pedestrian-Detection-main/evaluation_script/KAIST_annotation.json',
                        help='Please put the path of the annotation file. Only support json format.')
    parser.add_argument('--rstFiles', type=str, nargs='+', default=['evaluation_script/MLPD_result.json'],
                        help='Please put the path of the result file. Only support json, txt format.')
    parser.add_argument('--evalFig', type=str, default='KASIT_BENCHMARK.jpg',
                        help='Please put the output path of the Miss rate versus false positive per-image (FPPI) curve')
    args = parser.parse_args()

    args.rstFiles = 'D:/Project/MLPD-Multi-Label-Pedestrian-Detection-main/evaluation_script/yolov5_CMAFF_result.txt'
    phase = "Multispectral"
    results = [evaluate(args.annFile, args.rstFiles, phase)]

    # Sort results by MR_all
    results = sorted(results, key=lambda x: x['all'].summarize(0), reverse=True)
    draw_all(results, filename=args.evalFig)
