from pathlib import Path
import numpy as np
from cellpose import models, dynamics, plot
from tifffile import tifffile
import json
from copy import deepcopy
from roifile import ImagejRoi
import cv2
from tqdm import tqdm, trange
import pandas as pd

class CellamanderRecipe:

    def __init__(self, json):
        self.name = json['meta']['name']
        self.instructions = json['data']
        self._raw_json = deepcopy(json)

    # TODO: 
    def save(self, path):
        raise NotImplementedError

class InvalidRecipe(Exception):
    pass


class Cellamander:

    def __init__(self, **kwargs):
        """
        Constructor 
        Additional kwargs are passed to the CellposeModel constructor
        """
        self._model = models.CellposeModel(**kwargs)

    def _load_recipe(self, recipe):
        """
        Load a recipe from JSON object or file path

        Returns
        --------
        CellamanderRecipe instance
        """
        if isinstance(recipe, str):
            recipe = Path(recipe).resolve()

        if isinstance(recipe, Path):
            recipe = recipe.resolve()

            if recipe.suffix != ".json":
                raise InvalidFileType("{} must be a JSON file with suffix ending in `.json`".format(recipe.name))

            if not recipe.exists():
                raise FileNotFound("{} cannot be found".format(str(recipe)))

            with open(recipe, 'r') as f:
                recipe = CellamanderRecipe(json.load(f))

        if not isinstance(recipe, CellamanderRecipe):
            pass
            # raise InvalidRecipe()

        return recipe

    @staticmethod
    def _normalize_matrix(m, norm_to):
        """
        Normalize a matrix to be within the same range as another matrix

        Parameters
        --------
        m np.array The matrix to normalize
        norm_to np.array The reference
        """
        m = (m-np.min(m))/(np.max(m)-np.min(m))
        diff = (np.max(norm_to)-np.min(norm_to))

        return m*diff+np.min(norm_to)

    def mander(
        self, 
        images, 
        recipes, 
        batch_size=8,
        resample=True,
        normalize=True,
        invert=False,
        rescale=None,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,
        niter=None,
        augment=False,
        tile=True,
        tile_overlap=0.1,
        bsize=224,
        interp=True,
        progress=None
    ):
        """
        Generate masks using a set of images and recipes

        Parameters
        --------
        images np.array A set of images of the format n x row x col x channels
        recipes list A list of CellamanderRecipe instances
        
        The remainder parameters are passed to cellpose. 

        Returns
        --------
        list, list List of masks for each image, list of flows for each image
        """
        all_masks = []
        all_flows = []
        if len(images.shape) < 3:
            raise InsufficientChannels("Images should be channels-last")
        elif len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        loaded_recipes = []
        for recipe in recipes:
            loaded_recipes.append(self._load_recipe(recipe))

        for n in range(images.shape[0]):
            masks, flows = self._mander_image(
                images[n], 
                loaded_recipes, 
                batch_size=batch_size,
                resample=resample,
                normalize=normalize,
                invert=invert,
                rescale=rescale,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                niter=niter,
                augment=augment,
                tile=tile,
                tile_overlap=tile_overlap,
                bsize=bsize,
                interp=interp,
                progress=progress
            )
            all_masks.append(masks)
            all_flows.append(all_flows)

        return all_masks, all_flows

    def _mander_image(
        self, 
        image, 
        recipes, 
        batch_size=8,
        resample=True,
        normalize=True,
        invert=False,
        rescale=None,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,
        niter=None,
        augment=False,
        tile=True,
        tile_overlap=0.1,
        bsize=224,
        interp=True,
        progress=None
    ):
        """
        Generate masks for a given image and a set of recipes

        Parameters
        --------
        image np.array An image of the format row x col x channels
        recipes list A list of CellamanderRecipe instances
        
        The remainder parameters are passed to cellpose. 

        Returns
        --------
        dict, dict  Dictionary of masks for each recipe, flows for each recipe. 
                    Keys are recipe names
        """
        flows = {}
        for c in range(image.shape[-1]):
            _, tmp, _ = self._model.eval(
                image[...,c],
                batch_size=batch_size,
                resample=resample,
                channels=[0,0],
                normalize=normalize,
                invert=invert,
                rescale=rescale,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                niter=niter,
                augment=augment,
                tile=tile,
                tile_overlap=tile_overlap,
                bsize=bsize,
                interp=interp,
                compute_masks=False,
                progress=None
            )
            flows[c] = { 'dP': np.squeeze(tmp[1]), 'cellprob': np.squeeze(tmp[2]) }

        # Need this for proper niter calculation
        # See https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py
        if diameter is not None and diameter > 0:
            rescale = self._model.diam_mean / diameter
        elif rescale is None:
            diameter = self._model.diam_labels
            rescale = self._model.diam_mean / diameter

        all_masks = {}
        all_flows = {}
        
        device = None
        if self._model.gpu:
            device = self._model.device

        niter0 = (1 / rescale * 200)
        niter = niter0 if niter is None or niter == 0 else niter
        resize = [ image.shape[0], image.shape[1] ] if (not resample and rescale != 1) else None

        for recipe in recipes:
            dP = np.zeros_like(flows[0]['dP'])
            cellprob = np.zeros_like(flows[0]['cellprob'])

            norm_to = None

            for c,value in enumerate(recipe.instructions):
                dP += value['dP']*flows[c]['dP']
                cellprob += value['cellprob']*flows[c]['cellprob']
                if norm_to is None and value['cellprob'] > 0:
                    norm_to = c

            dP = Cellamander._normalize_matrix(dP, flows[norm_to]['dP'])
            cellprob = Cellamander._normalize_matrix(cellprob, flows[norm_to]['cellprob'])

            masks = []
            p = []
            outputs = dynamics.resize_and_compute_masks(
                dP,
                cellprob,
                niter=niter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                interp=interp,
                resize=resize,
                min_size=min_size,
                device=device
            )
            masks.append(outputs[0])
            p.append(outputs[1])

            all_masks[recipe.name] = np.squeeze(masks)
            all_flows[recipe.name] = [ plot.dx_to_circ(dP), dP, cellprob, np.squeeze(p) ]

        return all_masks, all_flows

    def find_recipes(self, images, roi_files, required_channels=[], absent_channels=[], ignore_dP_channels=[], ignore_cellprob_channels=[], step_size=0.25, fix_dPs=[], fix_cellprobs=[], **kwargs):
        """
        Determine performance of different recipes

        Searches a range of different weight combinations, using ground
        truth ROIs to determine performance.

        Parameters
        --------
        images np.array Images that have ground truth masks
        roi_files list|np.array Paths to ImageJ ROIs, a list of ROI paths for each image in images
        required_channels list List of any channels that must be present for this phenotype
        absent_channels list List of any channels that must be absent for this phenotype
        ignore_dP_channels list List of any channels whose dP matrices should not be evaluated
        ignore_cellprob_channels list List of any channels whose cellprob matrices should not be evaluated
        step_size float
        fix_dPs list List of floats or None. Channel i dP will be fixed to the value at index i
        fix_cellprobs list List of floats or None. Channel i dP will be fixed to the value at index i

        All other parameters passed to Cellamander.mander()

        Returns
        --------
        pd.DataFrame Dataframe containing performance metrics for each ROI
        pd.DataFrame Dataframe containing performance metrics for each predicted mask
        pd.DataFrame The recipe IDs and weights
        """

        # To save search time, we'll only take non-negative weights for the
        # target channel
        num_channels = images.shape[-1]

        weights = []
        for c in range(num_channels):
            if c in ignore_dP_channels:
                dP_r = np.array([0])
            elif c in required_channels:
                dP_r = np.arange(step_size, 1+step_size, step_size)
            elif c in absent_channels:
                dP_r = np.arange(-1, step_size, step_size)
            else:
                dP_r = np.arange(-1, 1+step_size, step_size)
            weights.append(dP_r)

        for c in range(num_channels):
            if c in ignore_cellprob_channels:
                cellprob_r = np.array([0])
            elif c in required_channels:
                cellprob_r = np.arange(step_size, 1+step_size, step_size)
            elif c in absent_channels:
                cellprob_r = np.arange(-1, step_size, step_size)
            else:
                cellprob_r = np.arange(-1, 1+step_size, step_size)
            weights.append(cellprob_r)

        weights = np.transpose(np.array(np.meshgrid(*(weights)))).reshape(-1, num_channels*2)

        for c,fixed_dP in enumerate(fix_dPs):
            if fixed_dP is not None:
                weights = weights[weights[...,c] == fixed_dP]

        for c,fixed_cellprob in enumerate(fix_cellprobs):
            if fixed_cellprob is not None:
                weights = weights[weights[...,c+num_channels] == fixed_cellprob]
        
        gt_dfs, pred_dfs = [], []

        for weights_idx in trange(weights.shape[0]):
            recipe_json = {
                'meta': { 'name': weights_idx },
                'data': []
            }
            for c in range(num_channels):
                recipe_json['data'].append({
                    'dP': weights[weights_idx, c],
                    'cellprob': weights[weights_idx, c+num_channels]
                })
            recipe = CellamanderRecipe(recipe_json)

            gt_df, pred_df = self._test_recipe(images, recipe, roi_files, **kwargs)
            gt_dfs.append(gt_df)
            pred_dfs.append(pred_df)

        gt_dfs = pd.concat(gt_dfs)
        pred_dfs = pd.concat(pred_dfs)

        columns = {}
        for c in range(num_channels):
            columns[c] = 'dP_{}'.format(c)
            columns[c+num_channels] = 'cellprob_{}'.format(c)
        weights = pd.DataFrame(weights).rename(columns=columns)

        return pred_dfs, gt_dfs, weights

    def _test_recipe(self, images, recipe, roi_files, **kwargs):
        gt_df_template = {
            'image_id': [],
            'roi': [],
            'recipe_id': [],
            'labels': [],
            'ious': [],
            'best_label': [],
            'best_iou': []
        }
        pred_df = {
            'image_id': [],
            'recipe_id': [],
            'iou': [],
            'miou': [],
            'miou_tp_50': [],
            'tp_50': [],
            'fp_50': [],
            'fn_50': [],
            'ap_50': []   
        }

        masks, flows = self.mander(images, [ recipe ], **kwargs)
        gt_mask = np.zeros_like(masks[0][recipe.name], dtype=np.int32)
        gt_df = []

        for n in range(images.shape[0]):
            tmp = deepcopy(gt_df_template)
            my_rois = roi_files[n]
            my_masks = masks[n][recipe.name]
            for roi_file in my_rois:
                roi = ImagejRoi.fromfile(roi_file)
                vertices = np.expand_dims(roi.coordinates(), axis=1).astype(np.int32)
                roi_poly = cv2.fillPoly(np.zeros_like(images[n][...,0]), [vertices], 1)
                gt_mask += roi_poly

                tmp['image_id'].append(n)
                tmp['roi'].append(roi.name)
                tmp['recipe_id'].append(recipe.name)

                labels = np.unique(my_masks[(roi_poly > 0) & (my_masks > 0)])
                ious = []
                for label in labels:
                    intersection = np.sum( ((my_masks == label) & (roi_poly > 0)) )
                    union = np.sum( ((my_masks == label) | (roi_poly > 0)) )
                    ious.append(intersection/union)

                tmp['labels'].append(labels)
                tmp['ious'].append(ious)

                if len(labels) > 0:
                    tmp['best_iou'].append(np.max(ious))
                    tmp['best_label'].append(labels[ious.index(np.max(ious))])
                else:
                    tmp['best_iou'].append(0.0)
                    tmp['best_label'].append(0)

            tmp = pd.DataFrame(tmp)
            all_labels = tmp['best_label'].unique()

            num_fn = tmp.loc[(tmp['best_iou'] < 0.5)].shape[0]
            num_tp = tmp.loc[(tmp['best_iou'] >= 0.5)].shape[0]
            num_fp = np.unique(my_masks[(my_masks != 0) & ~np.isin(my_masks, all_labels)]).shape[0]
            
            ap = num_tp / (num_tp+num_fp+num_fn)

            if ap > 0:
                intersection = np.sum((my_masks > 0) & (gt_mask > 0))
                union = np.sum((my_masks > 0) | (gt_mask > 0))
                if intersection == 0 or union == 0:
                    iou = 0
                else:
                    iou = intersection / union

                pred_df['image_id'].append(n)
                pred_df['recipe_id'].append(recipe.name)
                pred_df['iou'].append(iou)
                pred_df['miou'].append(np.mean(tmp['best_iou']))
                pred_df['miou_tp_50'].append(np.mean(tmp.loc[(tmp['best_iou'] >= 0.5), 'best_iou']))
                pred_df['tp_50'].append(num_tp)
                pred_df['fp_50'].append(num_fp)
                pred_df['fn_50'].append(num_fn)
                pred_df['ap_50'].append(ap)
            gt_df.append(tmp)

        gt_df = pd.concat(gt_df)
        pred_df = pd.DataFrame(pred_df)
        return gt_df, pred_df









