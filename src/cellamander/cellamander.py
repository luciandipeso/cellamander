from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import models, io
from tifffile import tifffile
import json

# images = [ [[]] ] # n x r x c x channels
# cs = Cellamander()
# cs.mander(images, { {}, {} })
# cs.mander(images, [ 'CD4.json', 'CD11c.json' ])
# recipe = cs.find_recipe(images, 'path/to/rois')
# cs.save_recipe(recipe, 'path/to/recipe')


class Cellamander:

    def __init__(self, **kwargs):
        """
        Constructor 
        Additional kwargs are passed to the CellposeModel constructor
        """
        self._model = models.CellposeModel(**kwargs)

    def _unpack_recipe(self, recipe):
        if isinstance(recipe, str):
            recipe = Path(recipe).resolve()

        if isinstance(recipe, Path):
            recipe = recipe.resolve()

            if recipe.suffix != ".json":
                raise InvalidFileType("{} must be a JSON file with suffix ending in `.json`".format(recipe.name))

            if not recipe.exists():
                raise FileNotFound("{} cannot be found".format(str(recipe)))

            with open(recipe, 'r') as f:
                recipe = json.load(f)

        recipe_name = recipe['meta']['name']
        recipe_instructions = recipe['data']

        return recipe_name, recipe_instructions

    def _normalize_matrix(self, m, norm_to):
        m = (m-np.min(m))/(np.max(m)-np.min(comb_vals))
        diff = (np.max(norm_to)-np.min(norm_to))

        return m*diff+np.min(norm_to)

    def mander(self, 
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
        progress=None
    ):
        # Get base flows
        flows = {}
        for c in images.shape[-1]:
            _, tmp, _ = self._model.eval(
                images[...,c],
                batch_size=batch_size,
                resample=resample,
                channels='[0,0]',
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
            flows[c] = { 'dP': flows[1], 'cellprob': flows[2] }

        # Need this for proper niter calculation
        # See https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py
        if diameter is not None and diameter > 0:
            rescale = self._model.diam_mean / diameter
        elif rescale is None:
            diameter = self._model.diam_labels
            rescale = self._model.diam_mean / diameter

        all_masks = {}
        all_flows = {}
        all_dP = {}
        all_p = {}
        all_cellprob = {}
        device = None
        if self._model.gpu:
            device = self._model.device

        niter0 = (1 / rescale * 200)
        niter = niter0 if niter is None or niter == 0 else niter
        resize = [ images.shape[1], images.shape[2] ] if (not resample and rescale != 1) else None
        nimg = images.shape[0]

        for recipe in recipes:
            """
            Recipe structure:
            {
                'meta': { 'name': 'My name here'},
                'data': {0: [0,0.2], [0,0.3], ...}
            }
            """
            name, recipe = self._unpack_recipe(recipe)

            dP = np.zeros_like(flows[0][0])
            cellprob = np.zeros_like(flows[0][1])

            norm_to = None

            for c,value in recipe:
                dP += value['dP']*flows[c][0]
                cellprob += value['cellprob']*flows[c][1]
                if norm_to is None and value[0] > 0 and value[1] > 0:
                    norm_to = c

            dP = self._normalize_matrix(dP, flows[norm_to][0])
            cellprob = self._normalize_matrix(cellprob, flows[norm_to][1])

            masks = []
            p = []
            iterator = range(nimg)
            for i in iterator:
                outputs = dynamics.resize_and_compute_masks(
                    dP[:, i],
                    cellprob[i],
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
            all_masks[name] = masks.squeeze()
            all_p[name] = p.squeeze()
            all_dP[name] = dP.squeeze()
            all_cellprob[name] = cellprob.squeeze()
            all_flows[name] = [ plot.dx_to_circ(all_dP[name]), all_dP[name], all_cellprob[name], all_p[name] ]

        return all_masks, all_flows



