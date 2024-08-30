from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import models, io
from tifffile import tifffile

# images = {
#     'CD4': img,
#     'CD8a': img2
# }
# cs = Cellamander(images)
# # Load model and get flows of all the images
# masks, flows, styles = cs.mander([{},{}]) # Run according to recipe
# cs.mander('filepath.json') # Run from stored recipe
# recipe = cs.find_recipe('path_to_rois')
# cs.save_recipe(recipe, 'where.json')


class Cellamander:
    
    def __init__(self, images, **kwargs):
        """
        Constructor 
        Additional kwargs are passed to the CellposeModel constructor

        Parameters
        --------
        images : dict
            Dictionary of numpy arrays with the channel name as the key. Should be single-
            channel images each.
        """
        self.model = models.CellposeModel(**kwargs)
        self.images = images
        # Check there are images, they are correct type

        self.flows = {}

    def _get_flows(self, **kwargs):
        for c, img in self.images.items():
            _, self.flows[c], _ = self.model.eval(
                rescale_intensity(img, out_range=(0,1)),
                **kwargs
            )
        self._blank = np.zeros_like(self.masks[c])

    def _normalize_matrix(self, m, norm_to):
        m = (m-np.min(m))/(np.max(m)-np.min(comb_vals))
        diff = (np.max(norm_to)-np.min(norm_to))

        return m*diff+np.min(norm_to)

    def mander(self, recipe, **kwargs):
        """
        Run the recipe on the loaded images

        Parameters
        --------
        recipe : str|Path|list
            The recipe to run. Can be a path to a JSON file or the loaded recipe

        """
        if len(self.flows.keys()) <= 0:
            self._get_flows(**kwargs)

        comb_vals = np.zeros_like(self._blank)
        comb_dP = np.zeros_like(self._blank)

        for c in recipe[0].keys():
            comb_vals += recipe[0][c] * self.flows[c][1]
            comb_dP += recipe[1][c] * self.flows[c][2]

        comb_vals = self._normalize_matrix(comb_vals, self.flows[recipe[2]][1])
        comb_dP = self._normalize_matrix(comb_dP, self.flows[recipe[2]][2])

        return self._make_masks([ comb_vals, comb_dP ], **kwargs)


