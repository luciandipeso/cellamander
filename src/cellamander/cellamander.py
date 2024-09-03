from pathlib import Path
import numpy as np
from cellpose import models, dynamics, plot
from tifffile import tifffile
import json
from copy import deepcopy

# images = [ [[]] ] # n x r x c x channels
# cs = Cellamander()
# cs.mander(images, { {}, {} })
# cs.mander(images, [ 'CD4.json', 'CD11c.json' ])
# recipe = cs.find_recipe(images, 'path/to/rois')
# cs.save_recipe(recipe, 'path/to/recipe')
class CellamanderRecipe:

    def __init__(self, json):
        self.name = json['meta']['name']
        self.instructions = json['data']
        self._raw_json = deepcopy(json)

    def save(self):
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
            raise InvalidRecipe()

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

        for n in images.shape[0]:
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
        interp=interp,
        progress=None
    ):
        """
        Generate masks for a given image and a set of recipes

        Parameters
        --------
        images np.array An image of the format row x col x channels
        recipes list A list of CellamanderRecipe instances
        
        The remainder parameters are passed to cellpose. 

        Returns
        --------
        dict, dict  Dictionary of masks for each recipe, flows for each recipe. 
                    Keys are recipe names
        """
        flows = {}
        for c in range(images.shape[-1]):
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
                if norm_to is None and value['dP'] > 0 and value['cellprob'] > 0:
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



