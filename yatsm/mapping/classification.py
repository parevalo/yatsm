""" Functions relevant for mapping categorical classification labels
"""
import logging

import numpy as np

from .utils import find_indices
from ..utils import find_results, iter_records

logger = logging.getLogger('yatsm')


def get_classification(date, result_location, image_ds,
                       after=False, before=False, qa=False,
                       pred_proba='none',
                       ndv=0, pattern='yatsm_r*', warn_on_empty=False):
    """ Output raster with classification results

    Args:
        date (int): ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before)
        pred_proba (string, optional): Include additional band(s) with
            classification value probabilities. Options are 'none' for no
            probabilities, 'max' for the probability associated with the
            assigned map label, 'all' for all class probabilities and "diff'
            for the labels and probabilities of the first and second most
            likely classes, as well as the difference in ther probabilities.
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results
        warn_on_empty (bool, optional): Log warning if result contained no
            result records (default: False)

    Returns:
        np.ndarray: 2D numpy array containing the classification map for the
            date specified

    """
    # Find results
    records = find_results(result_location, pattern)
    band_names = ['Classification']
    dtype = np.uint16
    
    # Get all possible classes. Required for 'all' and 'diff' pred-proba options
    for r in records:
        _r = np.load(r)
        if 'classes' not in _r.keys():
            continue
        else:
            classes = _r['classes']
            break

    if pred_proba == 'none':
        n_bands = 1
        dtype = np.uint8
    elif pred_proba == 'max':
        n_bands = 2
        band_names.append('Pred Proba (x10,000)')
    elif pred_proba == 'all':
        n_bands = classes.size + 1
        bnames = ['Pred Proba Class {} (x10,000)'.format(i) for i in classes] 
        band_names += bnames    
    elif pred_proba == 'diff':
        n_bands = 5
        band_names = ["Classes Set 1", "Set 1 proba", 
                      "Classes Set 2", "Set 2 proba", 'proba_diff']
    else:
        raise ValueError('Must select a valid option for prediction probability.'
                         "Options are: 'none', 'max', 'all' and 'diff'")
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=dtype) * int(ndv)

    logger.debug('Processing results')
    for rec, fname in iter_records(records, warn_on_empty=warn_on_empty,
                                   yield_filename=True):
        if 'class' not in rec.dtype.names:
            logger.warning('Results in {f} do not have classification labels'
                           .format(f=fname))
            continue
        if 'class_proba' not in rec.dtype.names and pred_proba != 'none':
            raise ValueError('Results do not have classification prediction'
                             ' probability values')

        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            raster[rec['py'][index],
                   rec['px'][index], 0] = rec['class'][index]
            if pred_proba == 'max':
                raster[rec['py'][index],
                       rec['px'][index], 1] = \
                    rec['class_proba'][index].max(axis=1) * 10000
            if pred_proba == 'all':
                raster[rec['py'][index],
                       rec['px'][index], 1:] = \
                    rec['class_proba'][index] * 10000
            if pred_proba == 'diff':
                set1_proba = rec['class_proba'][index].max(axis=1) * 10000
                set2_proba = np.partition(rec['class_proba'][index], -2)[:, -2]
                set2_ind = np.zeros(index.shape[0], dtype=np.uint16)
                for i in range(index.shape[0]):
                    set2_ind[i] = np.where(rec['class_proba'][index[i]] == set2_proba[i])[0][0]
                set2_proba *= 10000
                set2_class = classes[set2_ind]
                set_diff = set1_proba - set2_proba
                sets_array = np.stack([set1_proba, set2_class, set2_proba, 
                                       set_diff], axis=-1)
                raster[rec['py'][index],
                    rec['px'][index], 1:] = sets_array 

            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names
