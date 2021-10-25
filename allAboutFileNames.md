# Common files created/maintained/cached by data_fitter.py

* `lab_probs_seed=%d.pkl`: x->label probabilities for each attribute (whch includes label) of Labeled data (D).
* `unlab_probs.pkl`: x->label probabilities for each attribute (whch includes label) of Unlabeled data (U).
* `corrected_data.pkl`: Calibration corrected unlabeled data. This pickle file has the following structure:
    ```
        self.corrected_dataset = {
            "data": (np.array(x), np.array(y)), 
            "arm_index_to_indices": arm_idx_to_idxs, 
            "x_index_to_arm_indices": x_index_to_arm_indices,
            "noep_arm_index_to_indices": noep_arm_idx_to_idxs, 
            "noep_x_index_to_arm_indices": noep_x_index_to_arm_indices,
            "unsampled_arm_index_to_indices": unsampled_arm_idx_to_idxs,
            "unsampled_x_index_to_arm_indices": unsampled_x_index_to_arm_indices,
        }
    ```
    see `_populate_corrected_dataset` routine of data_fitter.py. `no prefix, noep, unsampled` prefix correspond to `Cal:Full, Cal:Temp, Cal:Raw` of the writeup. 
* `unlab_corrs.pkl`: Service model correctness bits for all of unlabaled data ::: Agree(S(x), y).
* `kernel.pb`, `kernel_no_edge_potential.pb`: Calibration model files corresponding to `Cal:Full`, `Cal:Temp`.

# Dataset specific cache files

## MF-CelebA (celeba.py)

* `unlab_idxs.npy`: NP array of indices in to the full CelebA data that make U.
* `arms.npy`: Cartesian space of attributes: `4096x12` -- 12 binary attributes.
* `subset_lab_idxs_seed=%d.pkl`: Subset of indices that make the Labeled dataset (for 3 seeds).

## MF-IMDB (imdb.py)

* `service_cache/mf_imdb/{lab,unlab}_{idxs,probs}` -- Labeled, unlabeled and their indices, probabilities resp. 

## AC-COCOS (cocos3.py), AC-COCOS10K (cocos3_10k.py)

* `{train,val}_{preds,ids,gt}.pkl` Predictions, ids, ground-truth for train, validation splits.  
