def separate_opt_params(model):
    # group parameters
    cnn_params = []
    rnn_params = []
    tfm_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for k, p in model.named_parameters():
        if "atst_frame" not in k:
            if "cnn" in k:
                cnn_params.append(p)
            else:
                rnn_params.append(p)
        else:
            if "blocks.0." in k:
                tfm_params[1].append(p)
            elif "blocks.1." in k:
                tfm_params[2].append(p)
            elif "blocks.2." in k:
                tfm_params[3].append(p)
            elif "blocks.3." in k:
                tfm_params[4].append(p)
            elif "blocks.4." in k:
                tfm_params[5].append(p)
            elif "blocks.5." in k:
                tfm_params[6].append(p)
            elif "blocks.6." in k:
                tfm_params[7].append(p)
            elif "blocks.7." in k:
                tfm_params[8].append(p)
            elif "blocks.8" in k:
                tfm_params[9].append(p)
            elif "blocks.9." in k:
                tfm_params[10].append(p)
            elif "blocks.10." in k:
                tfm_params[11].append(p)
            elif "blocks.11." in k:
                tfm_params[12].append(p)
            elif ".norm_frame." in k:
                tfm_params[13].append(p)
            else:
                tfm_params[0].append(p)
    return cnn_params, rnn_params, list(reversed(tfm_params))
