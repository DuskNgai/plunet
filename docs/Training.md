# Training

Unet is designed to work out-of-the-box and ideally will not require training your own model.

However, in some cases, your data may be too far out of the distribution of our training data. In this case, re-train the model using your corrected data, together with our main training dataset.

Here are some steps you can follow in order to re-train Unet:

## Step 1: Prepare your training dataset

Unet assumes a specific data structure for creating the training dataloaders:

```bash
data_dir/
├── imagesTr/       # Directory containing training images
│   ├── img1.glp.png    # Image file (currently requires .png format)
│   ├── img2.glp.png    # Image file
│   └── ...
├── imagesVal/      # Directory containing validation images
│   ├── img1.glp.png    # Image file
│   ├── img2.glp.png    # Image file
│   └── ...
├── labelsTr/       # Directory containing training labels
│   ├── label1.glpOPC.png  # Label file (currently requires .png format)
│   ├── label2.glpOPC.png  # Label file
│   └── ...
└── labelsVal/      # Directory containing validation labels
    ├── label1.glpOPC.png  # Label file
    ├── label2.glpOPC.png  # Label file
    └── ..."
```

The data_dir argument is then passed to the training procedure (see [Step 2](#step-2-perform-training)).


# Step 2: Perform training

Performing the training is simple. After activating your virtual Python environment, you can type:
```shell
plunet train
```
to receive help with the input arguments. You will see that the only parameter you need to provide is the --data-dir argument:

```shell
plunet train --data-dir <path-to-your-training-data>
```
This is exactly the folder you prepared in [Step 1](#step-1-prepare-your-training-dataset). 

Running this command should start the training and store the fully trained model in the ./checkpoint folder.

**Note:** Training can take up to several days. We therefore recommend that you perform training on a device with a CUDA-enabled GPU.
