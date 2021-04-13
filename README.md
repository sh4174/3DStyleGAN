## 3D-StyleGAN for medical images

3D-StyleGAN is for the generative modeling of full 3D medical images.

Please see the official repo (https://github.com/NVlabs/stylegan2) of StyleGAN2 from NVIDIA for the original code and its license. 

## Requirements 
The requirements of the original code + 
(TF 1.14 --> TF 2.4.0), Python 3.8, nibabel

## Create TFRecord with 3D Medical Images (NIFTI)

```.bash
python dataset_tool.py create_from_images3d [TFRecord_Folder/TFRecord_Name] [NIFTI Data Folder] --shuffle 1
```

## Train 3D-StyleGAN

```.bash
python run_training.py --num-gpus=4 --data-dir=[TF_Record_Folder] --config=[Training_Config] --dataset=[TFRecord_Name] --total-kimg=6000
```

[Training_Config] needs to be filled by the name of prefixed configuration in run_training.py

## Image Generation

![Uncurated Generated Images](figures/UncuratedGeneratedImages.png)

```.bash
python run_generator.py generate-images --network=[Trained_Network_Path] --seeds=66,230,389,1518,1020,11,1104,1120,1031 --truncation-psi=0.0
```

## Style Mixing 

![Style Mixing Example](figures/StyleMixing.png)

```.bash
python run_generator.py style-mixing-example --network=../trained_networks/2mm_f96.pkl --row-seeds=3181 --col-seeds=1104,1120 --truncation-psi=0.0 --col-styles=6-9
```


Please contact Sungmin Hong (HMS/MGH, shong20@mgh.harvard.edu) and Razvan Marinescu (MIT, razvan@csail.mit.edu) if you have any questions. 

