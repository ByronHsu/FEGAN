# FEGAN
## progress
- downsample flow map to 1/2
- add self-attention after upsampling
- the discriminator should classify real distorted image as false(0)
- change dis layer to 6
- add smooth gradient constraint
## todo
- Refine flow map
  - add smooth angle constraint
  - add self flow constraint(the flow map should be consistent after rotating by 90, 180, 270 degree)
  - add cross flow constraint(F(x) = F(x'), x' is rotated from x by 90 degree)
  - add inter flow constraint(the flow should point to the center)
## dataset
- MCindoor_fisheye: https://drive.google.com/drive/folders/1I9LpS8N5_z8ke_AT2zI6UVccm9vW21By?usp=sharing